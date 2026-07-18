#!/usr/bin/env python3
"""Build storm-heavy 44-dim expert dataset for legacy MultiDiscrete retraining."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.drl.scenario_generator import ScenarioGenerator, ScenarioType
from src.drl.edge_env import NODE_IDS, SERVICE_IDS, build_mock_dataset
from src.drl.reward import evaluate_objective_for_placement
from src.solver.milp_model import solve_placement
from src.variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS


ACTION_DIMS = [4, 4, 4, 12, 12, 4]
DEFAULT_SCENARIOS = [
    ScenarioType.STORM_TEST,
    ScenarioType.MEMORY_PRESSURE,
    ScenarioType.NODE_FAILURE,
    ScenarioType.CASCADE_FAILURE,
]


def _build_dataset_from_state(state: np.ndarray):
    st = np.asarray(state, dtype=np.float32).reshape(-1)
    ds = build_mock_dataset()

    e_cpu_unit = st[8:12].tolist()
    for i, node in enumerate(ds.nodes):
        node.energy_cost = max(0.1, float(e_cpu_unit[i]))

    w_c = float(st[41])
    w_d = float(st[42])
    w_a = float(st[43])
    total = w_c + w_d + w_a
    if total > 0:
        ds.w_c, ds.w_d, ds.w_a = w_c / total, w_d / total, w_a / total

    cpu_util = st[0:4].tolist()
    mem_used = st[4:8].tolist()
    for i, node in enumerate(ds.nodes):
        occupied = float(cpu_util[i]) * node.cap_cpu
        node.cap_cpu = max(0.5, node.cap_cpu - occupied)
        node.cap_mem_gb = max(0.1, node.cap_mem_gb - float(mem_used[i]))

    det_var_idx = int(np.argmax(st[12:15])) if float(np.sum(st[12:15])) > 0.0 else 0
    det_var = list(DETECTION_VARIANTS)[det_var_idx]
    for k in list(ds.x_prev):
        ds.x_prev[k] = 0.0
    for svc_idx, svc_id in enumerate(SERVICE_IDS):
        slot = st[15 + svc_idx * 4: 15 + svc_idx * 4 + 4]
        node_idx = int(np.argmax(slot)) if float(np.sum(slot)) > 0.0 else 0
        node_id = NODE_IDS[node_idx]
        if svc_id == "m3":
            var = det_var
        elif svc_id == "m4":
            var = list(GEN_AI_VARIANTS)[0]
        else:
            var = "standard"
        if (svc_id, var, node_id) in ds.x_prev:
            ds.x_prev[(svc_id, var, node_id)] = 1.0
    return ds


def _placement_to_action(placement: dict[str, tuple[str, str]]) -> list[int] | None:
    action = [0] * 6
    for idx, svc_id in enumerate(SERVICE_IDS):
        if svc_id not in placement:
            return None
        variant, node_id = placement[svc_id]
        if node_id not in NODE_IDS:
            return None
        node_idx = NODE_IDS.index(node_id)
        if idx == 3:
            vi = list(DETECTION_VARIANTS).index(variant) if variant in DETECTION_VARIANTS else 0
            action[idx] = vi * 4 + node_idx
        elif idx == 4:
            vi = list(GEN_AI_VARIANTS).index(variant) if variant in GEN_AI_VARIANTS else 0
            action[idx] = vi * 4 + node_idx
        else:
            action[idx] = node_idx
    for i, dim in enumerate(ACTION_DIMS):
        if action[i] < 0 or action[i] >= dim:
            return None
    return action


def _sample_scenario(
    rng: random.Random,
    scenarios: list[ScenarioType],
    weights: list[float],
) -> ScenarioType:
    x = rng.random()
    acc = 0.0
    for s, w in zip(scenarios, weights):
        acc += w
        if x <= acc:
            return s
    return scenarios[-1]


def _migration_count(ds, placement: dict[str, tuple[str, str]]) -> int:
    return int(
        sum(
            1
            for svc, (var, node) in placement.items()
            if float(ds.x_prev.get((svc, var, node), 0.0)) < 0.5
        )
    )


def _parse_scenarios(raw: str) -> tuple[list[ScenarioType], list[float]]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    scenarios: list[ScenarioType] = []
    weights: list[float] = []
    for p in parts:
        if ":" not in p:
            continue
        name, w = p.split(":", 1)
        scenarios.append(ScenarioType(name.strip()))
        weights.append(max(0.0, float(w)))
    if not scenarios:
        scenarios = list(DEFAULT_SCENARIOS)
        weights = [1.0] * len(scenarios)
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0 / len(scenarios)] * len(scenarios)
    else:
        weights = [w / total for w in weights]
    return scenarios, weights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="results/drl_44_stormsafe_expert.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-size", type=int, default=7000)
    ap.add_argument("--max-attempt-multiplier", type=int, default=10)
    ap.add_argument("--milp-time-limit", type=int, default=20)
    ap.add_argument(
        "--scenario-weights",
        default="storm_test:0.50,memory_pressure:0.20,node_failure:0.15,cascade_failure:0.15",
        help="Comma list: scenario:weight",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    gen = ScenarioGenerator(seed=args.seed)
    scenarios, weights = _parse_scenarios(args.scenario_weights)

    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(args.target_size, 1) * max(args.max_attempt_multiplier, 1)
    while len(rows) < args.target_size and attempts < max_attempts:
        attempts += 1
        scenario = _sample_scenario(rng, scenarios, weights)
        state = gen.sample_state(scenario=scenario).astype(np.float32)
        ds = _build_dataset_from_state(state)
        try:
            res = solve_placement(ds, time_limit=args.milp_time_limit, verbose=False)
        except Exception:
            continue
        if res is None or res.status not in {"optimal", "feasible"}:
            continue
        action = _placement_to_action(res.placement)
        if action is None:
            continue
        try:
            br = evaluate_objective_for_placement(ds, res.placement)
            objective_j = float(br.objective_j)
            reward = float(br.reward)
        except Exception:
            objective_j = float(getattr(res, "objective_value", 0.0))
            reward = -objective_j
        mig = _migration_count(ds, res.placement)
        rows.append(
            {
                "state": state.tolist(),
                "action": action,
                "reward": reward,
                "source": "milp_oracle_stormsafe",
                "scenario": scenario.value,
                "migration_count": int(mig),
                "storm_max": int(ds.v_storm_max),
                "objective_J": objective_j,
            }
        )

    # Shuffle for BC data loader robustness
    np_rng.shuffle(rows)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    by_scenario: dict[str, int] = {}
    for row in rows:
        s = str(row.get("scenario", "unknown"))
        by_scenario[s] = by_scenario.get(s, 0) + 1

    print(
        json.dumps(
            {
                "output": str(out_path),
                "target_size": int(args.target_size),
                "written": len(rows),
                "attempts": int(attempts),
                "scenarios": [s.value for s in scenarios],
                "scenario_weights": weights,
                "scenario_counts": by_scenario,
            },
            ensure_ascii=False,
        )
    )
    return 0 if len(rows) >= args.target_size else 1


if __name__ == "__main__":
    raise SystemExit(main())
