#!/usr/bin/env python3
"""Build mixed expert dataset for legacy 44-dim BC/PPO training.

Why this exists:
- Live correction data is sparse and biased.
- Pure regret data over-focuses a few hard cases.
- We need broad synthetic MILP-oracle coverage + hard-case emphasis.

Output format:
One JSON object per line with fields:
  {"state": [...44...], "action": [...6...], "reward": float, "source": str}
Compatible with:
  python3 src/drl/offline_trainer.py --from-jsonl <output>
"""

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

from src.drl.scenario_generator import ScenarioGenerator
from src.solver.milp_model import solve_placement
from src.drl.edge_env import build_mock_dataset, NODE_IDS, SERVICE_IDS
from src.drl.reward import evaluate_objective_for_placement
from src.variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS


ACTION_DIMS = [4, 4, 4, 12, 12, 4]


def _build_dataset_from_state(state: list[float]) -> Any:
    """Create reward dataset aligned with simulator/live 44-dim state layout."""
    st = np.asarray(state, dtype=np.float32)
    ds = build_mock_dataset()

    # Energy cost override from live/sim state.
    e_cpu_unit = st[8:12].tolist()
    for i, node in enumerate(ds.nodes):
        node.energy_cost = max(0.1, float(e_cpu_unit[i]))

    # Objective weights override.
    w_c = float(st[41])
    w_d = float(st[42])
    w_a = float(st[43])
    total = w_c + w_d + w_a
    if total > 0:
        ds.w_c, ds.w_d, ds.w_a = w_c / total, w_d / total, w_a / total

    # Capacity headroom override.
    cpu_util = st[0:4].tolist()
    mem_used = st[4:8].tolist()
    for i, node in enumerate(ds.nodes):
        occupied = float(cpu_util[i]) * node.cap_cpu
        node.cap_cpu = max(0.5, node.cap_cpu - occupied)
        node.cap_mem_gb = max(0.1, node.cap_mem_gb - float(mem_used[i]))

    # x_prev override from one-hot slots (state[15:39]).
    det_var_idx = int(np.argmax(st[12:15])) if st[12:15].sum() > 0 else 0
    det_var = list(DETECTION_VARIANTS)[det_var_idx]
    for k in list(ds.x_prev):
        ds.x_prev[k] = 0.0
    for svc_idx, svc_id in enumerate(SERVICE_IDS):
        slot = st[15 + svc_idx * 4: 15 + svc_idx * 4 + 4]
        node_idx = int(np.argmax(slot)) if slot.sum() > 0 else 0
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


def _milp_placement_to_action(placement: dict[str, tuple[str, str]] | dict[str, list[str]]) -> list[int] | None:
    """Convert MILP placement dict into legacy MultiDiscrete action."""
    action = [0] * 6
    for idx, svc_id in enumerate(SERVICE_IDS):
        if svc_id not in placement:
            return None
        variant, node_id = placement[svc_id]
        variant = str(variant)
        node_id = str(node_id)
        if node_id not in NODE_IDS:
            return None
        node_idx = NODE_IDS.index(node_id)
        if idx == 3:
            try:
                vi = list(DETECTION_VARIANTS).index(variant)
            except ValueError:
                vi = 0
            action[idx] = vi * 4 + node_idx
        elif idx == 4:
            try:
                vi = list(GEN_AI_VARIANTS).index(variant)
            except ValueError:
                vi = 0
            action[idx] = vi * 4 + node_idx
        else:
            action[idx] = node_idx
    for i, dim in enumerate(ACTION_DIMS):
        if action[i] < 0 or action[i] >= dim:
            return None
    return action


def _action_to_placement(action: list[int]) -> dict[str, tuple[str, str]]:
    a = [int(x) for x in action]
    def _vn(x: int, variants: list[str]) -> tuple[str, str]:
        vi = max(0, min(len(variants) - 1, x // 4))
        ni = max(0, min(3, x % 4))
        return variants[vi], NODE_IDS[ni]

    det_var, det_node = _vn(a[3], list(DETECTION_VARIANTS))
    gen_var, gen_node = _vn(a[4], list(GEN_AI_VARIANTS))
    return {
        "m0": ("standard", NODE_IDS[min(3, a[0])]),
        "m1": ("standard", NODE_IDS[min(3, a[1])]),
        "m2": ("standard", NODE_IDS[min(3, a[2])]),
        "m3": (det_var, det_node),
        "m4": (gen_var, gen_node),
        "m5": ("standard", NODE_IDS[min(3, a[5])]),
    }


def _reward_from_state_action(state: list[float], action: list[int]) -> float:
    ds = _build_dataset_from_state(state)
    placement = _action_to_placement(action)
    try:
        br = evaluate_objective_for_placement(ds, placement)
        return float(br.reward)
    except Exception:
        return 0.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _collect_from_regret(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        for row in _load_jsonl(p):
            state = row.get("state")
            action = row.get("action")
            if not isinstance(state, list) or not isinstance(action, list):
                continue
            if len(state) != 44 or len(action) != 6:
                continue
            try:
                rec = {
                    "state": [float(x) for x in state],
                    "action": [int(x) for x in action],
                    "reward": float(row.get("reward", _reward_from_state_action(state, action))),
                    "source": "regret",
                }
                out.append(rec)
            except Exception:
                continue
    return out


def _collect_from_live_corrections(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        for row in _load_jsonl(p):
            state = row.get("state")
            projected_action = row.get("projected_action")
            if not isinstance(state, list) or not isinstance(projected_action, list):
                continue
            if len(state) != 44 or len(projected_action) != 6:
                continue
            try:
                reward = _reward_from_state_action(state, projected_action)
                out.append(
                    {
                        "state": [float(x) for x in state],
                        "action": [int(x) for x in projected_action],
                        "reward": reward,
                        "source": "live_correction",
                    }
                )
            except Exception:
                continue
    return out


def _collect_synthetic(samples: int, seed: int, milp_time_limit: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    gen = ScenarioGenerator(seed=seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < samples and attempts < samples * 8:
        attempts += 1
        state = gen.sample_state().astype(np.float32)
        ds = _build_dataset_from_state(state.tolist())
        try:
            res = solve_placement(ds, time_limit=milp_time_limit, verbose=False)
        except Exception:
            continue
        if res is None or res.status not in {"optimal", "feasible"}:
            continue
        act = _milp_placement_to_action(res.placement)
        if act is None:
            continue
        rows.append(
            {
                "state": state.tolist(),
                "action": act,
                "reward": float(getattr(res, "reward", -float(res.objective_value))),
                "source": f"synthetic_{rng.randint(0, 99)}",
            }
        )
    return rows


def _sample_with_replacement(rng: random.Random, rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not rows or k <= 0:
        return []
    if k <= len(rows):
        return rng.sample(rows, k)
    return [rows[rng.randrange(0, len(rows))] for _ in range(k)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build mixed expert dataset for DRL BC/PPO")
    ap.add_argument("--output", default="results/drl_mixed_expert_v8.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-size", type=int, default=6000)
    ap.add_argument("--synthetic-count", type=int, default=4500)
    ap.add_argument("--milp-time-limit", type=int, default=10)
    ap.add_argument(
        "--regret-jsonl",
        nargs="*",
        default=["results/drl_expert_from_regret.jsonl", "results/drl_expert_from_regret_v7.jsonl"],
    )
    ap.add_argument(
        "--live-corrections-jsonl",
        nargs="*",
        default=["results/storm_corrections_live.jsonl"],
    )
    ap.add_argument("--regret-share", type=float, default=0.35, help="share of target-size")
    ap.add_argument("--live-share", type=float, default=0.20, help="share of target-size")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_rows = _collect_synthetic(args.synthetic_count, seed=args.seed, milp_time_limit=args.milp_time_limit)
    regret_rows = _collect_from_regret([Path(p) for p in args.regret_jsonl])
    live_rows = _collect_from_live_corrections([Path(p) for p in args.live_corrections_jsonl])

    n_regret = max(0, int(args.target_size * args.regret_share))
    n_live = max(0, int(args.target_size * args.live_share))
    n_synth = max(0, args.target_size - n_regret - n_live)

    mixed: list[dict[str, Any]] = []
    mixed.extend(_sample_with_replacement(rng, synthetic_rows, n_synth))
    mixed.extend(_sample_with_replacement(rng, regret_rows, n_regret))
    mixed.extend(_sample_with_replacement(rng, live_rows, n_live))
    rng.shuffle(mixed)

    with out_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    src_counts: dict[str, int] = {}
    for row in mixed:
        src = str(row.get("source", "unknown"))
        src_counts[src] = src_counts.get(src, 0) + 1

    print(
        json.dumps(
            {
                "output": str(out_path),
                "target_size": args.target_size,
                "written": len(mixed),
                "pool_sizes": {
                    "synthetic_pool": len(synthetic_rows),
                    "regret_pool": len(regret_rows),
                    "live_pool": len(live_rows),
                },
                "sampled_counts": {
                    "synthetic": n_synth,
                    "regret": n_regret,
                    "live": n_live,
                },
                "source_breakdown": src_counts,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
