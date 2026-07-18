#!/usr/bin/env python3
"""Regret-case distiller: convert drl_regret_cases.jsonl into MILP expert trajectories.

Reads the worst-case DRL-vs-MILP gap records produced by
  run_offline_comparison.py --dump-state-action
and converts each saved MILP placement into a DRL MultiDiscrete action,
then computes the MILP reward via evaluate_objective_for_placement().

Output is a JSONL file consumable by:
  python3 src/drl/offline_trainer.py --from-jsonl results/drl_expert_from_regret.jsonl

Typical usage
─────────────
  # 1. Collect regret cases
  python3 src/experiments/run_offline_comparison.py \\
      --cycles 500 --modes milp drl --dump-state-action

  # 2. Distill expert trajectories
  python3 src/drl/regret_distiller.py

  # 3. BC warm-start from expert data
  python3 src/drl/offline_trainer.py \\
      --from-jsonl results/drl_expert_from_regret.jsonl \\
      --output src/drl/models/bc_adaptive.zip --smoke

  # 4. PPO fine-tune with curriculum
  python3 src/drl/train_simulator.py \\
      --bc-checkpoint src/drl/models/bc_adaptive.zip \\
      --use-scenarios --curriculum-stage 4 --scenario-profile energy \\
      --model-path src/drl/models/ppo_adaptive.zip --timesteps 200000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from drl.edge_env import NODE_IDS, SERVICE_IDS, build_mock_dataset  # noqa: E402
from drl.reward import evaluate_objective_for_placement              # noqa: E402

log = logging.getLogger("regret-distiller")

# ── Action encoding constants ─────────────────────────────────────────────────
_DETECTION_VARIANTS = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
_GEN_AI_VARIANTS    = ["qwen-1.5b-nano", "llama-3b-small", "gemma2-2b-medium"]
_NODE_IDS           = list(NODE_IDS)   # ["n0", "n1", "n2", "n3"]
_SVC_ORDER          = list(SERVICE_IDS)  # ["m0", "m1", "m2", "m3", "m4", "m5"]
_ACTION_DIMS        = [4, 4, 4, 12, 12, 4]


def _placement_to_action(placement: dict[str, Any]) -> list[int] | None:
    """Convert a MILP placement dict → DRL MultiDiscrete action.

    Placement format (from regret case milp_placement field):
        {"m0": ["standard", "n2"], "m3": ["yolo26-nano", "n0"], ...}
    or equivalently with tuple values:
        {"m0": ("standard", "n2"), ...}

    Returns None if any service is missing or uses an unknown node/variant.
    """
    action = [0] * 6
    for svc_idx, svc_id in enumerate(_SVC_ORDER):
        entry = placement.get(svc_id)
        if entry is None:
            log.warning("Missing service %s in placement — skipping case", svc_id)
            return None
        variant, node_id = str(entry[0]), str(entry[1])
        if node_id not in _NODE_IDS:
            log.warning("Unknown node_id '%s' for service %s — skipping case", node_id, svc_id)
            return None
        node_idx = _NODE_IDS.index(node_id)

        if svc_idx == 3:   # m3 detection variant
            if variant not in _DETECTION_VARIANTS:
                log.warning("Unknown det variant '%s' — using nano", variant)
                variant = _DETECTION_VARIANTS[0]
            action[svc_idx] = _DETECTION_VARIANTS.index(variant) * 4 + node_idx
        elif svc_idx == 4:  # m4 gen-ai variant
            if variant not in _GEN_AI_VARIANTS:
                log.warning("Unknown gen-ai variant '%s' — using nano", variant)
                variant = _GEN_AI_VARIANTS[0]
            action[svc_idx] = _GEN_AI_VARIANTS.index(variant) * 4 + node_idx
        else:
            action[svc_idx] = node_idx

    # Validate bounds
    for i, (a, dim) in enumerate(zip(action, _ACTION_DIMS)):
        if a < 0 or a >= dim:
            log.warning("Action[%d]=%d out of range [0, %d) — skipping case", i, a, dim)
            return None

    return action


def _compute_milp_reward(state: list[float], placement: dict[str, Any]) -> float:
    """Evaluate reward = -J for a MILP placement given the scenario state."""
    state_arr = np.asarray(state, dtype=np.float32)
    ds = build_mock_dataset()

    # Override energy costs from state[8:12]
    nodes_sorted = sorted(ds.nodes, key=lambda n: n.node_id)
    for i, node in enumerate(nodes_sorted[:4]):
        node.energy_cost = max(0.1, float(state_arr[8 + i]))

    # Override objective weights from state[41:44]
    w_c, w_d, w_a = float(state_arr[41]), float(state_arr[42]), float(state_arr[43])
    total = w_c + w_d + w_a
    if total > 0:
        ds.w_c, ds.w_d, ds.w_a = w_c / total, w_d / total, w_a / total

    # Build placement dict in the format evaluate_objective_for_placement expects
    typed_placement = {
        svc_id: (str(entry[0]), str(entry[1]))
        for svc_id, entry in placement.items()
    }

    try:
        obj = evaluate_objective_for_placement(ds, typed_placement)
        return float(obj.reward)
    except Exception as exc:
        log.warning("Reward evaluation failed: %s", exc)
        return 0.0


def distill(
    input_path: Path,
    output_path: Path,
    min_gap_pct: float,
    hard_scenarios_only: bool,
    hard_scenario_list: list[str],
) -> int:
    """Main distillation loop.

    Returns the number of expert trajectories written.
    """
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        log.error("Run: python3 src/experiments/run_offline_comparison.py --dump-state-action")
        return 0

    cases: list[dict] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    log.info("Loaded %d regret cases from %s", len(cases), input_path)

    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_fh:
        for i, case in enumerate(cases):
            # ── Filter by gap threshold ───────────────────────────────────────
            gap_pct = case.get("gap_pct")
            if gap_pct is None or gap_pct < min_gap_pct:
                continue

            # ── Filter by hard scenario ───────────────────────────────────────
            scenario = case.get("scenario", "")
            if hard_scenarios_only and scenario not in hard_scenario_list:
                continue

            state = case.get("state")
            milp_placement = case.get("milp_placement")
            if not state or not milp_placement:
                log.debug("Case %d missing state or milp_placement — skipping", i)
                continue

            if len(state) != 44:
                log.debug("Case %d state dim=%d != 44 — skipping", i, len(state))
                continue

            action = _placement_to_action(milp_placement)
            if action is None:
                continue

            reward = _compute_milp_reward(state, milp_placement)

            record = {
                "state":    state,
                "action":   action,
                "reward":   round(reward, 6),
                "scenario": scenario,
                "gap_pct":  gap_pct,
            }
            out_fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1

    log.info(
        "Distilled %d / %d cases → %s  (min_gap_pct=%.1f%%)",
        written, len(cases), output_path, min_gap_pct,
    )
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert drl_regret_cases.jsonl → MILP expert trajectory JSONL for BC training."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "results" / "drl_regret_cases.jsonl"),
        help="Input JSONL from run_offline_comparison.py --dump-state-action "
             "(default: results/drl_regret_cases.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "drl_expert_from_regret.jsonl"),
        help="Output expert trajectory JSONL (default: results/drl_expert_from_regret.jsonl)",
    )
    parser.add_argument(
        "--min-gap-pct",
        type=float,
        default=0.0,
        help="Only distill cases where DRL gap ≥ this %% (default: 0.0 = all positive-gap cases).",
    )
    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="Only distill cases from hard scenarios "
             "(energy_saving, storm_test, energy_inversion, cascade_failure, memory_pressure, n3_saturated).",
    )
    parser.add_argument(
        "--hard-scenarios",
        nargs="+",
        default=[
            "energy_saving", "storm_test", "energy_inversion",
            "cascade_failure", "memory_pressure", "n3_saturated",
        ],
        help="List of scenario names considered 'hard' (used with --hard-only).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    n = distill(
        input_path=Path(args.input),
        output_path=Path(args.output),
        min_gap_pct=args.min_gap_pct,
        hard_scenarios_only=args.hard_only,
        hard_scenario_list=args.hard_scenarios,
    )
    print(f"Expert trajectories written: {n}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
