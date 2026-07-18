"""
run_scenario_eval.py

Evaluates MILP and DRL+BC on each scenario_*.json file.

Strategy
--------
The trained DRL model expects a fixed 44-dim obs (4 nodes, 6 services).
The scenario files have 3 nodes and 5 services. To make a fair comparison
both solvers are evaluated on the **same** problem instance: the EdgeEnv
mock topology (4 nodes, 6 services) with the scenario's parameters injected
(node energy costs, objective weights w_c/w_d/w_a, theta_max, v_storm_max).

For each scenario file:
  1. Build a scenario-adapted dataset (mock topology + scenario params).
  2. MILP: solve_placement on that dataset → milp_objective.
  3. DRL : load PPO model, build matching state vector, run policy → placement
           → evaluate_objective_for_placement → drl_objective.
  4. Write milp_objective + drl_objective back into the JSON file.

Usage
-----
  python scripts/run_scenario_eval.py [--dry-run]

Dependencies (already in project venv):
  pyomo, highspy, stable-baselines3, sb3-contrib, gymnasium, torch
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "solver"))
sys.path.insert(0, str(SRC / "drl"))

from solver.dataset_generator import (
    MILPDataset, NodeSpec, ServiceSpec,
    dataset_from_dict,
)
from solver.milp_model import solve_placement
from drl.reward import evaluate_objective_for_placement
from drl.edge_env import (
    EdgeEnv, build_mock_dataset,
    OBS_DIM, NODE_IDS, SERVICE_IDS,
)
from config import MILP_W_C, MILP_W_D, MILP_W_A


# ── constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR  = ROOT / "results"
MODEL_PATH   = SRC / "drl" / "models" / "ppo_scenarios.zip"
SCENARIO_TYPES = [
    "energy_saving",
    "node_migration",
    "quality_maximise",
    "ram_pressure",
    "storm_test",
]

# Default CPU utilisation / mem used injected into mock state when building
# the scenario-adapted observation (realistic idle-ish values).
_DEFAULT_CPU_UTIL = [0.30, 0.30, 0.30, 0.10]   # fraction 0–1
_DEFAULT_MEM_USED = [1.20, 1.20, 2.40, 0.50]   # GB


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_scenario_dataset(scenario: dict, mock_ds: MILPDataset) -> MILPDataset:
    """Build an evaluation dataset from mock topology + scenario parameters.

    Node energy costs are taken from the scenario file (up to 3 nodes;
    the 4th mock node keeps its calibrated value). Weights, theta_max, and
    v_storm_max come from the scenario file.
    """
    scenario_nodes = scenario.get("nodes", [])
    e_cpu_override = [float(n["energy_cost"]) for n in scenario_nodes]

    updated_nodes: list[NodeSpec] = []
    for i, nd in enumerate(mock_ds.nodes):
        e_cpu = e_cpu_override[i] if i < len(e_cpu_override) else nd.energy_cost
        updated_nodes.append(NodeSpec(
            node_id=nd.node_id,
            cap_cpu=nd.cap_cpu,
            cap_mem_gb=nd.cap_mem_gb,
            energy_cost=e_cpu,
            e_mem_unit=nd.e_mem_unit,
        ))

    return MILPDataset(
        nodes=updated_nodes,
        services=mock_ds.services,
        r_req=mock_ds.r_req,
        r_mem=mock_ds.r_mem,
        acc=mock_ds.acc,
        x_prev=mock_ds.x_prev,
        theta_max=float(scenario.get("theta_max", mock_ds.theta_max)),
        v_storm_max=int(scenario.get("v_storm_max", mock_ds.v_storm_max)),
        w_c=float(scenario.get("w_c", MILP_W_C)),
        w_d=float(scenario.get("w_d", MILP_W_D)),
        w_a=float(scenario.get("w_a", MILP_W_A)),
    )


def _build_state_for_dataset(ds: MILPDataset) -> np.ndarray:
    """Build the 44-dim observation for the PPO model that encodes ds parameters.

    Layout (matches EdgeEnv._build_live_state):
      [0:4]   cpu_util
      [4:8]   mem_used_gb
      [8:12]  e_cpu_unit  ← from ds.nodes
      [12:15] detection variant one-hot (default: first = nano)
      [15:39] placement one-hot 6×4 (default: all on n0)
      [39]    e2e_latency_ms
      [40]    migrations
      [41:44] w_c, w_d, w_a  ← from ds weights
    """
    state = np.zeros(OBS_DIM, dtype=np.float32)

    # cpu utilisation [0:4]
    for i, v in enumerate(_DEFAULT_CPU_UTIL):
        state[i] = float(v)

    # mem used GB [4:8]
    for i, v in enumerate(_DEFAULT_MEM_USED):
        state[4 + i] = float(v)

    # e_cpu unit [8:12]
    for i, nd in enumerate(ds.nodes):
        state[8 + i] = float(nd.energy_cost)

    # detection one-hot [12:15]: nano=index 0
    state[12] = 1.0

    # placement one-hot [15:39]: all services on n0 (index 0)
    for svc_idx in range(len(SERVICE_IDS)):
        state[15 + svc_idx * 4 + 0] = 1.0  # n0

    # e2e latency [39]
    state[39] = 300.0

    # migrations [40]
    state[40] = 0.0

    # weights [41:44]
    state[41] = float(ds.w_c)
    state[42] = float(ds.w_d)
    state[43] = float(ds.w_a)

    return state


def eval_milp(ds: MILPDataset, verbose: bool = False) -> float | None:
    """Run MILP solver on ds. Returns objective_J or None on failure."""
    result = solve_placement(ds, time_limit=60, verbose=verbose)
    if result is None:
        return None
    return float(result.objective_value)


def eval_drl(ds: MILPDataset, model, n_samples: int = 30) -> float:
    """Run PPO model n_samples times and return mean objective_J.

    We inject the scenario state, collect actions, and evaluate placements
    against the scenario-adapted dataset.
    """
    state = _build_state_for_dataset(ds)
    obs   = state.reshape(1, -1)
    tmp_env = EdgeEnv(mock=True)

    j_vals: list[float] = []
    for _ in range(n_samples):
        tmp_env._state = state.copy()
        action, _ = model.predict(obs, deterministic=False)
        placement  = tmp_env._action_to_placement(np.asarray(action[0]))

        breakdown = evaluate_objective_for_placement(ds, placement)
        j_vals.append(breakdown.objective_j)

    return float(np.mean(j_vals))


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    print("=" * 60)
    print("Scenario Stress-Test Evaluation")
    print(f"  Model : {MODEL_PATH}")
    print(f"  Dry   : {dry_run}")
    print("=" * 60)

    # ── load PPO model ────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"[ERROR] PPO model not found: {MODEL_PATH}")
        sys.exit(1)

    from stable_baselines3 import PPO
    print(f"\nLoading PPO model from {MODEL_PATH.name} …")
    model = PPO.load(str(MODEL_PATH))
    print("  OK")

    mock_ds = build_mock_dataset()

    # ── process each scenario type ────────────────────────────────────────────
    for stype in SCENARIO_TYPES:
        files = sorted(RESULTS_DIR.glob(f"scenario_{stype}_*.json"))
        if not files:
            print(f"\n[SKIP] No files for {stype}")
            continue

        print(f"\n── {stype.upper()} ({len(files)} files) ──")

        for fp in files:
            scenario = json.loads(fp.read_text())

            # Skip if already evaluated
            if "milp_objective" in scenario and "drl_objective" in scenario:
                print(f"  {fp.name}  [already done, skipping]")
                continue

            # Build scenario-adapted dataset (mock topology + scenario params)
            eval_ds = _build_scenario_dataset(scenario, mock_ds)

            # ── MILP ──────────────────────────────────────────────────────────
            milp_j = eval_milp(eval_ds, verbose=False)
            if milp_j is None:
                print(f"  {fp.name}  MILP infeasible — skipping")
                continue

            # ── DRL ───────────────────────────────────────────────────────────
            drl_j = eval_drl(eval_ds, model, n_samples=30)

            print(f"  {fp.name}  milp_J={milp_j:.4f}  drl_J={drl_j:.4f}  "
                  f"gap={100*(drl_j - milp_j)/abs(milp_j):+.1f}%")

            if not dry_run:
                scenario["milp_objective"] = milp_j
                scenario["drl_objective"]  = drl_j
                fp.write_text(json.dumps(scenario, indent=2))

    print("\nDone." if not dry_run else "\nDry-run complete — files not modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MILP+DRL on scenario files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write results to files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
