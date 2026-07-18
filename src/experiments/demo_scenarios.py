"""
demo_scenarios.py — Prove MILP and DRL+BC make sensible decisions across
10 diverse real-world scenarios.

What this script does
─────────────────────
For each of the 10 ScenarioGenerator scenarios it:
  1. Builds a synthetic cluster state (cpu_util, RAM, energy costs per node)
     that reflects the scenario (e.g., n3 overloaded, n0 on cheap power, …)
  2. Runs the real MILP solver (solve_placement) on that state
  3. Runs the DRL+BC model (PPO.predict) on the same state
  4. Prints a side-by-side comparison

Reading the output
──────────────────
  * MILP placement: which node (n0–n3) each of the 6 services lands on
  * J (objective): higher is better (energy-saving + accuracy - disruption)
  * Solve time: how long MILP took vs DRL's near-instant inference
  * When MILP puts services on non-n3 nodes it proves the solver is making
    non-trivial, scenario-aware decisions rather than always picking n3.

Run
───
    cd /home/ubuntu/KLTN_project
    .venv/bin/python -m src.experiments.demo_scenarios
"""
from __future__ import annotations

import sys
import os
import time

# Make solver importable
_src = os.path.join(os.path.dirname(__file__), "..", "..")
_solver = os.path.join(_src, "src", "solver")
for _p in [_src, _solver, os.path.join(_src, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# ── Internal imports ──────────────────────────────────────────────────────────
from src.drl.scenario_generator import ScenarioGenerator, ScenarioType
from src.solver.dataset_generator import (
    MILPDataset, NodeSpec, ServiceSpec,
    DEFAULT_SERVICE_TYPE_POOL,
)
from src.solver.milp_model import solve_placement

# ── Real cluster constants ────────────────────────────────────────────────────
_NODE_IDS    = ["edge-nodes-1", "edge-nodes-2", "edge-nodes-3", "edge-nodes-4"]
_NODE_CAP_CPU= [4.0, 2.0, 4.0, 4.0]   # total allocatable vCPU
_NODE_CAP_MEM= [3.82, 3.82, 7.75, 7.75]   # total allocatable GB RAM
_NODE_SHORT  = ["n0", "n1", "n2", "n3"]

# Service definitions — must match live cluster
_SVC_IDS   = ["m0", "m1", "m2", "m3", "m4", "m5"]
_SVC_TYPES = ["gateway", "ingest", "preprocess", "detection", "gen_ai", "postprocess"]
_MIG_COSTS = [1.0, 1.0, 1.5, 3.0, 4.0, 1.0]   # disruption cost T[m]

# Variant resource requirements: (cpu_cores, ram_gb, quality_score)
_VARIANT_SPECS: dict[str, dict[str, tuple[float, float, float]]] = {
    "standard":       {"standard":       (0.10, 0.10, 1.00)},
    "gateway":        {"standard":       (0.10, 0.10, 1.00)},
    "ingest":         {"standard":       (0.10, 0.10, 1.00)},
    "preprocess":     {"standard":       (0.20, 0.20, 1.00)},
    "postprocess":    {"standard":       (0.10, 0.10, 1.00)},
    "detection":      {"yolo26-nano":    (0.25, 0.30, 0.75),
                       "yolo26-small":   (0.50, 0.50, 0.85),
                       "yolo26-medium":  (1.00, 0.80, 0.95)},
    "gen_ai":         {"light":          (0.50, 1.00, 0.80),
                       "heavy":          (1.50, 2.00, 0.95)},
}


def _build_milp_dataset(state: np.ndarray) -> MILPDataset:
    """Convert a 44-dim scenario state into a MILPDataset for solve_placement().

    State layout (from scenario_generator.py):
      [0:4]   cpu_util per node (fraction 0-1)
      [4:8]   mem_gb used per node
      [8:12]  e_cpu_unit W/core per node
      [15:39] previous placement one-hot (6 svcs × 4 nodes)
      [41:44] w_c, w_d, w_a
    """
    # ── Nodes: available capacity = total - currently used ───────────────────
    nodes = []
    for i in range(4):
        cpu_util  = float(np.clip(state[i], 0.0, 0.98))
        mem_used  = float(max(0.0, state[4 + i]))
        e_cpu     = float(max(0.5, state[8 + i]))   # W/core
        cpu_avail = max(0.05, _NODE_CAP_CPU[i] * (1.0 - cpu_util))
        mem_avail = max(0.05, _NODE_CAP_MEM[i] - mem_used)
        e_mem     = round(max(0.1, 18.7 / max(_NODE_CAP_MEM[i], 1.0)), 4)
        nodes.append(NodeSpec(
            node_id=_NODE_IDS[i],
            cap_cpu=round(cpu_avail, 2),
            cap_mem_gb=round(mem_avail, 2),
            energy_cost=round(e_cpu, 3),
            e_mem_unit=e_mem,
        ))

    # ── Services: fixed real-world specs ────────────────────────────────────
    services = []
    r_req: dict[tuple[str, str], float] = {}
    r_mem: dict[tuple[str, str], float] = {}
    acc:   dict[tuple[str, str], float] = {}

    for j, (svc_id, svc_type, mig) in enumerate(
        zip(_SVC_IDS, _SVC_TYPES, _MIG_COSTS)
    ):
        pool = DEFAULT_SERVICE_TYPE_POOL.get(svc_type, ["standard"])
        services.append(ServiceSpec(
            service_id=svc_id,
            service_type=svc_type,
            migration_cost=mig,
            valid_variants=pool,
        ))
        variant_map = _VARIANT_SPECS.get(svc_type, _VARIANT_SPECS["standard"])
        for var in pool:
            cpu_r, mem_r, q = variant_map.get(var, (0.10, 0.10, 1.0))
            r_req[(svc_id, var)] = cpu_r
            r_mem[(svc_id, var)] = mem_r
            acc[(svc_id, var)]   = q

    # ── Previous placement: set to all-zeros for demo
    # (avoids migration-cost bias from the hardcoded all-n0 prior in _base_state,
    #  so the demo shows the pure energy + accuracy placement preference)
    x_prev: dict[tuple[str, str, str], float] = {}
    for j, svc in enumerate(services):
        for var in svc.valid_variants:
            for nid in _NODE_IDS:
                x_prev[(svc.service_id, var, nid)] = 0.0

    w_c = float(state[41])
    w_d = float(state[42])
    w_a = float(state[43])

    return MILPDataset(
        nodes=nodes, services=services,
        r_req=r_req, r_mem=r_mem, acc=acc, x_prev=x_prev,
        theta_max=1.3,
        # x_prev is all-zeros (fresh deployment) so every service counts as
        # "migrating" from a null prior.  Set v_storm_max = num_services to
        # disable the migration-storm constraint in the demo.
        v_storm_max=len(services),
        w_c=w_c, w_d=w_d, w_a=w_a,
    )


def _drl_predict(state: np.ndarray, model_path: str) -> list[int] | None:
    """Return DRL+BC placement as list of node indices [0-3] for m0..m5."""
    try:
        from sb3_contrib import MaskablePPO
        from src.drl.edge_env import compute_action_masks
        model = MaskablePPO.load(model_path)
        obs = state.reshape(1, -1)
        masks = compute_action_masks(state).reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        return [int(a) for a in action.flatten()]
    except Exception as exc:
        print(f"  [DRL] load/predict failed: {exc}")
        return None


def _fmt_placement(placement: dict[str, tuple[str, str]] | None,
                    action: list[int] | None) -> tuple[str, str]:
    """Format MILP and DRL placements as compact strings like 'n3 n3 n2 n2 n3 n3'."""
    if placement:
        milp_str = " ".join(
            _NODE_SHORT[_NODE_IDS.index(placement[svc][1])]
            if svc in placement and placement[svc][1] in _NODE_IDS else "??"
            for svc in _SVC_IDS
        )
    else:
        milp_str = "INFEASIBLE"

    if action:
        drl_str = " ".join(_NODE_SHORT[a % 4] for a in action)
    else:
        drl_str = "N/A      "

    return milp_str, drl_str


def run_demo(model_path: str = "src/drl/models/ppo_simulator.zip") -> None:
    gen = ScenarioGenerator(seed=42)

    header = (
        f"\n{'═'*90}\n"
        f"  KLTN Edge Placement Demo  —  10 Diverse Scenarios  (K=4 nodes, M=6 services)\n"
        f"{'═'*90}\n"
        f"  Nodes:    n0 ({_NODE_CAP_CPU[0]}vCPU/{_NODE_CAP_MEM[0]}GB  {19.87:.0f}W/core)  "
        f"n1 ({_NODE_CAP_CPU[1]}vCPU/{_NODE_CAP_MEM[1]}GB  {41.02:.0f}W/core)  "
        f"n2 ({_NODE_CAP_CPU[2]}vCPU/{_NODE_CAP_MEM[2]}GB  {19.81:.0f}W/core)  "
        f"n3 ({_NODE_CAP_CPU[3]}vCPU/{_NODE_CAP_MEM[3]}GB  {9.99:.1f}W/core)\n"
        f"  Services: m0=api-gw  m1=ingest  m2=preproc  m3=detection  m4=gen-ai  m5=postproc\n"
        f"{'─'*90}\n"
        f"  {'Scenario':<20} {'n3_avail':>8} {'n3_mem':>7} {'n3_e':>6}  "
        f"{'MILP placement':^22}  {'J':>7}  {'t(ms)':>6}  {'DRL+BC placement':^22}\n"
        f"{'─'*90}"
    )
    print(header)

    for stype in ScenarioType:
        state = gen.sample_state(scenario=stype)

        cpu_avail_n3 = _NODE_CAP_CPU[3] * (1.0 - float(state[3]))
        mem_avail_n3 = _NODE_CAP_MEM[3] - float(state[7])
        e_n3         = float(state[11])

        # ── MILP ─────────────────────────────────────────────────────────────
        ds = _build_milp_dataset(state)
        t0 = time.perf_counter()
        milp_placement = None
        milp_J = float("nan")
        try:
            result = solve_placement(ds)
            milp_ms = (time.perf_counter() - t0) * 1000.0
            if result and result.status in ("optimal", "feasible"):
                milp_placement = {svc: (var, nid) for svc, (var, nid) in result.placement.items()}
                milp_J = result.objective_value
        except Exception:
            milp_ms = (time.perf_counter() - t0) * 1000.0

        # ── DRL+BC ───────────────────────────────────────────────────────────
        drl_action = _drl_predict(state, model_path)

        milp_str, drl_str = _fmt_placement(milp_placement, drl_action)

        J_str = f"{milp_J:+.4f}" if not (milp_J != milp_J) else "  N/A  "
        print(
            f"  {stype.value:<20} {cpu_avail_n3:>8.2f} {mem_avail_n3:>7.2f} {e_n3:>6.1f}  "
            f"{milp_str:<22}  {J_str}  {milp_ms:>6.0f}  {drl_str}"
        )

    print(f"{'─'*90}")
    print("  Interpretation:")
    print("   * n3_avail < 0.5 vCPU → MILP forced to use other nodes (scenario diversity works)")
    print("   * Energy inversion rows → MILP uses n0 for lightweight services (energy-aware)")
    print("   * DRL+BC placement should closely match MILP in most scenarios")
    print(f"{'═'*90}\n")


def export_experts(
    n_samples: int,
    seed: int,
    redis_host: str = "localhost",
    redis_port: int = 6379,
) -> None:
    """Run MILP on diverse scenario states and push expert trajectories to Redis.

    This fixes degenerate BC training data (all-n3) by generating expert
    decisions under adversarial conditions where n3 is NOT optimal.
    """
    import json, redis as _redis
    rdb = _redis.Redis(host=redis_host, port=redis_port,
                       decode_responses=True, socket_timeout=5)
    rdb.ping()

    gen = ScenarioGenerator(seed=seed)
    pushed, skipped = 0, 0
    action_counts: dict[str, int] = {}

    for i in range(n_samples):
        state = gen.sample_state()   # random from 10-scenario adversarial mix
        ds    = _build_milp_dataset(state)
        try:
            result = solve_placement(ds)
        except Exception:
            skipped += 1
            continue
        if not result or result.status not in ("optimal", "feasible"):
            skipped += 1
            continue

        # Convert placement {svc: (variant, node)} → int action [0-3] per service
        action = []
        for svc_id in _SVC_IDS:
            _, node_id = result.placement.get(svc_id, (None, _NODE_IDS[3]))
            action.append(_NODE_IDS.index(node_id) if node_id in _NODE_IDS else 3)

        key = str(action)
        action_counts[key] = action_counts.get(key, 0) + 1

        record = {
            "state":  state.tolist(),
            "action": action,
            "reward": float(result.objective_value),
            "source": "synthetic_expert_adversarial",
        }
        rdb.lpush("milp:expert_trajectories", json.dumps(record))
        pushed += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_samples}] pushed={pushed}  "
                  f"unique_actions={len(action_counts)}")

    total = rdb.llen("milp:expert_trajectories")
    ratio = len(action_counts) / max(pushed, 1)
    print(f"\nDone. pushed={pushed}  skipped={skipped}  "
          f"unique_joint_actions={len(action_counts)}  ratio={ratio:.4f}")
    print(f"Redis milp:expert_trajectories now has {total} entries.")
    print("\nNext: retrain BC")
    print("  .venv/bin/python -m src.drl.offline_trainer "
          "--min-unique-action-ratio 0.004")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="src/drl/models/ppo_simulator.zip")
    p.add_argument(
        "--export-experts", type=int, default=0, metavar="N",
        help="Instead of showing the demo table, collect N diverse expert "
             "trajectories via MILP and push them to Redis for BC retraining.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    args = p.parse_args()

    if args.export_experts > 0:
        export_experts(args.export_experts, args.seed,
                       args.redis_host, args.redis_port)
    else:
        run_demo(model_path=args.model)
