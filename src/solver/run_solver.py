# run_solver.py
"""
Experiment runner for the MILP placement solver.
Builds a dataset from the live cluster, solves placement, exports results to CSV.

Called by edge_controller.py each control cycle:
  python3 run_solver.py --placement last_placement.json --output-dir results/

Weight defaults (w_c, w_d, w_a) are consistent between the function signature
and the CLI argparse defaults — both use (0.1, 0.2, 0.8) to heavily favour
detection quality. Change all three in sync if tuning.
"""

import csv, os, sys, json, argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from milp_model import solve_placement
from metrics_collector import build_dataset_from_cluster, get_metrics_quality_report
from dataset_generator import save_dataset
from config import MILP_W_C, MILP_W_D, MILP_W_A


def run_experiment(
    output_dir: str = "results",
    last_placement: dict = None,
    w_c: float = MILP_W_C,
    w_d: float = MILP_W_D,
    w_a: float = MILP_W_A,
    max_fallback_ratio: float = 0.6,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Build dataset from real cluster state ──
    print("1. Building dataset from cluster metrics...")
    import k8s_client as _k8s
    is_cold_start = not last_placement  # no prior state → treat all placements as initial
    # Emergency evacuation: services on a now-unavailable node have x_prev=0 for all
    # remaining nodes, so each needs v[m]=1.  Raise v_storm_max to cover all displaced
    # services so the MILP stays feasible when a node fails mid-run.
    available_nodes = set(_k8s.get_all_worker_nodes())
    displaced = sum(
        1 for info in (last_placement or {}).values()
        if info.get("node") and info["node"] not in available_nodes
    )
    base_storm = 5 if is_cold_start else 2
    v_storm_max = max(base_storm, displaced)
    if displaced:
        print(f"   Emergency evacuation: {displaced} service(s) on unavailable "
              f"node(s) — v_storm_max raised to {v_storm_max}")
    ds = build_dataset_from_cluster(
        last_placement=last_placement or {},
        w_c=w_c, w_d=w_d, w_a=w_a,
        v_storm_max=v_storm_max,
    )
    print(f"   {ds.summary()}")

    quality = get_metrics_quality_report()
    print(
        "   Metrics quality: "
        f"fallback_ratio={quality['fallback_ratio']:.4f} "
        f"({quality['total_fallbacks']}/{quality['total_samples']})"
    )
    if quality["fallback_ratio"] > max_fallback_ratio:
        print(
            "   Metrics quality gate failed: "
            f"fallback_ratio={quality['fallback_ratio']:.4f} > "
            f"threshold={max_fallback_ratio:.4f}"
        )
        sys.exit(2)

    # Save dataset JSON for experiment reproducibility
    ds_path = os.path.join(output_dir, f"dataset_{timestamp}.json")
    save_dataset(ds, ds_path)

    # ── Solve ──
    print("\n2. Running MILP solver (Pyomo + HiGHS)...")
    result = solve_placement(ds, verbose=False)

    if not result:
        print("   Solver failed — no feasible solution.")
        sys.exit(1)

    print(f"   Status: {result.status}  |  "
          f"J={result.objective_value:.4f} (normalized)  |  "
          f"t={result.solve_time:.2f}s")

    # ── Export placement CSV ──
    csv_path = os.path.join(output_dir, f"placement_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "service_id", "prev_variant_id", "prev_node_id",
            "variant_id", "node_id", "migrated", "migration_type"
        ])
        for svc, (var, node) in sorted(
            result.placement.items(),
            key=lambda x: int(x[0].replace("m", ""))
        ):
            prev_var, prev_node = result.prev_placement.get(svc, ("-", "-"))
            writer.writerow([
                svc, prev_var, prev_node,
                var, node,
                result.migrations.get(svc, False),
                result.migration_types.get(svc, "Stayed"),
            ])
    print(f"\n3. Placement CSV      → {csv_path}")

    # ── Export summary CSV ──
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["status",          result.status])
        writer.writerow(["solve_time_s",    f"{result.solve_time:.4f}"])
        writer.writerow(["objective_J",     f"{result.objective_value:.4f}"])
        # Raw objective components
        writer.writerow(["cost_energy",     f"{result.cost_energy:.4f}"])
        writer.writerow(["cost_disruption", f"{result.cost_disruption:.4f}"])
        writer.writerow(["gain_accuracy",   f"{result.gain_accuracy:.4f}"])
        # Normalized components (what actually drive J)
        writer.writerow(["norm_cost_energy",     f"{result.norm_cost_energy:.4f}"])
        writer.writerow(["norm_cost_disruption", f"{result.norm_cost_disruption:.4f}"])
        writer.writerow(["norm_gain_accuracy",   f"{result.norm_gain_accuracy:.4f}"])
        # Node-level breakdown for thesis analysis
        for node_id, usage in result.resource_usage.items():
            writer.writerow([f"resource_usage_{node_id}", f"{usage:.4f}"])
        for node_id, theta in result.theta.items():
            writer.writerow([f"theta_{node_id}", f"{theta:.4f}"])
    print(f"   Summary CSV        → {summary_path}")
    print(f"   Dataset JSON       → {ds_path}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement",   default=None,
                        help="Path to last_placement.json from controller")
    parser.add_argument("--output-dir",  default="results")
    # Weights: consistent with run_experiment() function defaults above
    parser.add_argument("--w-c", type=float, default=MILP_W_C,
                        help=f"Weight for energy cost (default {MILP_W_C})")
    parser.add_argument("--w-d", type=float, default=MILP_W_D,
                        help=f"Weight for migration disruption (default {MILP_W_D})")
    parser.add_argument("--w-a", type=float, default=MILP_W_A,
                        help=f"Weight for detection quality (default {MILP_W_A})")
    parser.add_argument("--max-fallback-ratio", type=float, default=0.6,
                        help="Maximum acceptable telemetry fallback ratio before blocking solve")
    args = parser.parse_args()

    last_placement = {}
    if args.placement and os.path.exists(args.placement):
        with open(args.placement) as f:
            last_placement = json.load(f)

    run_experiment(
        output_dir=args.output_dir,
        last_placement=last_placement,
        w_c=args.w_c,
        w_d=args.w_d,
        w_a=args.w_a,
        max_fallback_ratio=args.max_fallback_ratio,
    )
