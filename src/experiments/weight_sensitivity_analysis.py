#!/usr/bin/env python3
"""
weight_sensitivity_analysis.py

Run a thesis-defensible sensitivity analysis for the MILP objective weights.

The previous version selected a "knee" by measuring distance to the ideal point
over a single synthetic dataset. This version is stricter:
  1. evaluates each weight tuple across multiple synthetic seeds;
  2. uses the thesis-scale topology by default: 4 nodes and 6 services;
  3. aggregates mean/std metrics per weight tuple;
  4. computes the true non-dominated Pareto front over
     (maximize accuracy, minimize energy, minimize disruption);
  5. reports several defensible profiles instead of pretending that one weight
     tuple is universal for every operating policy.

Usage:
    cd /home/ubuntu/KLTN_project
    python3 src/experiments/weight_sensitivity_analysis.py

Outputs:
    results/weight_analysis/pareto_results.jsonl     raw per-seed runs
    results/weight_analysis/pareto_aggregate.jsonl   aggregated per-weight metrics
    results/weight_analysis/recommendation.json      selected Pareto profiles
    results/weight_analysis/sensitivity_report.txt   human-readable report
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIR = SRC_ROOT / "solver"

for p in [str(SRC_ROOT), str(SOLVER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from solver.dataset_generator import generate_dataset
from solver.milp_model import solve_placement

OUTPUT_DIR = PROJECT_ROOT / "results" / "weight_analysis"

DEFAULT_W_A_VALUES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
DEFAULT_W_C_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]
DEFAULT_SEEDS = [41, 42, 43, 44, 45]


@dataclass(frozen=True)
class WeightRun:
    w_c: float
    w_d: float
    w_a: float
    seed: int
    num_nodes: int
    num_services: int
    objective_J: float
    norm_energy: float
    norm_disruption: float
    norm_accuracy: float
    infeasible: bool
    solve_time_s: float
    error: str = ""


@dataclass(frozen=True)
class WeightAggregate:
    w_c: float
    w_d: float
    w_a: float
    runs: int
    feasible_runs: int
    infeasible_rate: float
    objective_mean: float
    objective_std: float
    accuracy_mean: float
    accuracy_std: float
    energy_mean: float
    energy_std: float
    disruption_mean: float
    disruption_std: float
    solve_time_mean: float
    pareto: bool = False
    distance_to_ideal: float | None = None


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def iter_weight_grid(
    w_a_values: list[float],
    w_c_values: list[float],
    min_w_d: float,
) -> list[tuple[float, float, float]]:
    weights: list[tuple[float, float, float]] = []
    for w_a, w_c in itertools.product(w_a_values, w_c_values):
        w_d = round(1.0 - w_a - w_c, 4)
        if w_d + 1e-9 >= min_w_d:
            weights.append((round(w_c, 4), round(w_d, 4), round(w_a, 4)))
    return weights


def run_weight_grid(
    *,
    num_nodes: int,
    num_services: int,
    seeds: list[int],
    w_a_values: list[float],
    w_c_values: list[float],
    min_w_d: float,
    v_storm_max: int,
    time_limit: int,
) -> list[WeightRun]:
    """Run every weight tuple across multiple synthetic scenarios."""

    weights = iter_weight_grid(w_a_values, w_c_values, min_w_d)
    total = len(weights) * len(seeds)
    print(
        f"Running {total} MILP solves "
        f"({len(weights)} weights x {len(seeds)} seeds, "
        f"nodes={num_nodes}, services={num_services}, v_storm_max={v_storm_max})..."
    )

    results: list[WeightRun] = []
    run_idx = 0
    for w_c, w_d, w_a in weights:
        for seed in seeds:
            run_idx += 1
            try:
                ds = generate_dataset(
                    num_nodes=num_nodes,
                    num_services=num_services,
                    seed=seed,
                    v_storm_max=v_storm_max,
                    w_c=w_c,
                    w_d=w_d,
                    w_a=w_a,
                )
                result = solve_placement(ds, verbose=False, time_limit=time_limit)
            except Exception as exc:
                print(
                    f"  [{run_idx}/{total}] seed={seed} "
                    f"w_a={w_a:.2f} w_c={w_c:.2f} w_d={w_d:.2f} -> ERROR: {exc}"
                )
                results.append(
                    WeightRun(
                        w_c=w_c,
                        w_d=w_d,
                        w_a=w_a,
                        seed=seed,
                        num_nodes=num_nodes,
                        num_services=num_services,
                        objective_J=1.0,
                        norm_energy=1.0,
                        norm_disruption=1.0,
                        norm_accuracy=0.0,
                        infeasible=True,
                        solve_time_s=0.0,
                        error=str(exc),
                    )
                )
                continue

            if result is None:
                print(
                    f"  [{run_idx}/{total}] seed={seed} "
                    f"w_a={w_a:.2f} w_c={w_c:.2f} w_d={w_d:.2f} -> INFEASIBLE"
                )
                results.append(
                    WeightRun(
                        w_c=w_c,
                        w_d=w_d,
                        w_a=w_a,
                        seed=seed,
                        num_nodes=num_nodes,
                        num_services=num_services,
                        objective_J=1.0,
                        norm_energy=1.0,
                        norm_disruption=1.0,
                        norm_accuracy=0.0,
                        infeasible=True,
                        solve_time_s=0.0,
                    )
                )
                continue

            print(
                f"  [{run_idx}/{total}] seed={seed} "
                f"w_a={w_a:.2f} w_c={w_c:.2f} w_d={w_d:.2f} -> "
                f"J={result.objective_value:.4f} "
                f"acc={result.norm_gain_accuracy:.3f} "
                f"energy={result.norm_cost_energy:.3f} "
                f"disrupt={result.norm_cost_disruption:.3f} "
                f"t={result.solve_time:.2f}s"
            )
            results.append(
                WeightRun(
                    w_c=w_c,
                    w_d=w_d,
                    w_a=w_a,
                    seed=seed,
                    num_nodes=num_nodes,
                    num_services=num_services,
                    objective_J=result.objective_value,
                    norm_energy=result.norm_cost_energy,
                    norm_disruption=result.norm_cost_disruption,
                    norm_accuracy=result.norm_gain_accuracy,
                    infeasible=False,
                    solve_time_s=result.solve_time,
                )
            )

    return results


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _safe_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def aggregate_results(results: list[WeightRun]) -> list[WeightAggregate]:
    grouped: dict[tuple[float, float, float], list[WeightRun]] = {}
    for r in results:
        grouped.setdefault((r.w_c, r.w_d, r.w_a), []).append(r)

    aggregates: list[WeightAggregate] = []
    for (w_c, w_d, w_a), group in sorted(grouped.items()):
        feasible = [r for r in group if not r.infeasible]
        objectives = [r.objective_J for r in feasible]
        accs = [r.norm_accuracy for r in feasible]
        energies = [r.norm_energy for r in feasible]
        disruptions = [r.norm_disruption for r in feasible]
        solve_times = [r.solve_time_s for r in feasible]

        aggregates.append(
            WeightAggregate(
                w_c=w_c,
                w_d=w_d,
                w_a=w_a,
                runs=len(group),
                feasible_runs=len(feasible),
                infeasible_rate=1.0 - (len(feasible) / len(group)),
                objective_mean=_safe_mean(objectives, default=1.0),
                objective_std=_safe_std(objectives),
                accuracy_mean=_safe_mean(accs),
                accuracy_std=_safe_std(accs),
                energy_mean=_safe_mean(energies, default=1.0),
                energy_std=_safe_std(energies),
                disruption_mean=_safe_mean(disruptions, default=1.0),
                disruption_std=_safe_std(disruptions),
                solve_time_mean=_safe_mean(solve_times),
            )
        )

    return aggregates


def dominates(a: WeightAggregate, b: WeightAggregate, eps: float = 1e-9) -> bool:
    """Return True if a Pareto-dominates b."""

    at_least_as_good = (
        a.infeasible_rate <= b.infeasible_rate + eps
        and a.accuracy_mean >= b.accuracy_mean - eps
        and a.energy_mean <= b.energy_mean + eps
        and a.disruption_mean <= b.disruption_mean + eps
    )
    strictly_better = (
        a.infeasible_rate < b.infeasible_rate - eps
        or a.accuracy_mean > b.accuracy_mean + eps
        or a.energy_mean < b.energy_mean - eps
        or a.disruption_mean < b.disruption_mean - eps
    )
    return at_least_as_good and strictly_better


def find_pareto_front(aggregates: list[WeightAggregate]) -> list[WeightAggregate]:
    feasible = [a for a in aggregates if a.feasible_runs > 0]
    front = [a for a in feasible if not any(dominates(other, a) for other in feasible)]
    return sorted(front, key=lambda a: (-a.accuracy_mean, a.energy_mean, a.disruption_mean))


def _norm(value: float, lo: float, hi: float) -> float:
    return (value - lo) / (hi - lo) if hi > lo else 0.5


def distance_to_ideal(candidate: WeightAggregate, population: list[WeightAggregate]) -> float:
    """Distance to ideal over normalized metrics.

    Ideal point: max accuracy, min energy, min disruption, zero infeasibility.
    The ranges are computed on all aggregated feasible candidates, while the
    selected candidate is constrained to the Pareto front.
    """

    feasible = [a for a in population if a.feasible_runs > 0]
    accs = [a.accuracy_mean for a in feasible]
    energies = [a.energy_mean for a in feasible]
    disruptions = [a.disruption_mean for a in feasible]
    infeasible_rates = [a.infeasible_rate for a in feasible]

    acc_n = _norm(candidate.accuracy_mean, min(accs), max(accs))
    energy_n = _norm(candidate.energy_mean, min(energies), max(energies))
    disruption_n = _norm(candidate.disruption_mean, min(disruptions), max(disruptions))
    infeasible_n = _norm(candidate.infeasible_rate, min(infeasible_rates), max(infeasible_rates))

    return math.sqrt(
        (1.0 - acc_n) ** 2
        + energy_n**2
        + disruption_n**2
        + infeasible_n**2
    )


def with_pareto_metadata(
    aggregates: list[WeightAggregate],
    pareto_front: list[WeightAggregate],
) -> list[WeightAggregate]:
    pareto_keys = {(a.w_c, a.w_d, a.w_a) for a in pareto_front}
    updated: list[WeightAggregate] = []
    for a in aggregates:
        is_pareto = (a.w_c, a.w_d, a.w_a) in pareto_keys
        dist = distance_to_ideal(a, aggregates) if a.feasible_runs > 0 else None
        updated.append(
            WeightAggregate(
                **{
                    **asdict(a),
                    "pareto": is_pareto,
                    "distance_to_ideal": dist,
                }
            )
        )
    return updated


def select_balanced_knee(
    pareto_front: list[WeightAggregate],
    aggregates: list[WeightAggregate],
) -> WeightAggregate | None:
    if not pareto_front:
        return None
    return min(
        pareto_front,
        key=lambda a: (
            distance_to_ideal(a, aggregates),
            -a.accuracy_mean,
            a.energy_mean,
            a.disruption_mean,
        ),
    )


def select_quality_first(
    pareto_front: list[WeightAggregate],
    aggregates: list[WeightAggregate],
    tolerance: float,
) -> WeightAggregate | None:
    if not pareto_front:
        return None
    max_acc = max(a.accuracy_mean for a in pareto_front)
    candidates = [a for a in pareto_front if a.accuracy_mean >= max_acc - tolerance]
    return min(
        candidates,
        key=lambda a: (
            distance_to_ideal(a, aggregates),
            a.energy_mean,
            a.disruption_mean,
            -a.accuracy_mean,
        ),
    )


def select_stability_first(pareto_front: list[WeightAggregate]) -> WeightAggregate | None:
    if not pareto_front:
        return None
    return min(
        pareto_front,
        key=lambda a: (a.disruption_mean, -a.accuracy_mean, a.energy_mean),
    )


def select_efficiency_first(pareto_front: list[WeightAggregate]) -> WeightAggregate | None:
    if not pareto_front:
        return None
    return min(
        pareto_front,
        key=lambda a: (a.energy_mean, -a.accuracy_mean, a.disruption_mean),
    )


def compact(a: WeightAggregate | None) -> dict[str, float | int | bool] | None:
    if a is None:
        return None
    return {
        "w_c": a.w_c,
        "w_d": a.w_d,
        "w_a": a.w_a,
        "runs": a.runs,
        "feasible_runs": a.feasible_runs,
        "infeasible_rate": a.infeasible_rate,
        "accuracy_mean": a.accuracy_mean,
        "accuracy_std": a.accuracy_std,
        "energy_mean": a.energy_mean,
        "energy_std": a.energy_std,
        "disruption_mean": a.disruption_mean,
        "disruption_std": a.disruption_std,
        "objective_mean": a.objective_mean,
        "objective_std": a.objective_std,
        "distance_to_ideal": a.distance_to_ideal,
        "pareto": a.pareto,
    }


def outcome_key(a: WeightAggregate, ndigits: int = 4) -> tuple[float, float, float, float]:
    return (
        round(a.accuracy_mean, ndigits),
        round(a.energy_mean, ndigits),
        round(a.disruption_mean, ndigits),
        round(a.infeasible_rate, ndigits),
    )


def pareto_outcome_groups(
    pareto_front: list[WeightAggregate],
) -> list[dict[str, object]]:
    """Group Pareto-equivalent weight tuples by their aggregated outcomes."""

    grouped: dict[tuple[float, float, float, float], list[WeightAggregate]] = {}
    for a in pareto_front:
        grouped.setdefault(outcome_key(a), []).append(a)

    groups: list[dict[str, object]] = []
    for key, items in grouped.items():
        representative = min(
            items,
            key=lambda a: (
                a.distance_to_ideal if a.distance_to_ideal is not None else float("inf"),
                -a.w_a,
                a.w_c,
                a.w_d,
            ),
        )
        groups.append(
            {
                "outcome_key": key,
                "representative": compact(representative),
                "equivalent_weights": [
                    {"w_c": a.w_c, "w_d": a.w_d, "w_a": a.w_a}
                    for a in sorted(items, key=lambda x: (-x.w_a, x.w_c, x.w_d))
                ],
            }
        )

    return sorted(
        groups,
        key=lambda g: (
            -float(g["outcome_key"][0]),  # type: ignore[index]
            float(g["outcome_key"][1]),   # type: ignore[index]
            float(g["outcome_key"][2]),   # type: ignore[index]
        ),
    )


def format_row(a: WeightAggregate) -> str:
    return (
        f"W_A={a.w_a:.2f} W_C={a.w_c:.2f} W_D={a.w_d:.2f} | "
        f"acc={a.accuracy_mean:.4f}±{a.accuracy_std:.4f} "
        f"energy={a.energy_mean:.4f}±{a.energy_std:.4f} "
        f"disrupt={a.disruption_mean:.4f}±{a.disruption_std:.4f} "
        f"infeas={a.infeasible_rate:.2f} "
        f"dist={a.distance_to_ideal if a.distance_to_ideal is not None else float('nan'):.4f}"
    )


def sensitivity_report(
    *,
    aggregates: list[WeightAggregate],
    pareto_front: list[WeightAggregate],
    balanced: WeightAggregate | None,
    quality_first: WeightAggregate | None,
    stability_first: WeightAggregate | None,
    efficiency_first: WeightAggregate | None,
    outcome_groups: list[dict[str, object]],
    seeds: list[int],
    num_nodes: int,
    num_services: int,
    v_storm_max: int,
    quality_tolerance: float,
) -> str:
    feasible = [a for a in aggregates if a.feasible_runs > 0]
    lines = [
        "=" * 78,
        "MILP OBJECTIVE WEIGHT SENSITIVITY ANALYSIS - TRUE PARETO FRONT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 78,
        "",
        "Objective:",
        "  minimize J = W_C * C_norm + W_D * D_norm - W_A * A_norm",
        "",
        "Selection method:",
        "  1. Evaluate each weight tuple across multiple synthetic seeds.",
        "  2. Aggregate mean/std metrics per weight tuple.",
        "  3. Keep only true Pareto non-dominated tuples:",
        "     maximize accuracy, minimize energy, minimize disruption.",
        "  4. Pick profile-specific recommendations from that Pareto front.",
        "",
        f"Scenario scale: nodes={num_nodes}, services={num_services}, v_storm_max={v_storm_max}",
        f"Seeds: {seeds}",
        f"Aggregated weights: {len(feasible)} feasible / {len(aggregates)} total",
        f"Pareto front size: {len(pareto_front)}",
        f"Unique Pareto outcome groups: {len(outcome_groups)}",
        "",
        "-" * 78,
        "TOP 5 BY MEAN ACCURACY",
        "-" * 78,
    ]

    for i, a in enumerate(sorted(feasible, key=lambda x: (-x.accuracy_mean, x.energy_mean))[:5], 1):
        lines.append(f"  {i}. {format_row(a)}")

    lines += ["", "-" * 78, "TOP 5 BY MEAN ENERGY EFFICIENCY", "-" * 78]
    for i, a in enumerate(sorted(feasible, key=lambda x: (x.energy_mean, -x.accuracy_mean))[:5], 1):
        lines.append(f"  {i}. {format_row(a)}")

    lines += ["", "-" * 78, "TOP 5 BY MEAN STABILITY", "-" * 78]
    for i, a in enumerate(sorted(feasible, key=lambda x: (x.disruption_mean, -x.accuracy_mean))[:5], 1):
        lines.append(f"  {i}. {format_row(a)}")

    lines += ["", "-" * 78, "TRUE PARETO FRONT", "-" * 78]
    for i, a in enumerate(pareto_front, 1):
        lines.append(f"  {i}. {format_row(a)}")

    lines += ["", "-" * 78, "PARETO OUTCOME GROUPS", "-" * 78]
    for i, group in enumerate(outcome_groups, 1):
        rep = group["representative"]
        weights = group["equivalent_weights"]
        if not isinstance(rep, dict) or not isinstance(weights, list):
            continue
        weight_text = ", ".join(
            f"({w['w_c']:.2f},{w['w_d']:.2f},{w['w_a']:.2f})"
            for w in weights
            if isinstance(w, dict)
        )
        lines.append(
            f"  {i}. representative W_A={rep['w_a']:.2f} W_C={rep['w_c']:.2f} "
            f"W_D={rep['w_d']:.2f} | "
            f"acc={rep['accuracy_mean']:.4f} energy={rep['energy_mean']:.4f} "
            f"disrupt={rep['disruption_mean']:.4f} | "
            f"equivalent weights: {weight_text}"
        )

    lines += ["", "=" * 78, "RECOMMENDED PROFILES", "=" * 78]
    profile_rows = [
        ("Balanced Pareto knee", balanced),
        (f"Quality-first Pareto knee (within {quality_tolerance:.2f} of max accuracy)", quality_first),
        ("Stability-first Pareto point", stability_first),
        ("Efficiency-first Pareto point", efficiency_first),
    ]
    for label, item in profile_rows:
        lines.append("")
        lines.append(label + ":")
        lines.append("  " + (format_row(item) if item else "No feasible Pareto point"))

    lines += [
        "",
        "Thesis interpretation:",
        "  - Use the balanced knee when the thesis claims a neutral trade-off policy.",
        "  - Use the quality-first knee when the thesis explicitly prioritizes Edge AI quality.",
        "  - Report the other two profiles as sensitivity evidence, not as the main policy.",
        "=" * 78,
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--num-services", type=int, default=6)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--w-a-values", type=parse_float_list, default=DEFAULT_W_A_VALUES)
    parser.add_argument("--w-c-values", type=parse_float_list, default=DEFAULT_W_C_VALUES)
    parser.add_argument("--min-w-d", type=float, default=0.05)
    parser.add_argument("--v-storm-max", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=30)
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Running multi-seed MILP weight sensitivity grid...")
    raw_results = run_weight_grid(
        num_nodes=args.num_nodes,
        num_services=args.num_services,
        seeds=args.seeds,
        w_a_values=args.w_a_values,
        w_c_values=args.w_c_values,
        min_w_d=args.min_w_d,
        v_storm_max=args.v_storm_max,
        time_limit=args.time_limit,
    )

    raw_path = OUTPUT_DIR / "pareto_results.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in raw_results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"\n[2/5] Raw per-seed results saved -> {raw_path}")

    aggregates = aggregate_results(raw_results)
    pareto_front = find_pareto_front(aggregates)
    aggregates = with_pareto_metadata(aggregates, pareto_front)
    pareto_front = [a for a in aggregates if a.pareto]
    pareto_front = sorted(pareto_front, key=lambda a: (-a.accuracy_mean, a.energy_mean, a.disruption_mean))

    aggregate_path = OUTPUT_DIR / "pareto_aggregate.jsonl"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        for a in sorted(aggregates, key=lambda x: (x.w_a, x.w_c, x.w_d)):
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
    print(f"[3/5] Aggregated per-weight results saved -> {aggregate_path}")

    balanced = select_balanced_knee(pareto_front, aggregates)
    quality_first = select_quality_first(pareto_front, aggregates, args.quality_tolerance)
    stability_first = select_stability_first(pareto_front)
    efficiency_first = select_efficiency_first(pareto_front)
    outcome_groups = pareto_outcome_groups(pareto_front)

    report = sensitivity_report(
        aggregates=aggregates,
        pareto_front=pareto_front,
        balanced=balanced,
        quality_first=quality_first,
        stability_first=stability_first,
        efficiency_first=efficiency_first,
        outcome_groups=outcome_groups,
        seeds=args.seeds,
        num_nodes=args.num_nodes,
        num_services=args.num_services,
        v_storm_max=args.v_storm_max,
        quality_tolerance=args.quality_tolerance,
    )

    report_path = OUTPUT_DIR / "sensitivity_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[4/5] Report saved -> {report_path}")
    print(report)

    recommendation = {
        "method": "true_pareto_front_multi_seed",
        "generated_at": datetime.now().isoformat(),
        "scenario": {
            "num_nodes": args.num_nodes,
            "num_services": args.num_services,
            "seeds": args.seeds,
            "v_storm_max": args.v_storm_max,
            "min_w_d": args.min_w_d,
            "quality_tolerance": args.quality_tolerance,
        },
        "objective_space": {
            "maximize": ["accuracy_mean"],
            "minimize": ["energy_mean", "disruption_mean", "infeasible_rate"],
        },
        "balanced_pareto_knee": compact(balanced),
        "quality_first_pareto_knee": compact(quality_first),
        "stability_first_pareto_point": compact(stability_first),
        "efficiency_first_pareto_point": compact(efficiency_first),
        "pareto_front": [compact(a) for a in pareto_front],
        "pareto_outcome_groups": outcome_groups,
    }

    rec_path = OUTPUT_DIR / "recommendation.json"
    rec_path.write_text(json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[5/5] Recommendation saved -> {rec_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
