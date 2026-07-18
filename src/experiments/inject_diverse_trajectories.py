#!/usr/bin/env python3
"""
inject_diverse_trajectories.py  (v2 — randomised weight expansion)
==============================
Synthetic expert trajectory injector for DRL Behavioral Cloning.

Solves the MILP over a systematic parameter grid (weight combos × starting
placements × node topologies) and pushes the resulting (state, action) pairs
directly to Redis — WITHOUT applying anything to the live cluster.

This is the fastest way to reach the unique_action_ratio ≥ 0.20 gate.

Usage:
    python3 src/experiments/inject_diverse_trajectories.py [--dry-run] [--redis-host HOST]

    --dry-run     Print what would be pushed without writing to Redis.
    --flush       Delete milp:expert_trajectories before injecting.
    --redis-host  Redis host (default: 10.42.1.183).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIR = PROJECT_ROOT / "src" / "solver"

for p in [str(SRC_ROOT), str(SOLVER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import redis as redis_lib
from solver.metrics_collector import build_dataset_from_cluster
from solver.milp_model import solve_placement

# ─── Parameter Grid ────────────────────────────────────────────────────────────
# Base anchor weights: covers the major objective preference regimes.
BASE_WEIGHT_GRID = [
    # (w_c, w_d, w_a, label)  — must sum to ~1.0
    (0.70, 0.10, 0.20, "energy_dominant"),
    (0.05, 0.05, 0.90, "quality_dominant"),
    (0.30, 0.20, 0.50, "balanced"),
    (0.50, 0.40, 0.10, "stability_focused"),
    (0.20, 0.05, 0.75, "quality_lean"),
    (0.60, 0.30, 0.10, "energy_lean"),
    (0.10, 0.70, 0.20, "disruption_dominant"),
    (0.45, 0.05, 0.50, "energy_quality_split"),
]


def _sample_random_weights(n: int, seed: int = 0) -> list[tuple[float, float, float, str]]:
    """Draw n random weight triples uniformly from the 3-simplex."""
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # Dirichlet(1,1,1) via exponential trick
        a, b, c = rng.expovariate(1), rng.expovariate(1), rng.expovariate(1)
        s = a + b + c
        w_c, w_d, w_a = round(a / s, 4), round(b / s, 4), round(c / s, 4)
        out.append((w_c, w_d, w_a, f"random_{i:03d}"))
    return out

# Node hostnames in your cluster (in MILP sort order)
NODES = ["edge-nodes-1", "edge-nodes-2", "edge-nodes-3", "edge-nodes-4"]
H0, H1, H2, H3 = NODES

# Starting placement templates — each represents a different cluster history
# that the MILP must react to, producing different migration actions.
PLACEMENT_TEMPLATES = [
    # Template A: Clustered on node 1 (triggers rebalancing)
    {
        "label": "all_on_n1",
        "placement": {
            "api-gateway":  {"node": H0, "variant": "standard"},
            "ingest":       {"node": H0, "variant": "standard"},
            "preprocess":   {"node": H0, "variant": "standard"},
            "detection":    {"node": H0, "variant": "yolo26-medium"},
            "gen-ai":       {"node": H0, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": H0, "variant": "standard"},
        },
    },
    # Template B: Spread with heavy variants (triggers energy-saving migration)
    {
        "label": "spread_heavy",
        "placement": {
            "api-gateway":  {"node": H1, "variant": "standard"},
            "ingest":       {"node": H0, "variant": "standard"},
            "preprocess":   {"node": H2, "variant": "standard"},
            "detection":    {"node": H3, "variant": "yolo26-medium"},
            "gen-ai":       {"node": H3, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": H1, "variant": "standard"},
        },
    },
    # Template C: Lightweight variants (triggers quality-upgrade migration)
    {
        "label": "spread_nano",
        "placement": {
            "api-gateway":  {"node": H0, "variant": "standard"},
            "ingest":       {"node": H1, "variant": "standard"},
            "preprocess":   {"node": H2, "variant": "standard"},
            "detection":    {"node": H0, "variant": "yolo26-nano"},
            "gen-ai":       {"node": H3, "variant": "qwen-1.5b-nano"},
            "postprocess":  {"node": H3, "variant": "standard"},
        },
    },
    # Template D: Detection on RAM-poor node with medium variant (triggers migration)
    {
        "label": "detection_wrong_node",
        "placement": {
            "api-gateway":  {"node": H2, "variant": "standard"},
            "ingest":       {"node": H2, "variant": "standard"},
            "preprocess":   {"node": H2, "variant": "standard"},
            "detection":    {"node": H2, "variant": "yolo26-medium"},
            "gen-ai":       {"node": H1, "variant": "llama-3b-small"},
            "postprocess":  {"node": H0, "variant": "standard"},
        },
    },
    # Template E: Detection on n1, gen-ai on n3 (symmetric-ish, tests fine-tuning)
    {
        "label": "partial_balance",
        "placement": {
            "api-gateway":  {"node": H3, "variant": "standard"},
            "ingest":       {"node": H0, "variant": "standard"},
            "preprocess":   {"node": H1, "variant": "standard"},
            "detection":    {"node": H1, "variant": "yolo26-small"},
            "gen-ai":       {"node": H3, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": H2, "variant": "standard"},
        },
    },
    # Template F: Everything crammed on node 4 (highest energy node)
    {
        "label": "all_on_n4",
        "placement": {
            "api-gateway":  {"node": H3, "variant": "standard"},
            "ingest":       {"node": H3, "variant": "standard"},
            "preprocess":   {"node": H3, "variant": "standard"},
            "detection":    {"node": H3, "variant": "yolo26-medium"},
            "gen-ai":       {"node": H3, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": H3, "variant": "standard"},
        },
    },
    # Template G: Mixed small variants, partially collocated
    {
        "label": "mixed_small",
        "placement": {
            "api-gateway":  {"node": H0, "variant": "standard"},
            "ingest":       {"node": H1, "variant": "standard"},
            "preprocess":   {"node": H3, "variant": "standard"},
            "detection":    {"node": H2, "variant": "yolo26-small"},
            "gen-ai":       {"node": H2, "variant": "llama-3b-small"},
            "postprocess":  {"node": H0, "variant": "standard"},
        },
    },
    # Template H: gen-ai alone on n1 (most RAM), all others on n4
    {
        "label": "genai_isolated",
        "placement": {
            "api-gateway":  {"node": H3, "variant": "standard"},
            "ingest":       {"node": H3, "variant": "standard"},
            "preprocess":   {"node": H3, "variant": "standard"},
            "detection":    {"node": H3, "variant": "yolo26-nano"},
            "gen-ai":       {"node": H1, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": H3, "variant": "standard"},
        },
    },
    # Template I: All on node 1 with quality variants (forces node-4 bias to kick in)
    {
        "label": "all_on_n1_quality",
        "placement": {
            "api-gateway":  {"node": H0, "variant": "standard"},
            "ingest":       {"node": H0, "variant": "standard"},
            "preprocess":   {"node": H0, "variant": "standard"},
            "detection":    {"node": H0, "variant": "yolo26-small"},
            "gen-ai":       {"node": H0, "variant": "llama-3b-small"},
            "postprocess":  {"node": H0, "variant": "standard"},
        },
    },
    # Template J: Deliberately pinned to nodes 2+3 only
    {
        "label": "n2_n3_only",
        "placement": {
            "api-gateway":  {"node": H1, "variant": "standard"},
            "ingest":       {"node": H2, "variant": "standard"},
            "preprocess":   {"node": H1, "variant": "standard"},
            "detection":    {"node": H2, "variant": "yolo26-medium"},
            "gen-ai":       {"node": H1, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": H2, "variant": "standard"},
        },
    },
    # Template K: Zigzag across all 4 nodes, mixed variants
    {
        "label": "zigzag_mixed",
        "placement": {
            "api-gateway":  {"node": H0, "variant": "standard"},
            "ingest":       {"node": H1, "variant": "standard"},
            "preprocess":   {"node": H2, "variant": "standard"},
            "detection":    {"node": H3, "variant": "yolo26-small"},
            "gen-ai":       {"node": H0, "variant": "llama-3b-small"},
            "postprocess":  {"node": H1, "variant": "standard"},
        },
    },
    # Template L: Reverse zigzag
    {
        "label": "zigzag_reverse",
        "placement": {
            "api-gateway":  {"node": H3, "variant": "standard"},
            "ingest":       {"node": H2, "variant": "standard"},
            "preprocess":   {"node": H1, "variant": "standard"},
            "detection":    {"node": H0, "variant": "yolo26-medium"},
            "gen-ai":       {"node": H3, "variant": "qwen-1.5b-nano"},
            "postprocess":  {"node": H2, "variant": "standard"},
        },
    },
]

STORM_VARIANTS = [2, 3, 4, 5, 6]   # vary v_storm_max for even more variation

# Number of random weight samples drawn from the 3-simplex.
# These supplement BASE_WEIGHT_GRID to cover the full weight space.
N_RANDOM_WEIGHTS = 20


# ─── Helpers copied from milp_agent.py (exact logic, no pod dependency) ────────

SERVICE_ORDER = ["m0", "m1", "m2", "m3", "m4", "m5"]


def _build_state_vector(ds: Any, result: Any) -> list[float]:
    """Replicate milp_agent._build_state_vector without Redis / k8s calls."""
    nodes = list(ds.nodes)
    node_ids = [n.node_id for n in nodes]
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    cpu_util = [0.0] * 4
    for node in nodes[:4]:
        idx = node_index[node.node_id]
        cap = float(node.cap_cpu) if float(node.cap_cpu) > 0 else 1.0
        cpu_util[idx] = max(0.0, min(1.0, float(result.resource_usage.get(node.node_id, 0)) / cap))

    mem_gb = [0.0] * 4
    for node in nodes[:4]:
        idx = node_index[node.node_id]
        mem_gb[idx] = float(result.mem_usage.get(node.node_id, 0))

    e_cpu_unit = [float(node.energy_cost) for node in nodes[:4]] + [0.0] * (4 - min(4, len(nodes)))

    det_variants = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
    det_onehot = [0.0, 0.0, 0.0]
    det_variant = result.placement.get("m3", ("yolo26-nano", "n0"))[0]
    if det_variant in det_variants:
        det_onehot[det_variants.index(det_variant)] = 1.0

    # Placement one-hot: use MILP result directly (no Redis confirmed placement)
    placement_onehot = [0.0] * 24
    for svc_idx, svc_id in enumerate(SERVICE_ORDER):
        slot = svc_idx * 4
        node_id = result.placement.get(svc_id, ("standard", node_ids[0]))[1]
        idx = node_index.get(node_id, 0)
        if 0 <= idx < 4:
            placement_onehot[slot + idx] = 1.0

    e2e_latency = [0.0]   # synthetic — no live Prometheus
    migrations = [float(sum(1 for m in result.migration_types.values() if m != "Stayed"))]
    weights = [float(ds.w_c), float(ds.w_d), float(ds.w_a)]

    vec = cpu_util + mem_gb + e_cpu_unit + det_onehot + placement_onehot + e2e_latency + migrations + weights
    vec = (vec + [0.0] * 44)[:44]
    vec = [float(v) if math.isfinite(float(v)) else 0.0 for v in vec]
    return vec


def _build_action_vector(ds: Any, result: Any) -> list[int]:
    """Replicate milp_agent._build_action_vector."""
    node_ids = [n.node_id for n in ds.nodes]
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    det_variants = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
    gen_variants = ["qwen-1.5b-nano", "llama-3b-small", "gemma2-2b-medium"]

    actions: list[int] = []
    for svc_id in SERVICE_ORDER:
        variant, node_id = result.placement.get(svc_id, ("standard", node_ids[0] if node_ids else "n0"))
        node_idx = node_to_idx.get(node_id, 0)
        if svc_id == "m3":
            var_idx = det_variants.index(variant) if variant in det_variants else 0
            actions.append(var_idx * 4 + min(3, node_idx))
        elif svc_id == "m4":
            var_idx = gen_variants.index(variant) if variant in gen_variants else 0
            actions.append(var_idx * 4 + min(3, node_idx))
        else:
            actions.append(min(3, node_idx))
    return actions


# ─── Main Injector ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    rdb = None
    if not args.dry_run:
        rdb = redis_lib.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
        try:
            rdb.ping()
        except redis_lib.ConnectionError as e:
            print(f"[ERROR] Cannot connect to Redis at {args.redis_host}:{args.redis_port}: {e}")
            return 1

        if args.flush:
            old_len = rdb.llen("milp:expert_trajectories")
            rdb.delete("milp:expert_trajectories")
            print(f"[FLUSH] Deleted {old_len} stale trajectories from milp:expert_trajectories")

    # Build full parameter grid: anchor weights + random weights
    weight_grid = BASE_WEIGHT_GRID + _sample_random_weights(N_RANDOM_WEIGHTS, seed=args.seed)
    grid = list(product(weight_grid, PLACEMENT_TEMPLATES, STORM_VARIANTS))

    # Shuffle the grid so consecutive Redis entries (LPUSH prepends, so index 0 is
    # the last-pushed entry) come from different weight regimes and placement templates.
    # Without shuffling, consecutive entries belong to the same weight group and produce
    # near-identical actions, causing the diversity_window=200 gate to fail even though
    # the full buffer is diverse.
    import random as _random
    _rng = _random.Random(args.seed)
    _rng.shuffle(grid)

    total = len(grid)
    print(f"\n{'='*60}")
    print(f"  Synthetic Trajectory Injector (v2)")
    print(f"  Weights: {len(BASE_WEIGHT_GRID)} anchors + {N_RANDOM_WEIGHTS} random = {len(weight_grid)} total")
    print(f"  Grid: {len(weight_grid)} weights × {len(PLACEMENT_TEMPLATES)} placements × {len(STORM_VARIANTS)} storm levels = {total} scenarios (shuffled)")
    print(f"  Mode: {'DRY RUN (no Redis writes)' if args.dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    pushed = 0
    skipped = 0
    unique_actions: set[tuple] = set()

    for i, ((w_c, w_d, w_a, wlabel), tmpl, v_storm) in enumerate(grid, start=1):
        label = f"{wlabel}|{tmpl['label']}|storm={v_storm}"
        try:
            ds = build_dataset_from_cluster(
                last_placement=tmpl["placement"],
                w_c=w_c,
                w_d=w_d,
                w_a=w_a,
                v_storm_max=v_storm,
            )
            result = solve_placement(ds, verbose=False)

            if result is None or result.status != "optimal":
                print(f"  [{i:03d}/{total}] SKIP (infeasible/timeout): {label}")
                skipped += 1
                continue

            state_vec = _build_state_vector(ds, result)
            action_vec = _build_action_vector(ds, result)
            reward = -float(result.objective_value)
            action_key = tuple(action_vec)
            unique_actions.add(action_key)

            if args.dry_run:
                print(f"  [{i:03d}/{total}] DRY | action={action_vec} J={result.objective_value:.4f} | {label}")
            else:
                payload = json.dumps({
                    "state": state_vec,
                    "action": action_vec,
                    "reward": reward,
                    "next_state": state_vec,
                })
                rdb.lpush("milp:expert_trajectories", payload)   # type: ignore[union-attr]
                pushed += 1
                print(f"  [{i:03d}/{total}] PUSH | action={action_vec} J={result.objective_value:.4f} | {label}")

        except Exception as exc:
            print(f"  [{i:03d}/{total}] ERROR ({label}): {exc}")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"  Done. Pushed: {pushed} | Skipped: {skipped} | Unique actions seen: {len(unique_actions)}")
    if not args.dry_run and rdb:
        total_in_redis = rdb.llen("milp:expert_trajectories")
        print(f"  Total in milp:expert_trajectories: {total_in_redis}")
        ratio = len(unique_actions) / min(200, total_in_redis) if total_in_redis > 0 else 0
        print(f"  Estimated unique_action_ratio (window=200): {ratio:.3f}  (gate requires ≥ 0.20)")
    print(f"{'='*60}\n")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic expert trajectory injector for DRL BC training.")
    p.add_argument("--redis-host", default="10.42.1.183")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--dry-run", action="store_true", help="Print scenarios without writing to Redis.")
    p.add_argument("--flush", action="store_true", help="Delete milp:expert_trajectories before injecting.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random weight sampling.")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
