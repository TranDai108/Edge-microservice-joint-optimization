#!/usr/bin/env python3
"""
KLTN Scenario Runner
====================
Manually trigger specific placement scenarios for thesis evaluation.

Usage:
    python3 scripts/run_scenario.py --scenario <name> [--dry-run]

Available scenarios:
    energy_saving       - Force detection:yolo26-nano to minimise energy
    quality_maximise    - Force detection:yolo26-medium on the highest-RAM node
    balanced            - Balanced weights, let MILP decide freely
    node_migration      - Pin services to specific nodes via x_prev manipulation
    storm_test          - Trigger maximum simultaneous migrations
    ram_pressure        - Load RAM-yolo26-medium services to test C2_mem constraint
"""

import argparse, json, sys, os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))

import k8s_client

from solver.dataset_generator import NodeSpec, ServiceSpec, MILPDataset, save_dataset
from solver.metrics_collector import build_dataset_from_cluster
from solver.milp_model import solve_placement
from variant_catalog import DETECTION_ROLLOUT_TIMEOUT_S

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REGISTRY = "192.168.100.3:5000"


def _parse_redis_port(value: str | None, default: int = 6379) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


REDIS_HOST = os.getenv("REDIS_HOST", "redis.default.svc.cluster.local")
REDIS_PORT = _parse_redis_port(os.getenv("REDIS_PORT"))
NODE_MAP_FALLBACK = {
    "n0": "edge-nodes-1",
    "n1": "edge-nodes-2",
    "n2": "edge-nodes-3",
    "n3": "edge-nodes-4",
}
MILP_TO_DEPLOY = {
    "m0": "api-gateway",
    "m1": "ingest",
    "m2": "preprocess",
    "m3": "detection",
    "m4": "gen-ai",
    "m5": "postprocess",
}

# ─── Note on variant reference ───────────────────────────────────────────────────────────────
# Variant changes are applied via k8s_client.set_deployment_image() +
# k8s_client.set_deployment_env() — see apply_to_cluster() below.


def discover_node_map() -> dict[str, str]:
    """
    Build MILP node_id -> hostname mapping dynamically from current worker nodes.
    Falls back to static map if discovery fails.
    """
    hostnames = sorted(k8s_client.get_all_worker_nodes())
    if not hostnames:
        return dict(NODE_MAP_FALLBACK)
    return {f"n{i}": hostname for i, hostname in enumerate(hostnames)}


def scenario_hosts(node_map: dict[str, str]) -> tuple[str, str, str, str]:
    """Return 4 hostnames for scenario templates, repeating last host if needed."""
    hosts = [node_map[k] for k in sorted(node_map.keys())]
    if not hosts:
        hosts = [
            "edge-nodes-1",
            "edge-nodes-2",
            "edge-nodes-3",
            "edge-nodes-4",
        ]
    while len(hosts) < 4:
        hosts.append(hosts[-1])
    return hosts[0], hosts[1], hosts[2], hosts[3]


def node_name(node_id: str, node_map: dict[str, str]) -> str:
    return node_map.get(node_id, node_id)


def _connect_redis(redis_host: str, redis_port: int):
    """Create a Redis client for controller handoff."""
    try:
        from ha.redis_client import make_redis_client  # type: ignore

        if redis_host == REDIS_HOST and redis_port == REDIS_PORT:
            return make_redis_client()
    except Exception:
        pass

    import redis  # type: ignore

    return redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
        socket_timeout=2,
    )


def _build_node_scores(result, node_map: dict[str, str]) -> dict[str, int]:
    """Build scheduler-extender style scores from a scenario MILP result."""
    scores: dict[str, int] = {}
    for node_id, usage in result.resource_usage.items():
        hostname = node_name(node_id, node_map)
        scores[hostname] = max(0, 50 - int(float(usage) * 5))
    for _, (_, node_id) in result.placement.items():
        hostname = node_name(node_id, node_map)
        scores[hostname] = min(100, scores.get(hostname, 50) + 50)
    return scores


def build_controller_payload(result, ds, node_map: dict[str, str]) -> dict:
    """Serialize a scenario result using the same schema as milp:placement."""
    now_iso = datetime.utcnow().isoformat()
    nodes_sorted = sorted(ds.nodes, key=lambda n: n.node_id)
    return {
        "placement": {
            MILP_TO_DEPLOY.get(svc_id, svc_id): {
                "node": node_name(node_id, node_map),
                "variant": variant,
            }
            for svc_id, (variant, node_id) in result.placement.items()
        },
        "migration_types": {
            MILP_TO_DEPLOY.get(svc_id, svc_id): mig_type
            for svc_id, mig_type in result.migration_types.items()
        },
        "timestamp": now_iso,
        "objective": float(result.objective_value),
        "solve_time": float(result.solve_time),
        "status": result.status,
        "cost_energy": float(result.cost_energy),
        "cost_disruption": float(result.cost_disruption),
        "gain_accuracy": float(result.gain_accuracy),
        "norm_cost_energy": float(result.norm_cost_energy),
        "norm_cost_disruption": float(result.norm_cost_disruption),
        "norm_gain_accuracy": float(result.norm_gain_accuracy),
        "node_scores": _build_node_scores(result, node_map),
        "e_cpu_unit": [round(float(n.energy_cost), 4) for n in nodes_sorted],
        "resource_usage": {
            node: round(float(usage), 4)
            for node, usage in getattr(result, "resource_usage", {}).items()
        },
        "mem_usage": {
            node: round(float(usage), 4)
            for node, usage in getattr(result, "mem_usage", {}).items()
        },
        "background_load": {
            node: {
                "cpu_cores": round(
                    float(getattr(result, "background_cpu", {}).get(node, 0.0)), 4
                ),
                "mem_gb": round(
                    float(getattr(result, "background_mem", {}).get(node, 0.0)), 4
                ),
            }
            for node in sorted(
                set(getattr(result, "background_cpu", {}))
                | set(getattr(result, "background_mem", {}))
            )
        },
        "source": "scenario_runner",
    }


def handoff_to_controller(
    result,
    ds,
    node_map: dict[str, str],
    *,
    redis_host: str,
    redis_port: int,
    ttl_s: int,
    set_mode: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Publish a scenario placement to Redis and let the live controller reconcile it.
    """
    if result is None:
        print("  No result to publish.")
        return

    payload = build_controller_payload(result, ds, node_map)
    if dry_run:
        print("  [DRY RUN] Would publish scenario placement to Redis:")
        print(f"    ▸ key=milp:placement ttl={ttl_s}s redis={redis_host}:{redis_port}")
        if set_mode:
            print(f"    ▸ set system:mode={set_mode}")
        for deploy, info in sorted(payload["placement"].items()):
            print(f"    ▸ {deploy}: node={info['node']} variant={info['variant']}")
        return

    rdb = _connect_redis(redis_host, redis_port)
    rdb.ping()

    if set_mode:
        rdb.set("system:mode", set_mode)
        print(f"  system:mode set → {set_mode}")

    mode = (rdb.get("system:mode") or "milp").strip().lower()
    if mode == "drl":
        print(
            "  [WARN] system:mode=drl: variant-controller prefers drl:placement. "
            "Use --set-mode milp or --set-mode shadow if this scenario should drive reconciliation."
        )
    elif mode == "hybrid":
        print(
            "  [WARN] system:mode=hybrid: controller may choose drl:placement if it is fresh and better."
        )

    rdb.setex("milp:placement", ttl_s, json.dumps(payload))
    rdb.setex("milp:solve_timestamp", ttl_s, payload["timestamp"])

    for svc_id, mig_type in result.migration_types.items():
        if "AI Model" not in mig_type:
            continue
        variant, node_id = result.placement.get(svc_id, ("standard", ""))
        event = {
            "service": MILP_TO_DEPLOY.get(svc_id, svc_id),
            "variant": variant,
            "node": node_name(node_id, node_map),
        }
        rdb.publish("milp:events", json.dumps(event))

    print(
        f"  Published scenario placement → milp:placement "
        f"(ttl={ttl_s}s, mode={mode}, redis={redis_host}:{redis_port})"
    )


def print_result(result, ds, scenario_name: str, node_map: dict[str, str]):
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    if result is None:
        print("  ❌ INFEASIBLE — no placement found")
        return

    print(f"  Status  : {result.status}")
    print(f"  J       : {result.objective_value:.4f}")
    print(f"  C_norm  : {result.norm_cost_energy:.4f}  (energy)")
    print(f"  D_norm  : {result.norm_cost_disruption:.4f}  (disruption)")
    print(f"  A_norm  : {result.norm_gain_accuracy:.4f}  (quality)")
    print(f"  Solve   : {result.solve_time:.3f}s")
    print()
    print("  Node capacity snapshot:")
    for n in ds.nodes:
        host = node_name(n.node_id, node_map)
        print(
            f"    {n.node_id} ({host}) -> CPU cap={n.cap_cpu:.1f}c, "
            f"RAM cap={n.cap_mem_gb:.2f}GB"
        )
    print()
    print("  Placement:  (prev_variant @ prev_node  →  new_variant @ new_node)")
    print(f"  {'Service':<4}  {'Before':^30}  {'After':^30}  {'Status':<12}  Type")
    print(f"  {'-'*4}  {'-'*30}  {'-'*30}  {'-'*12}  {'-'*22}")
    for svc_id, (new_var, new_node) in sorted(result.placement.items()):
        is_migrated = result.migrations.get(svc_id, False)
        mtype       = result.migration_types.get(svc_id, "Stayed")
        new_node_h  = node_name(new_node, node_map)

        # Look up where this service was BEFORE this decision cycle
        prev = result.prev_placement.get(svc_id)
        if prev:
            prev_var, prev_node = prev
            prev_node_h = node_name(prev_node, node_map)
            before_str = f"{prev_var:8s} @ {prev_node_h}"
        else:
            # No previous placement recorded (fresh deployment / unknown)
            before_str = f"{'(new)':8s}   {'—':15s}"

        after_str = f"{new_var:8s} @ {new_node_h}"

        # Status icon
        if not is_migrated:
            status = "  stayed   "
            arrow  = "─►"
        else:
            status = "⬆ MIGRATED"
            # Indicate what specifically changed
            if mtype == "Node Migration":
                arrow = "─►"   # same variant, different node
            elif mtype == "AI Model Redeployment":
                arrow = "↑►"   # variant changed, same node
            elif mtype == "Node Migration + AI Redeployment":
                arrow = "↑►"   # both node AND variant changed
            else:
                arrow = "─►"

        print(f"  {svc_id:<4}  {before_str:<30}  {arrow} {after_str:<28}  {status}  [{mtype}]")
    print()
    print("  Node Resource Usage & Energy Breakdown:")
    cluster_e_cpu = 0.0
    cluster_e_mem = 0.0
    for n in ds.nodes:
        cpu_u  = result.resource_usage.get(n.node_id, 0)
        mem_u  = result.mem_usage.get(n.node_id, 0)
        node_h = node_name(n.node_id, node_map)

        # Resource utilisation bars (20-char width)
        cpu_bar = "\u2588" * int(cpu_u / n.cap_cpu * 20)    if n.cap_cpu    > 0 else ""
        mem_bar = "\u2588" * int(mem_u / n.cap_mem_gb * 20) if n.cap_mem_gb > 0 else ""

        # Dual-energy contribution for this node’s current load:
        #   E_cpu_contrib = E_cpu_unit [W/core] x R_cpu_used [cores]
        #   E_mem_contrib = E_mem_unit [W/GB]   x R_mem_used [GB]
        e_cpu_w = n.energy_cost * cpu_u
        e_mem_w = n.e_mem_unit  * mem_u
        e_total = e_cpu_w + e_mem_w
        cluster_e_cpu += e_cpu_w
        cluster_e_mem += e_mem_w

        print(f"    {n.node_id} ({node_h})")
        print(f"      CPU: {cpu_bar:<20s} {cpu_u:.3f}/{n.cap_cpu:.1f}c"
              f"   -> E_cpu: {e_cpu_w:6.2f}W  [{n.energy_cost:.2f} W/core]")
        print(f"      RAM: {mem_bar:<20s} {mem_u:.3f}/{n.cap_mem_gb:.2f}GB"
              f" -> E_mem: {e_mem_w:6.2f}W  [{n.e_mem_unit:.2f} W/GB]")
        print(f"      {'─'*20}   Node energy total : {e_total:.2f}W")
    print()
    print(f"  Cluster energy total  : {cluster_e_cpu + cluster_e_mem:.2f}W"
          f"  (CPU {cluster_e_cpu:.2f}W + DRAM {cluster_e_mem:.2f}W)")
    print()



# ─── Scenario Definitions ─────────────────────────────────────────────────────

def scenario_energy_saving(dry_run: bool, node_map: dict[str, str]):
    """
    SCENARIO 1: Energy Saving
    ─────────────────────────
    Type: AI Model Redeployment (variant change: yolo26-medium → yolo26-nano)
    Goal: Minimise energy by switching detection to its lightest variant.
    How:  Set w_c=0.7 (energy dominant), w_d=0.1, w_a=0.2.
            The MILP will naturally pick detection:yolo26-nano over yolo26-medium.
        Expected: detection migrates to yolo26-nano on cheapest-E_unit node.
    """
    print("\n[Scenario 1] Energy Saving — yolo26-medium → yolo26-nano redeployment")
    h0, h1, h2, h3 = scenario_hosts(node_map)
    ds = build_dataset_from_cluster(
        last_placement={
            "api-gateway":  {"node": h1, "variant": "standard"},
            "ingest":       {"node": h0, "variant": "standard"},
            "preprocess":   {"node": h2, "variant": "standard"},
            "detection":    {"node": h3, "variant": "yolo26-medium"},
            "gen-ai":       {"node": h3, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": h1, "variant": "standard"},
        },
        w_c=0.7, w_d=0.1, w_a=0.2,   # energy dominant
        v_storm_max=3,
    )
    result = solve_placement(ds)
    print_result(result, ds, "Energy Saving (w_c=0.7)", node_map)
    return ds, result


def scenario_quality_maximise(dry_run: bool, node_map: dict[str, str]):
    """
    SCENARIO 2: Quality Maximise
    ────────────────────────────
        Type: AI Model Redeployment (variant change: yolo26-nano → yolo26-medium)
            + possible Node Migration (to node with most RAM for yolo26-medium model)
    Goal: Maximise detection accuracy regardless of energy cost.
    How:  Set w_a=0.85, w_c=0.05, w_d=0.0 (no disruption cost).
          With zero disruption penalty, the accuracy gain from upgrading
            detection:yolo26-nano → yolo26-medium will typically dominate.
            The MILP will also prefer edge-nodes-2 (3.82 GB RAM) for yolo26-medium.
        Expected: detection:yolo26-medium on edge-nodes-2 (n1) — the RAM-rich node.
    """
    print("\n[Scenario 2] Quality Maximise — prefer yolo26-medium on RAM-rich node")
    h0, h1, h2, h3 = scenario_hosts(node_map)
    ds = build_dataset_from_cluster(
        last_placement={
            # Realistic starting point: detection:yolo26-nano, spread across cheap nodes
            "api-gateway":  {"node": h0, "variant": "standard"},
            "ingest":       {"node": h1, "variant": "standard"},
            "preprocess":   {"node": h2, "variant": "standard"},
            "detection":    {"node": h0, "variant": "yolo26-nano"},
            "gen-ai":       {"node": h3, "variant": "qwen-1.5b-nano"},
            "postprocess":  {"node": h3, "variant": "standard"},
        },
        w_c=0.3, w_d=0.1, w_a=0.95,   # quality dominant, zero disruption penalty
        v_storm_max=5,
    )
    result = solve_placement(ds)
    print_result(result, ds, "Quality Maximise (w_a=0.95, w_d=0.1)", node_map)
    return ds, result


def scenario_balanced(dry_run: bool, node_map: dict[str, str]):
    """
    SCENARIO 3: Balanced
    --------------------
    Type: General placement optimization with balanced objective weights.
    Goal: Provide a safe, readable demo baseline before stressing energy,
          quality, migration, or RAM constraints.
    How:  Use moderate energy/disruption weights and accuracy-dominant quality.
          The last_placement is spread across nodes so the output is easy to
          compare against the optimized solution.
    Expected: MILP may keep stable services in place while choosing efficient
              nodes/variants for detection and gen-ai.
    """
    print("\n[Scenario 3] Balanced — default trade-off across energy, disruption, quality")
    h0, h1, h2, h3 = scenario_hosts(node_map)
    ds = build_dataset_from_cluster(
        last_placement={
            "api-gateway":  {"node": h0, "variant": "standard"},
            "ingest":       {"node": h1, "variant": "standard"},
            "preprocess":   {"node": h2, "variant": "standard"},
            "detection":    {"node": h3, "variant": "yolo26-small"},
            "gen-ai":       {"node": h0, "variant": "llama-3b-small"},
            "postprocess":  {"node": h2, "variant": "standard"},
        },
        w_c=0.2,
        w_d=0.1,
        w_a=0.7,
        v_storm_max=3,
    )
    result = solve_placement(ds)
    print_result(result, ds, "Balanced (w_c=0.2, w_d=0.1, w_a=0.7)", node_map)
    return ds, result


def scenario_node_migration(dry_run: bool, node_map: dict[str, str]):
    """
    SCENARIO 3: Node Migration
    ──────────────────────────
    Type: Pure Node Migration (same variant, different node)
    Goal: Rebalance load — move detection away from edge-nodes-1 which
          is currently hosting both detection and ingest.
    How:  Set x_prev so detection + ingest are both on n0 (edge-nodes-1).
          With balanced weights, the MILP will migrate one to free up n0.
    Expected: At least one service migrates node, same variant retained.
    """
    print("\n[Scenario 3] Node Migration — rebalance overloaded node")
    h0, h1, h2, h3 = scenario_hosts(node_map)
    ds = build_dataset_from_cluster(
        last_placement={
            "api-gateway":  {"node": h0, "variant": "standard"},
            "ingest":       {"node": h0, "variant": "standard"},  # co-located with detection
            "preprocess":   {"node": h0, "variant": "standard"},  # all on n0
            "detection":    {"node": h0, "variant": "yolo26-medium"},
            "gen-ai":       {"node": h0, "variant": "llama-3b-small"},
            "postprocess":  {"node": h1, "variant": "standard"},
        },
        w_c=0.3, w_d=0.2, w_a=0.5,   # balanced
        v_storm_max=3,
    )
    result = solve_placement(ds)
    print_result(result, ds, "Node Migration (node rebalance)", node_map)
    return ds, result


def scenario_storm_test(dry_run: bool, node_map: dict[str, str]):
    """
    SCENARIO 4: Migration Storm Test
    ─────────────────────────────────
    Type: Maximum simultaneous migrations (storm boundary test)
    Goal: Force the MILP to migrate as many services as possible while
          staying within V_storm_max. Tests C5 stability constraint.
    How:  Set all services on wrong nodes (opposite of optimal),
          lift v_storm_max to 5, set w_d=0.0 (no disruption penalty).
    Expected: All 5 services migrate simultaneously to optimal nodes.
    """
    print("\n[Scenario 4] Migration Storm — max simultaneous migrations")
    h0, _, _, _ = scenario_hosts(node_map)
    ds = build_dataset_from_cluster(
        last_placement={
            # Intentionally bad: everything crammed onto edge-nodes-1
            # which has E_mem_unit=9.74 W/GB (highest of all 3 nodes).
            # The MILP strongly prefers to move to edge-nodes-2 (4.90 W/GB)
            # or at least spread load across nodes.
            "api-gateway":  {"node": h0, "variant": "standard"},
            "ingest":       {"node": h0, "variant": "standard"},
            "preprocess":   {"node": h0, "variant": "standard"},
            "detection":    {"node": h0, "variant": "yolo26-medium"},
            "gen-ai":       {"node": h0, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": h0, "variant": "standard"},
        },
        w_c=0.6, w_d=0.001, w_a=0.4,  # energy dominant; near-zero disruption
        v_storm_max=6,                # allow all services to migrate
    )
    result = solve_placement(ds)
    print_result(result, ds, "Storm Test (v_storm_max=5, w_c=0.6)", node_map)
    return ds, result


def scenario_ram_pressure(dry_run: bool, node_map: dict[str, str]):
    """
    SCENARIO 5: RAM Pressure (C2_mem constraint active)
    ────────────────────────────────────────────────────
    Type: Node Migration forced by RAM constraint
    Goal: Demonstrate C2_mem hard limit in action.
          Pack RAM-yolo26-medium services on nodes 1 and 3 (1.92 GB each).
          The MILP MUST move detection:yolo26-medium to edge-nodes-2 (3.82 GB)
          because C2_mem blocks it from fitting on 1.92 GB nodes when
          other services are also present.
    How:  Artificially inflate R_mem values to stress the constraint.
    Expected: detection:yolo26-medium migrates to n1 (edge-nodes-2, 3.82 GB).
    """
    print("\n[Scenario 5] RAM Pressure — C2_mem forces node choice")
    h0, h1, h2, h3 = scenario_hosts(node_map)

    # Get live data then artificially stress RAM
    ds = build_dataset_from_cluster(
        last_placement={
            "api-gateway":  {"node": h0, "variant": "standard"},
            "ingest":       {"node": h2, "variant": "standard"},
            "preprocess":   {"node": h3, "variant": "standard"},
            "detection":    {"node": h0, "variant": "yolo26-medium"},
            "gen-ai":       {"node": h3, "variant": "gemma2-2b-medium"},
            "postprocess":  {"node": h1, "variant": "standard"},
        },
        w_c=0.2, w_d=0.05, w_a=0.75,
        v_storm_max=3,
    )

    # Inflate RAM requirements to stress C2_mem:
    # detection:yolo26-medium → 1.5 GB (fills most of a 1.92 GB node by itself)
    # api-gateway     → 0.6 GB (combined with detection: 2.1 GB > 1.92 GB → INFEASIBLE on n0/n2)
    import copy
    ds_stressed = copy.copy(ds)
    new_r_mem = dict(ds.r_mem)
    svc_name_by_id = {
        "m0": "api-gateway",
        "m1": "ingest",
        "m2": "preprocess",
        "m3": "detection",
        "m4": "gen-ai",
        "m5": "postprocess",
    }
    for svc in ds.services:
        svc_name = svc_name_by_id.get(svc.service_id, svc.service_type)
        for var in svc.valid_variants:
            key = (svc.service_id, var)
            if svc_name == "detection":
                scale = {"yolo26-nano": 0.6, "yolo26-small": 1.1, "yolo26-medium": 1.5}[var]
                new_r_mem[key] = scale
            elif svc_name == "gen-ai":
                scale = {
                    "qwen-1.5b-nano": 1.2,
                    "llama-3b-small": 2.1,
                    "gemma2-2b-medium": 3.2,
                }[var]
                new_r_mem[key] = scale
            elif svc_name == "api-gateway":
                new_r_mem[key] = 0.6
    ds_stressed.r_mem = new_r_mem  # type: ignore

    result = solve_placement(ds_stressed)
    print("  [RAM values used for stress test]")
    for svc in ds_stressed.services:
        for var in svc.valid_variants:
            k = (svc.service_id, var)
            print(f"    {svc.service_id}/{var}: R_mem={ds_stressed.r_mem[k]:.3f}GB")
    print_result(result, ds_stressed, "RAM Pressure (C2_mem active)", node_map)
    return ds_stressed, result


# ─── Apply placement to K8s ───────────────────────────────────────────────────

def apply_to_cluster(result, ds, dry_run: bool, node_map: dict[str, str], force_all: bool = False):
    """
    Apply the MILP solution to the live K8s cluster via Kubernetes Python API.

    Args:
        force_all: When True (used with --apply in scenario mode), applies
                   EVERY service to the MILP's decided position — regardless
                   of whether the MILP flagged it as 'migrated' or 'stayed'.
                   This is necessary because the scenario's hardcoded x_prev
                   often does not match the real cluster state. The MILP's
                   output is the ground truth for where each service SHOULD be.
                   When False (used by edge_controller), only applies services
                   the MILP explicitly decided to move, to avoid unnecessary
                   rollout restarts during normal operation.
    """
    if result is None:
        print("  No result to apply.")
        return
    if dry_run:
        print("  [DRY RUN] Would apply the following Kubernetes API actions:")
    else:
        mode = "FULL placement" if force_all else "migrations only"
        print(f"  Applying placement to cluster ({mode}) via Kubernetes API...")

    milp_to_deploy = {
        "m0": "api-gateway", "m1": "ingest", "m2": "preprocess",
        "m3": "detection",   "m4": "gen-ai", "m5": "postprocess",
    }

    for svc_id, (var, node_id) in sorted(result.placement.items()):
        # In normal mode: skip services the MILP marked as stayed.
        # In force_all mode: apply every service unconditionally.
        is_migrated = result.migrations.get(svc_id, False)
        if not force_all and not is_migrated:
            print(f"  [{milp_to_deploy.get(svc_id, svc_id)}] Stayed — skipping.")
            continue

        deploy = milp_to_deploy.get(svc_id, svc_id)
        node_h = node_name(node_id, node_map)
        image  = f"{REGISTRY}/detection:{var}" if deploy == "detection" else f"{REGISTRY}/{deploy}:latest"
        action = "MIGRATE" if is_migrated else "FORCE"
        print(f"  [{deploy}] {action} → node={node_h}  variant={var}")

        if dry_run:
            print(f"    ▸ patch_deployment_node_selector({deploy!r}, {node_h!r})")
            print(f"    ▸ set_deployment_image({deploy!r}, {deploy!r}, {image!r})")
            if deploy == "detection":
                print(f"    ▸ set_deployment_env('detection', 'VARIANT_ID', {var!r})")
            timeout = DETECTION_ROLLOUT_TIMEOUT_S.get(var, 45)
            print(f"    ▸ rollout_restart_deployment({deploy!r})")
            print(f"    ▸ wait_rollout_complete({deploy!r}, timeout_s={timeout})")
        else:
            k8s_client.patch_deployment_node_selector(deploy, node_h)
            k8s_client.set_deployment_image(deploy, deploy, image)
            if deploy == "detection":
                k8s_client.set_deployment_env(deploy, "VARIANT_ID", var)
            k8s_client.rollout_restart_deployment(deploy)
            timeout = DETECTION_ROLLOUT_TIMEOUT_S.get(var, 45)
            k8s_client.wait_rollout_complete(deploy, timeout_s=timeout)


# ─── Main ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "energy_saving":      scenario_energy_saving,
    "quality_maximise":   scenario_quality_maximise,
    "balanced":           scenario_balanced,
    "node_migration":     scenario_node_migration,
    "storm_test":         scenario_storm_test,
    "ram_pressure":       scenario_ram_pressure,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLTN Scenario Runner")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()) + ["all"],
                        default="all", help="Which scenario to run")
    parser.add_argument("--apply", action="store_true",
                        help="Apply MILP result to live K8s cluster, forcing ALL services "
                             "to the MILP's decided position (not just flagged migrations).")
    parser.add_argument("--handoff-to-controller", "--publish", action="store_true",
                        help="Publish the scenario result to Redis as milp:placement and let "
                             "the live controller/variant-controller reconcile Kubernetes.")
    parser.add_argument("--redis-host", default=REDIS_HOST,
                        help=f"Redis host for --handoff-to-controller (default: {REDIS_HOST})")
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT,
                        help=f"Redis port for --handoff-to-controller (default: {REDIS_PORT})")
    parser.add_argument("--handoff-ttl", type=int, default=35,
                        help="TTL in seconds for the published milp:placement payload.")
    parser.add_argument("--set-mode", choices=["milp", "shadow", "hybrid", "drl"],
                        help="Optionally set Redis system:mode before publishing. Use milp or "
                             "shadow when this scenario should be the controller's desired state.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what kubectl commands would run, without executing")
    args = parser.parse_args()
    if args.apply and args.handoff_to_controller:
        parser.error("Use either --apply or --handoff-to-controller, not both.")

    node_map = discover_node_map()
    print("Detected worker topology:")
    for node_id in sorted(node_map.keys()):
        print(f"  {node_id} -> {node_map[node_id]}")

    scenarios_to_run = list(SCENARIOS.items()) if args.scenario == "all" else \
                       [(args.scenario, SCENARIOS[args.scenario])]

    for name, fn in scenarios_to_run:
        ds, result = fn(dry_run=args.dry_run, node_map=node_map)
        if args.apply or (args.dry_run and not args.handoff_to_controller):
            # force_all=True: apply the FULL MILP solution to the cluster.
            # Scenarios use hardcoded x_prev that may not match the real cluster
            # state, so we cannot rely on the MILP's migration flags alone.
            # We want the cluster to end up exactly where the math says it should be.
            apply_to_cluster(result, ds, dry_run=args.dry_run, node_map=node_map, force_all=True)
        if args.handoff_to_controller:
            handoff_to_controller(
                result,
                ds,
                node_map,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
                ttl_s=args.handoff_ttl,
                set_mode=args.set_mode,
                dry_run=args.dry_run,
            )
        # Save dataset for reproducibility
        if result is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dataset(ds, RESULTS_DIR / f"scenario_{name}_{ts}.json")
            print(f"  Dataset saved → results/scenario_{name}_{ts}.json")
