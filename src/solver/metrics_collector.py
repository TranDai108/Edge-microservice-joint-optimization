# solver/metrics_collector.py
"""
Translates live Prometheus + Kubernetes API data into a MILPDataset.
Called by run_solver.py once per control cycle before each MILP solve.

Metric source responsibilities:
  - node_exporter    : node CPU utilization → E_unit[n] (Watts/core, linear model)
  - cAdvisor         : container CPU usage  → R_req[m,k] (cores, real measurement)
  - kltn-pipeline    : app-level metrics    → Q[m,k] accuracy, e2e latency, request rate
  - Kubernetes API   : node allocatable CPU/RAM → Cap[n]  (replaces kubectl)

Phase 2 upgrade path:
  - node_exporter E_unit → replace with Kepler kepler_node_platform_joules_total
    (function get_energy_cost_per_core will be swapped; interface stays the same)

Prometheus query note:
    node_cpu_seconds_total should be queried by node=hostname where available.
    For clusters that still expose only instance=IP:9100, this collector falls back
    to dynamic Kubernetes InternalIP discovery (no hardcoded IP mapping).
"""

import json, logging, math, requests, sys, os

if __package__ is None or __package__ == "":
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, _project_root)
    from dataset_generator import MILPDataset, NodeSpec, ServiceSpec
    from variant_catalog import (
        DEFAULT_DETECTION_VARIANT,
        DETECTION_VARIANTS,
        DETECTION_OFFLINE_ACCURACY,
        DETECTION_CPU_SCALE,
        DETECTION_MEM_SCALE,
        DEFAULT_GEN_AI_VARIANT,
        GEN_AI_VARIANTS,
        GEN_AI_OFFLINE_QUALITY,
        GEN_AI_CPU_SCALE,
        GEN_AI_MEM_SCALE,
    )
    import k8s_client
else:
    from solver.dataset_generator import MILPDataset, NodeSpec, ServiceSpec
    # k8s_client lives at project root — import by inserting root path
    _project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from variant_catalog import (
        DEFAULT_DETECTION_VARIANT,
        DETECTION_VARIANTS,
        DETECTION_OFFLINE_ACCURACY,
        DETECTION_CPU_SCALE,
        DETECTION_MEM_SCALE,
        DEFAULT_GEN_AI_VARIANT,
        GEN_AI_VARIANTS,
        GEN_AI_OFFLINE_QUALITY,
        GEN_AI_CPU_SCALE,
        GEN_AI_MEM_SCALE,
    )
    import k8s_client

from config import MILP_W_C, MILP_W_D, MILP_W_A
log = logging.getLogger("metrics-collector")

# ── Prometheus endpoint (configurable for local vs in-cluster execution) ──
PROM = os.getenv("PROM_URL", "http://localhost:32090/api/v1/query")

# ── Static fallback topology used only if discovery fails ──
NODES_FALLBACK = [
    {"id": "n0", "hostname": "edge-nodes-1"},
    {"id": "n1", "hostname": "edge-nodes-2"},
    {"id": "n2", "hostname": "edge-nodes-3"},
    {"id": "n3", "hostname": "edge-nodes-4"},
]

# Backward compatibility for existing experiment scripts that import NODES.
NODES = NODES_FALLBACK

# ── Service definitions — each service carries its own valid variants ──
# Index i → service_id "m{i}". Must stay consistent with edge_controller.py.
# Non-AI services use ["standard"] — a single variant with Q=1.0.
# AI services declare their model family; Q is read live from Prometheus.
SERVICE_DEFS = [
    {"name": "api-gateway", "type": "gateway",    "variants": ["standard"]},
    {"name": "ingest",      "type": "ingest",     "variants": ["standard"]},
    {"name": "preprocess",  "type": "preprocess", "variants": ["standard"]},
    {"name": "detection",   "type": "detection",  "variants": DETECTION_VARIANTS},
    {"name": "gen-ai",      "type": "gen_ai",     "variants": GEN_AI_VARIANTS},
    {"name": "postprocess", "type": "postprocess","variants": ["standard"]},
]

# ── Cold-start migration cost per service ──
# Measured: time kubectl rollout restart && kubectl rollout status
# Divided by 10 to keep same order of magnitude as r_req (CPU cores)
MIGRATION_COST = {
    "api-gateway": 0.3,
    "ingest":      0.3,
    "preprocess":  0.3,
    "detection":   1.8,   # ~18s cold-start for heavy YOLO model
    "gen-ai":      8.0,   # model warm-up / prompt-engine cold start
    "postprocess": 0.3,
}

# ── Node power profiles ─ LINEAR FALLBACK ONLY ──────────────────────────────
# Used when Kepler is unavailable. Phase 2 (Kepler) replaces these with
# live kepler_node_platform_joules_total measurements. Keep for resilience.
NODE_POWER_PROFILES = {
    "edge-nodes-1": {"p_idle": 8.0, "p_max": 30.0},
    "edge-nodes-2": {"p_idle": 8.0, "p_max": 30.0},
    "edge-nodes-3": {"p_idle": 8.0, "p_max": 30.0},
    "edge-nodes-4": {"p_idle": 8.0, "p_max": 30.0},  # ← adjust to actual hardware spec
}

# ── Offline quality table — fallback when Prometheus has no live data ──
# Used when a detection variant has not been active recently.
ACCURACY_OFFLINE = {
    ("detection", variant): score
    for variant, score in DETECTION_OFFLINE_ACCURACY.items()
}
ACCURACY_OFFLINE.update({
    ("gen_ai", variant): score
    for variant, score in GEN_AI_OFFLINE_QUALITY.items()
})

# ── Static CPU fallback — only used when cAdvisor is unavailable ──
CPU_FALLBACK = {
    "api-gateway": 0.10,
    "ingest":      0.08,
    "preprocess":  0.15,
    "detection":   0.50,
    "gen-ai":      0.45,
    "postprocess": 0.08,
}

# ── Cache node capacity (allocatable CPU cores — stable during a run) ──
_cap_cache: dict[str, float] = {}
_mem_cache: dict[str, float] = {}
_nodes_cache: list[dict[str, str]] | None = None
_quality_counters: dict[str, int] = {
    "node_cpu_fallback": 0,
    "pod_cpu_fallback": 0,
    "pod_mem_fallback": 0,
    "accuracy_offline_fallback": 0,
}


def reset_caches() -> None:
    """Clear all per-cycle node caches so a topology change (e.g. node failure)
    is picked up on the next build_dataset_from_cluster() call."""
    global _nodes_cache, _cap_cache, _mem_cache
    _nodes_cache = None
    _cap_cache.clear()
    _mem_cache.clear()


def _reset_quality_counters() -> None:
    for key in _quality_counters:
        _quality_counters[key] = 0


def _mark_quality_fallback(key: str) -> None:
    if key in _quality_counters:
        _quality_counters[key] += 1


def get_metrics_quality_report() -> dict:
    total_fallbacks = sum(_quality_counters.values())
    total_samples = max(len(_get_nodes()), 1) + len(SERVICE_DEFS) * 3
    fallback_ratio = total_fallbacks / total_samples
    return {
        "counters": dict(_quality_counters),
        "total_fallbacks": total_fallbacks,
        "total_samples": total_samples,
        "fallback_ratio": round(fallback_ratio, 4),
    }


# ─────────────────────────── Internal helpers ──────────────────────────────────

def _prom(query: str) -> float | None:
    """Run a PromQL instant query, return first scalar result or None."""
    try:
        r = requests.get(PROM, params={"query": query}, timeout=5)
        results = r.json().get("data", {}).get("result", [])
        return float(results[0]["value"][1]) if results else None
    except Exception as e:
        log.warning(f"Prometheus query failed [{query[:60]}]: {e}")
        return None


def _get_nodes() -> list[dict[str, str]]:
    """Discover worker nodes dynamically; fall back to static map if needed."""
    global _nodes_cache
    if _nodes_cache is not None:
        return _nodes_cache

    hostnames = k8s_client.get_all_worker_nodes()
    if not hostnames:
        log.warning("Worker-node discovery failed — using static node fallback list")
        _nodes_cache = NODES_FALLBACK
        return _nodes_cache

    _nodes_cache = [
        {"id": f"n{i}", "hostname": h}
        for i, h in enumerate(sorted(hostnames))
    ]
    log.info("Discovered worker nodes: " + ", ".join(n["hostname"] for n in _nodes_cache))
    return _nodes_cache


# ─────────────────────────── Node metrics ─────────────────────────────────────

def get_node_capacity(hostname: str) -> float:
    """
    Allocatable CPU cores for a node. Cached after first read since
    this value does not change during an experiment run.
    Uses the Kubernetes API (k8s_client) instead of kubectl.
    """
    if hostname in _cap_cache:
        return _cap_cache[hostname]
    cap = k8s_client.get_node_capacity_cpu(hostname)
    _cap_cache[hostname] = cap
    return cap


def get_node_memory_gb(hostname: str) -> float:
    """
    Allocatable RAM (GB) for a node. Cached after first read.
    Uses the Kubernetes API (k8s_client) instead of kubectl.

    Real values confirmed:
      edge-nodes-1: 2010704Ki = 1.92 GB
      edge-nodes-2: 4005700Ki = 3.82 GB  ← 2× more, drives differentiation
      edge-nodes-3: 2010700Ki = 1.92 GB
    """
    if hostname in _mem_cache:
        return _mem_cache[hostname]
    mem_gb = k8s_client.get_node_allocatable_memory(hostname)
    _mem_cache[hostname] = mem_gb
    log.debug(f"  {hostname}: allocatable RAM = {mem_gb:.2f} GB")
    return mem_gb



def get_node_cpu_util(hostname: str) -> float:
    """
    CPU utilisation (0.0–1.0) for a node over the last 60s.

    Uses node_exporter with preferred node=hostname label.
    Falls back to instance=InternalIP:9100 if node label is unavailable.
    """
    q_node = f'1 - rate(node_cpu_seconds_total{{mode="idle",node="{hostname}"}}[60s])'
    val = _prom(q_node)
    if val is None:
        ip = k8s_client.get_node_internal_ip(hostname)
        if ip:
            q_ip = f'1 - rate(node_cpu_seconds_total{{mode="idle",instance="{ip}:9100"}}[60s])'
            val = _prom(q_ip)
    if val is None:
        _mark_quality_fallback("node_cpu_fallback")
        log.warning(f"No CPU data for {hostname} — using fallback 0.5")
        return 0.5
    return max(0.0, min(1.0, val))


def get_energy_cost_per_core(hostname: str) -> float:
    """
    E_unit[n] = Watts per allocatable CPU core.

    Phase 2 (Kepler): uses kepler_node_platform_joules_total via eBPF + ML
    estimation (OpenStack VMs have no RAPL, so Kepler auto-switches to its
    pre-trained power model). The instance= label in Kepler is the K8s node
    hostname — confirmed from Prometheus label inspection.

    Falls back to node_exporter CPU% linear model if Kepler unavailable.
    """
    # ── Try Kepler first (Phase 2 primary source) ──
    q_kepler = f'rate(kepler_node_platform_joules_total{{instance="{hostname}"}}[60s])'
    node_watts = _prom(q_kepler)
    if node_watts is not None and node_watts > 0:
        cap    = get_node_capacity(hostname)
        e_unit = node_watts / cap if cap > 0 else node_watts
        log.info(
            f"  [Kepler] {hostname}: {node_watts:.2f}W / {cap}cores "
            f"= E_unit={e_unit:.4f}"
        )
        return round(e_unit, 4)

    # ── Fallback: node_exporter linear power model ──
    log.warning(
        f"  [Kepler] No data for {hostname} — falling back to linear model"
    )
    return _get_energy_cost_linear(hostname)


def _get_energy_cost_linear(hostname: str) -> float:
    """Linear power model fallback. Used when Kepler is unavailable."""
    profile  = NODE_POWER_PROFILES.get(hostname, {"p_idle": 8.0, "p_max": 30.0})
    cpu_util = get_node_cpu_util(hostname)
    p_node   = profile["p_idle"] + (profile["p_max"] - profile["p_idle"]) * cpu_util
    cap      = get_node_capacity(hostname)
    e_unit   = p_node / cap if cap > 0 else p_node
    log.debug(
        f"  [Linear] {hostname}: cpu_util={cpu_util:.3f} "
        f"P={p_node:.2f}W cap={cap} E_unit={e_unit:.4f}"
    )
    return round(e_unit, 4)


def get_node_dram_energy_per_gb(hostname: str) -> float:
    """
    E_mem_unit[n] = DRAM Watts per allocatable GB of RAM at node n.

    Source: kepler_node_dram_joules_total (Kepler eBPF/PMU measurement)
    divided by the node's allocatable RAM (from kubectl).

    Proportional allocation model: each container is charged
        E_mem_unit * R_mem[m,k]
    as its share of the node's DRAM power budget. This mirrors the
    CPU accounting (E_cpu_unit * R_cpu) and is the standard approach
    used in cloud billing and energy attribution literature.

    Real values from cluster:
      edge-nodes-1: 18.705 W / 1.92 GB = 9.742 W/GB
      edge-nodes-2: 18.708 W / 3.82 GB = 4.897 W/GB  ← lower! (2x more RAM)
      edge-nodes-3: 18.704 W / 1.92 GB = 9.742 W/GB

    The asymmetry is physically real: edge-nodes-2 has the same DRAM
    idle power (~18.7 W) but twice the capacity, making it more
    energy-efficient per GB. The solver will exploit this.
    """
    q_dram = f'rate(kepler_node_dram_joules_total{{instance="{hostname}"}}[60s])'
    dram_watts = _prom(q_dram)
    if dram_watts is None or dram_watts <= 0:
        log.warning(
            f"  [Kepler DRAM] No data for {hostname} — "
            f"using fallback 18.7W / cap_mem"
        )
        mem_gb = get_node_memory_gb(hostname)
        return round(18.7 / mem_gb if mem_gb > 0 else 9.7, 4)

    mem_gb = get_node_memory_gb(hostname)
    e_mem = dram_watts / mem_gb if mem_gb > 0 else dram_watts
    log.info(
        f"  [Kepler DRAM] {hostname}: {dram_watts:.4f}W / {mem_gb:.2f}GB "
        f"= E_mem_unit={e_mem:.4f} W/GB"
    )
    return round(e_mem, 4)


# ─────────────────────────── Service / pod metrics ────────────────────────────

def get_pod_cpu_cores(svc_name: str) -> float:
    """
    Live CPU usage (cores) for a service container, from cAdvisor.
    Falls back to static values only if cAdvisor data is unavailable.
    """
    q = (
        f'sum(rate(container_cpu_usage_seconds_total{{'
        f'container="{svc_name}",namespace="default"}}[60s]))'
    )
    val = _prom(q)
    if val is not None:
        cores = round(max(val, 0.05), 4)
        log.debug(f"  [cAdvisor CPU] {svc_name}: {cores:.4f} cores")
        return cores

    fallback = CPU_FALLBACK.get(svc_name, 0.1)
    _mark_quality_fallback("pod_cpu_fallback")
    log.warning(f"  [cAdvisor CPU] No data for {svc_name} — static fallback {fallback}")
    return fallback


def get_pod_memory_gb(svc_name: str) -> float:
    """
    Live RAM usage (GB) for a service container, from cAdvisor.

    Uses container_memory_working_set_bytes — the most accurate live memory
    metric (excludes file cache, includes anonymous + kernel memory).

    Values confirmed from cluster:
      detection:   473 MB (YOLO model loaded in memory)
      api-gateway: 198 MB
      preprocess:   52 MB
      ingest:       42 MB
      postprocess:  41 MB
    """
    q = (
        f'sum(container_memory_working_set_bytes{{'
        f'container="{svc_name}",namespace="default"}}) / 1073741824'
    )  # 1073741824 = 1024^3 (bytes -> GB)
    val = _prom(q)
    if val is not None and val > 0:
        gb = round(max(val, 0.01), 4)   # floor at 10MB
        log.debug(f"  [cAdvisor MEM] {svc_name}: {gb:.4f} GB")
        return gb

    # Static fallback in GB
    fallback_mb = {
        "api-gateway": 200, "ingest": 45, "preprocess": 55,
        "detection": 475, "gen-ai": 512, "postprocess": 45,
    }
    gb = fallback_mb.get(svc_name, 100) / 1024
    _mark_quality_fallback("pod_mem_fallback")
    log.warning(f"  [cAdvisor MEM] No data for {svc_name} — static fallback {gb:.3f}GB")
    return gb



def get_container_energy_watts(svc_name: str, hostname: str) -> float | None:
    """
    Total Watts consumed by a container, from Kepler (Phase 2).

    Kepler uses eBPF to measure per-container energy including CPU, DRAM,
    and uncore components — more complete than cAdvisor CPU-only accounting.

    Label format confirmed: container_name="{svc_name}", instance="{hostname}"
    mode="dynamic" selects only active (non-idle) energy.

    Returns None if Kepler data is not yet available for this container.
    """
    q = (
        f'sum(rate(kepler_container_joules_total{{'
        f'container_name="{svc_name}",'
        f'container_namespace="default",'
        f'instance="{hostname}",'
        f'mode="dynamic"}}[60s]))'
    )
    watts = _prom(q)
    if watts is not None and watts >= 0:
        log.debug(f"  [Kepler container] {svc_name}@{hostname}: {watts:.4f}W")
        return round(watts, 4)
    return None


def get_variant_accuracy(variant: str) -> float:
    """
    Mean detection confidence for a YOLO variant over last 5 minutes.
    Sourced from kltn-pipeline job (app-level metric from detection/app.py).
    Falls back to offline table if variant not recently active.
    """
    q = (
        f'rate(detection_accuracy_sum{{variant="{variant}"}}[300s]) / '
        f'rate(detection_accuracy_count{{variant="{variant}"}}[300s])'
    )
    live = _prom(q)
    if live is not None and 0.1 <= live <= 1.0:
        log.debug(f"  [kltn-pipeline] Accuracy [{variant}] live: {live:.4f}")
        return round(live, 4)
    offline = ACCURACY_OFFLINE.get(("detection", variant), 0.7)
    _mark_quality_fallback("accuracy_offline_fallback")
    log.debug(f"  [kltn-pipeline] Accuracy [{variant}] offline: {offline}")
    return offline


def get_gen_ai_variant_quality(variant: str) -> float:
    """
    Offline quality proxy for gen-ai text quality per variant.
    Values are benchmarked scores from GEN_AI_OFFLINE_QUALITY in variant_catalog.py.
    Fallback: 0.5 (lowest known variant quality) for any unrecognised variant name.
    A live metric can be introduced later once gen_ai exports quality signals.
    """
    return ACCURACY_OFFLINE.get(("gen_ai", variant), 0.5)


# ── Observation-only metrics (controller logging — not MILP inputs) ──────────

def get_e2e_latency_ms() -> float:
    """End-to-end pipeline latency from api-gateway's E2E_LATENCY histogram."""
    q   = 'rate(e2e_latency_ms_sum[60s]) / rate(e2e_latency_ms_count[60s])'
    val = _prom(q)
    if val is None or math.isnan(val):
        return 999.0
    return round(val, 2)


def get_detection_accuracy() -> float:
    """Mean detection confidence across all variants over last 60s."""
    q   = 'rate(detection_accuracy_sum[60s]) / rate(detection_accuracy_count[60s])'
    val = _prom(q)
    return round(val, 4) if val is not None else 0.5


def get_active_variant() -> str:
    """
    Read VARIANT_ID env var from the running detection pod.
    Uses the Kubernetes API (k8s_client) instead of kubectl.
    """
    env_list = k8s_client.get_pod_env("detection")
    for item in env_list:
        if item.get("name") == "VARIANT_ID":
            return item.get("value", DEFAULT_DETECTION_VARIANT)
    return DEFAULT_DETECTION_VARIANT


def get_node_cpu_util_by_hostname(hostname: str) -> float:
    """Wrapper used by edge_controller for per-node CPU logging."""
    return get_node_cpu_util(hostname)


# ─────────────────────────── x_prev from controller state ─────────────────────

def build_x_prev(last_placement: dict) -> dict[tuple[str, str, str], float]:
    """
    Build x_prev[m,k,n] from the controller's last_placement dict.

    Iterates per-service valid_variants (from SERVICE_DEFS), so non-AI services
    only get x_prev entries for ["standard"], not for AI variants.
    This was a correctness bug in the old version (which used a global VARIANTS list).

    last_placement format:
        {
          "api-gateway": {"node": "edge-nodes-1", "variant": "standard"},
                    "detection":   {"node": "edge-nodes-3", "variant": "yolo26-medium"},
          ...
        }
    """
    nodes = _get_nodes()
    hostname_to_id = {n["hostname"]: n["id"] for n in nodes}
    x_prev: dict[tuple[str, str, str], float] = {}

    # Initialize sparse x_prev to 0.0 — only valid (m, k, n) combinations
    for i, sdef in enumerate(SERVICE_DEFS):
        svc_id = f"m{i}"
        for var in sdef["variants"]:
            for n in nodes:
                x_prev[(svc_id, var, n["id"])] = 0.0

    # Set 1.0 for each service's last known placement
    for i, sdef in enumerate(SERVICE_DEFS):
        svc_id   = f"m{i}"
        svc_name = sdef["name"]
        state    = last_placement.get(svc_name, {})
        h        = state.get("node", "")
        v        = state.get("variant", "standard")  # default "standard" for non-AI

        if h not in hostname_to_id:
            log.warning(
                f"  {svc_name}: unknown host '{h}' in last_placement — "
                f"x_prev treated as fresh (all zeros)"
            )
            continue

        nid = hostname_to_id[h]
        if v in sdef["variants"]:
            x_prev[(svc_id, v, nid)] = 1.0
        else:
            # Variant stored by old controller is not valid for this service's type.
            # Common cause: non-AI services were previously tagged with AI variants before
            # SERVICE_DEFS migration. Remap to first valid variant to preserve the
            # node placement in x_prev — prevents false migrations and keeps the
            # migration storm constraint (v_storm_max) satisfiable.
            fallback_var = sdef["variants"][0]
            x_prev[(svc_id, fallback_var, nid)] = 1.0
            log.warning(
                f"  {svc_name}: variant '{v}' ∉ valid_variants "
                f"{sdef['variants']} — remapped to '{fallback_var}' on {h}"
            )

    return x_prev


# ─────────────────────────── Main entry point ─────────────────────────────────

def build_dataset_from_cluster(
    last_placement: dict,
    w_c: float = MILP_W_C,
    w_d: float = MILP_W_D,
    w_a: float = MILP_W_A,
    theta_max: float = 1.3,
    v_storm_max: int = 2,
) -> MILPDataset:
    """
    Build a real MILPDataset from live cluster state.
    Called by run_solver.py once per control cycle before each MILP solve.
    """
    reset_caches()
    _reset_quality_counters()
    log.info("Building dataset from cluster...")

    nodes_meta = _get_nodes()
    scenario_load = k8s_client.get_scenario_load_by_node()
    background_cpu: dict[str, float] = {}
    background_mem: dict[str, float] = {}
    for n in nodes_meta:
        host = n["hostname"]
        load = scenario_load.get(host, {})
        background_cpu[n["id"]] = max(0.0, float(load.get("cpu_cores", 0.0)))
        background_mem[n["id"]] = max(0.0, float(load.get("mem_gb", 0.0)))

    # ── Nodes: read both CPU capacity and RAM capacity ──
    nodes = []
    for n in nodes_meta:
        cap_cpu  = get_node_capacity(n["hostname"])
        cap_mem  = get_node_memory_gb(n["hostname"])
        e_cpu    = get_energy_cost_per_core(n["hostname"])
        e_mem    = get_node_dram_energy_per_gb(n["hostname"])
        nodes.append(NodeSpec(
            node_id=n["id"],
            cap_cpu=cap_cpu,
            cap_mem_gb=cap_mem,
            energy_cost=e_cpu,
            e_mem_unit=e_mem,
        ))
        log.info(
            f"  Node {n['id']} ({n['hostname']}): "
            f"CPU={cap_cpu}c RAM={cap_mem:.2f}GB "
            f"E_cpu={e_cpu} W/core  E_mem={e_mem} W/GB"
        )
        bg_cpu = background_cpu.get(n["id"], 0.0)
        bg_mem = background_mem.get(n["id"], 0.0)
        if bg_cpu > 0.0 or bg_mem > 0.0:
            log.info(
                f"  Scenario load {n['id']} ({n['hostname']}): "
                f"CPU={bg_cpu:.4f}c MEM={bg_mem:.4f}GB "
                f"headroom≈{max(0.0, cap_cpu - bg_cpu):.4f}c/"
                f"{max(0.0, cap_mem - bg_mem):.4f}GB"
            )

    # ── Services (with per-service valid_variants from SERVICE_DEFS) ──
    services = []
    for i, sdef in enumerate(SERVICE_DEFS):
        services.append(ServiceSpec(
            service_id=f"m{i}",
            service_type=sdef["type"],
            migration_cost=MIGRATION_COST.get(sdef["name"], 0.3),
            valid_variants=sdef["variants"],
        ))

    # ── R_req[m,k] (CPU), R_mem[m,k] (RAM) and Q[m,k] ──
    r_req: dict[tuple[str, str], float] = {}
    r_mem: dict[tuple[str, str], float] = {}
    acc:   dict[tuple[str, str], float] = {}

    for i, sdef in enumerate(SERVICE_DEFS):
        svc_id   = f"m{i}"
        svc_name = sdef["name"]
        base_cpu = get_pod_cpu_cores(svc_name)
        base_mem = get_pod_memory_gb(svc_name)

        for var in sdef["variants"]:
            if svc_name == "detection":
                # YOLO scales with variant — yolo26-medium uses 2× CPU and ~1.5× RAM
                cpu_scale = DETECTION_CPU_SCALE.get(var, 1.0)
                mem_scale = DETECTION_MEM_SCALE.get(var, 1.0)
                r_req[(svc_id, var)] = round(base_cpu * cpu_scale, 4)
                r_mem[(svc_id, var)] = round(base_mem * mem_scale, 4)
                acc[(svc_id, var)]   = get_variant_accuracy(var)
            elif svc_name == "gen-ai":
                cpu_scale = GEN_AI_CPU_SCALE.get(var, 1.0)
                mem_scale = GEN_AI_MEM_SCALE.get(var, 1.0)
                r_req[(svc_id, var)] = round(base_cpu * cpu_scale, 4)
                r_mem[(svc_id, var)] = round(base_mem * mem_scale, 4)
                acc[(svc_id, var)]   = get_gen_ai_variant_quality(var)
            else:
                # Non-AI service: resource usage same across variants, Q=1.0
                r_req[(svc_id, var)] = round(base_cpu, 4)
                r_mem[(svc_id, var)] = round(base_mem, 4)
                acc[(svc_id, var)]   = 1.0

        log.info(
            f"  [{svc_name}]: "
            + " / ".join(
                f"{v}(CPU={r_req[(svc_id,v)]:.4f}c "
                f"MEM={r_mem[(svc_id,v)]:.4f}GB "
                f"Q={acc[(svc_id,v)]})"
                for v in sdef["variants"]
            )
        )

    # ── x_prev from controller's last known placement ──
    x_prev = build_x_prev(last_placement)

    log.info(
        f"Dataset ready: {len(nodes)} nodes, {len(services)} services. "
        f"theta_max={theta_max} v_storm_max={v_storm_max}"
    )
    quality = get_metrics_quality_report()
    log.info(
        "Metrics quality: "
        f"fallback_ratio={quality['fallback_ratio']:.4f} "
        f"fallbacks={quality['total_fallbacks']}/{quality['total_samples']}"
    )

    return MILPDataset(
        nodes=nodes, services=services,
        r_req=r_req, r_mem=r_mem, acc=acc, x_prev=x_prev,
        theta_max=theta_max, v_storm_max=v_storm_max,
        w_c=w_c, w_d=w_d, w_a=w_a,
        background_cpu=background_cpu,
        background_mem=background_mem,
    )


# ── Standalone test ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = build_dataset_from_cluster(last_placement={})
    print(ds.summary())
    print("\nNodes (CPU + RAM + Dual Energy):")
    for n in ds.nodes:
        print(f"  {n.node_id}: CPU={n.cap_cpu}c  RAM={n.cap_mem_gb:.2f}GB "
              f" E_cpu={n.energy_cost} W/core  E_mem={n.e_mem_unit} W/GB")
    print("\nServices:")
    for s in ds.services:
        print(f"  {s.service_id} ({s.service_type}): variants={s.valid_variants}")
    print("\nResource requirements (detection):")
    for var in ["yolo26-nano", "yolo26-small", "yolo26-medium"]:
        print(f"  detection/{var}: "
              f"CPU={ds.r_req[('m3', var)]}c  "
              f"RAM={ds.r_mem[('m3', var)]:.4f}GB  "
              f"Q={ds.acc[('m3', var)]}")
