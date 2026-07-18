
import subprocess
import pandas as pd
import time, os, json, logging, csv
import math
from pathlib import Path
from datetime import datetime, timezone
import sys

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
import k8s_client
from variant_catalog import (
    DEFAULT_DETECTION_VARIANT,
    DETECTION_ROLLOUT_TIMEOUT_S,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("edge-controller")

RESULTS_DIR      = Path(__file__).parent.parent.parent / "results"
EXPERIMENT_LOG   = RESULTS_DIR / "experiment_log.csv"
DECISION_LOG     = RESULTS_DIR / "controller_decisions.jsonl"
CONTROL_INTERVAL     = 30       # seconds between MILP solves
SLA_LATENCY_MS       = float(os.getenv("SLA_LATENCY_MS", "1500.0"))
SOLVER_TIMEOUT_S     = 120      # hard timeout to keep control loop responsive
MAX_FALLBACK_RATIO   = 0.6      # block solve when telemetry quality is too low
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hybrid / Redis config ─────────────────────────────────────────────────────
REDIS_HOST              = os.getenv("REDIS_HOST", "redis.default.svc.cluster.local")
HYBRID_MAX_MILP_AGE_S  = float(os.getenv("HYBRID_MAX_MILP_AGE_S", "120"))
HYBRID_MAX_DRL_AGE_S   = float(os.getenv("HYBRID_MAX_DRL_AGE_S", "20"))
HYBRID_REQUIRE_SAME_REF = os.getenv("HYBRID_REQUIRE_SAME_REF", "true").lower() in {
    "1", "true", "yes", "on"
}

from config import MILP_W_C, MILP_W_D, MILP_W_A
# ── MILP Solver Objective Weights ──
W_COST_ENERGY     = MILP_W_C  # w_c
W_COST_DISRUPTION = MILP_W_D  # w_d
W_GAIN_ACCURACY   = MILP_W_A  # w_a 

# Use the registry IP directly — the hostname 'registry' only resolves on the
# orchestrator node (via /etc/hosts) but NOT on the edge nodes, which causes
# ImagePullBackOff whenever a pod is migrated to a node that lacks the cache.
REGISTRY = "192.168.100.3:5000"

# ── Consistent mapping: MILP service ID → K8s deployment name ──
MILP_ID_TO_DEPLOY = {
    "m0": "api-gateway",
    "m1": "ingest",
    "m2": "preprocess",
    "m3": "detection",      # ← variant-aware AI service
    "m4": "gen-ai",
    "m5": "postprocess",
}

# ── K8s node hostname lookup ──
NODE_MAP = {
    "n0": "edge-nodes-1",
    "n1": "edge-nodes-2",
    "n2": "edge-nodes-3",
    "n3": "edge-nodes-4",
}

last_placement: dict = {}   # {"detection": {"node": "edge-nodes-1", "variant": "yolo26-nano"}, ...}
cycle_count = 0
_rdb = None                 # Redis client, initialised in __main__


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _init_redis():
    """Create a Redis client. Returns None if redis-py is unavailable."""
    if not _REDIS_AVAILABLE:
        log.warning("redis-py not installed — hybrid mode disabled")
        return None
    try:
        r = redis_lib.Redis(host=REDIS_HOST, decode_responses=True, socket_timeout=2)
        r.ping()
        log.info("Redis connected: %s", REDIS_HOST)
        return r
    except Exception as exc:
        log.warning("Redis unavailable (%s) — hybrid mode disabled", exc)
        return None


def _get_system_mode() -> str:
    """Read system:mode from Redis. Defaults to 'milp' when Redis is absent."""
    if _rdb is None:
        return "milp"
    try:
        mode = _rdb.get("system:mode")
        return mode if mode else "milp"
    except Exception:
        return "milp"


def _age_seconds(ts_str: str | None) -> float | None:
    """Return age in seconds for an ISO-8601 timestamp string, or None on error."""
    if not ts_str:
        return None
    try:
        s = ts_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return None


def _read_redis_placement(key: str) -> dict | None:
    """Read and JSON-decode a placement payload from Redis."""
    if _rdb is None:
        return None
    try:
        raw = _rdb.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _choose_hybrid_source(milp_redis: dict | None, drl_redis: dict | None) -> tuple[str, str]:
    """Mirror the logic in milp_agent._choose_hybrid_payload.

    Returns (source, reason) where source is 'milp' or 'drl'.
    Staleness and ref-timestamp guards match milp_agent to keep both
    decision points consistent.
    """
    if not milp_redis:
        return "drl", "milp_absent"
    if not drl_redis:
        return "milp", "drl_absent"

    milp_age = _age_seconds(milp_redis.get("timestamp"))
    drl_age  = _age_seconds(drl_redis.get("timestamp"))
    drl_ref  = drl_redis.get("milp_ref_timestamp")
    milp_ts  = milp_redis.get("timestamp")

    if milp_age is None or milp_age > HYBRID_MAX_MILP_AGE_S:
        return "milp", f"milp_stale_guard(age={milp_age})"
    if drl_age is None or drl_age > HYBRID_MAX_DRL_AGE_S:
        return "milp", f"drl_stale_guard(age={drl_age})"
    if not isinstance(drl_ref, str):
        return "milp", "drl_missing_ref"
    if HYBRID_REQUIRE_SAME_REF and drl_ref != milp_ts:
        return "milp", f"ref_mismatch(drl_ref={drl_ref} milp_ts={milp_ts})"

    drl_j  = drl_redis.get("objective")
    milp_j = milp_redis.get("objective")
    if isinstance(drl_j, (float, int)) and isinstance(milp_j, (float, int)) \
            and float(drl_j) < float(milp_j):
        return "drl", f"drl_win_aligned(J={drl_j:.4f}<{milp_j:.4f})"

    return "milp", f"milp_win_aligned(J={milp_j:.4f}<={drl_j})"


def apply_drl_placement(drl_payload: dict) -> int:
    """Apply DRL placement dict to Kubernetes. Returns migration count.

    Mirrors apply_placement() but reads from the drl:placement dict
    (keyed by deploy name → {node, variant}) instead of a MILP CSV.
    Only services whose node or variant differ from last_placement are
    restarted, so a stable DRL decision causes zero migrations.
    """
    global last_placement
    placement = drl_payload.get("placement", {})
    migrations = 0

    for deploy_name, info in placement.items():
        target_node = info.get("node", "")
        variant     = info.get("variant", "standard")

        prev = last_placement.get(deploy_name, {})
        if prev.get("node") == target_node and prev.get("variant") == variant:
            log.debug("[%s] DRL: Stayed — no change needed", deploy_name)
            last_placement[deploy_name] = {"node": target_node, "variant": variant}
            continue

        migrations += 1
        log.info("[%s] DRL placement → node=%s variant=%s", deploy_name, target_node, variant)

        if deploy_name == "detection":
            image = f"{REGISTRY}/detection:{variant}"
        else:
            image = f"{REGISTRY}/{deploy_name}:latest"

        k8s_client.patch_deployment_node_selector(deploy_name, target_node)
        k8s_client.set_deployment_image(deploy_name, deploy_name, image)
        if deploy_name == "detection":
            k8s_client.set_deployment_env(deploy_name, "VARIANT_ID", variant)
        k8s_client.rollout_restart_deployment(deploy_name)

        timeout = DETECTION_ROLLOUT_TIMEOUT_S.get(variant, 45)
        k8s_client.wait_rollout_complete(deploy_name, timeout_s=timeout)

        last_placement[deploy_name] = {"node": target_node, "variant": variant}
        log.info("[%s] DRL Done.", deploy_name)

    return migrations


def run(cmd: str, check: bool = True) -> str:
    """Subprocess shim — used only to launch the Python solver as a child process."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        log.error(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def init_cluster_state() -> dict:
    """
    Fetch the actual running state of pods from Kubernetes using the API.
    Replaces kubectl get pods … -o jsonpath shell calls.
    """
    log.info("Synchronising initial cluster state from Kubernetes API...")
    state = {}
    for milp_id, deploy_name in MILP_ID_TO_DEPLOY.items():
        node    = "edge-nodes-1"   # safe fallback
        variant = "standard"

        # Fetch real node via Kubernetes API
        pod_node = k8s_client.get_pod_node(deploy_name)
        if pod_node:
            node = pod_node

        # Dynamically fetch variant if the pod defines it via environment variables
        env_list = k8s_client.get_pod_env(deploy_name)
        for item in env_list:
            if item.get("name") == "VARIANT_ID":
                variant = item.get("value", variant)
                break

        state[deploy_name] = {"node": node, "variant": variant}
        log.info(f"  Found [{deploy_name}] on {node} (variant: {variant})")
    return state


def apply_placement(df: pd.DataFrame):
    """
    Read the MILP placement CSV (m0…m4 service IDs) and apply to K8s
    using the Kubernetes Python API.
    Updates last_placement using deploy names as keys for metrics_collector.
    """
    global last_placement
    for _, row in df.iterrows():
        milp_id     = row["service_id"]            # e.g. "m3"
        variant     = row.get("variant_id", DEFAULT_DETECTION_VARIANT)
        node_key    = row["node_id"]               # e.g. "n1"
        deploy_name = MILP_ID_TO_DEPLOY.get(milp_id, milp_id)
        target_node = NODE_MAP.get(node_key, node_key)
        mig_type    = row.get("migration_type", "Stayed")

        # Primary gate: trust the MILP's own migration decision.
        # This avoids a false-positive on cycle 1 when last_placement={}
        # causes every service to look "changed" vs None, which would trigger
        # a full rollout on all 5 services (up to 4+ minutes wasted).
        if mig_type == "Stayed":
            log.debug(f"[{deploy_name}] Stayed — no API action needed.")
            # Still update in-memory state so next cycle is accurate
            last_placement[deploy_name] = {"node": target_node, "variant": variant}
            continue

        log.info(f"[{deploy_name}] {mig_type} → node={target_node} variant={variant}")

        # Build image tag (only detection has variant-specific images)
        if deploy_name == "detection":
            image = f"{REGISTRY}/detection:{variant}"
        else:
            image = f"{REGISTRY}/{deploy_name}:latest"

        # Apply via Kubernetes API
        k8s_client.patch_deployment_node_selector(deploy_name, target_node)
        k8s_client.set_deployment_image(deploy_name, deploy_name, image)
        if deploy_name == "detection":
            k8s_client.set_deployment_env(deploy_name, "VARIANT_ID", variant)
        k8s_client.rollout_restart_deployment(deploy_name)

        timeout = DETECTION_ROLLOUT_TIMEOUT_S.get(variant, 45)
        k8s_client.wait_rollout_complete(deploy_name, timeout_s=timeout)

        # Update state — keyed by deploy_name so metrics_collector can read it
        last_placement[deploy_name] = {"node": target_node, "variant": variant}
        log.info(f"[{deploy_name}] Done.")


def get_latest_csv() -> pd.DataFrame | None:
    csvs = list(RESULTS_DIR.glob("placement_*.csv"))
    if not csvs:
        log.warning("No placement CSV found in results/")
        return None
    latest = max(csvs, key=os.path.getmtime)
    log.info(f"Reading placement from {latest.name}")
    return pd.read_csv(latest)


def get_energy_snapshot() -> dict:
    """Read per-node energy cost from the latest metrics collector run."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from solver.metrics_collector import get_energy_cost_per_core, NODE_POWER_PROFILES
        return {
            hostname: get_energy_cost_per_core(hostname)
            for hostname in NODE_POWER_PROFILES
        }
    except Exception as e:
        log.warning(f"Could not read energy snapshot: {e}")
        return {}


def init_experiment_log():
    if not EXPERIMENT_LOG.exists():
        with open(EXPERIMENT_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "timestamp", "cycle", "e2e_latency_ms", "detection_accuracy",
                "active_variant", "sla_violated", "migrations_this_cycle",
                "node_cpu_avg", "energy_cost_n0", "energy_cost_n1",
                "energy_cost_n2", "solver_result",
            ]).writeheader()


def _safe_float(value, default=0.0):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def log_decision_event(cycle: int, solver_result: str, solver_ok: bool,
                       migrations: int, metrics: dict, fallback_ratio: float | None = None):
    if solver_ok:
        reason_code = "placement_applied" if migrations > 0 else "placement_unchanged"
    elif solver_result == "timeout":
        reason_code = "solver_timeout"
    elif solver_result == "quality_gate":
        reason_code = "metrics_quality_low"
    else:
        reason_code = "solver_error"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "cycle": cycle,
        "reason_code": reason_code,
        "solver_result": solver_result,
        "solver_ok": solver_ok,
        "migrations": migrations,
        "fallback_ratio": fallback_ratio,
        "e2e_latency_ms": _safe_float(metrics.get("e2e_latency_ms", 0), default=0.0),
        "detection_accuracy": _safe_float(metrics.get("detection_accuracy", 0), default=0.0),
        "active_variant": metrics.get("active_variant", "unknown"),
    }
    with open(DECISION_LOG, "a") as f:
        f.write(json.dumps(payload) + "\n")


def log_cycle(metrics: dict, migrations: int, solver_ok: bool, energy: dict, solver_result: str = "ok"):
    e2e_latency = _safe_float(metrics.get("e2e_latency_ms", 0), default=0.0)
    detection_accuracy = _safe_float(metrics.get("detection_accuracy", 0), default=0.0)
    sla_violated = e2e_latency > SLA_LATENCY_MS
    node_cpus    = list(metrics.get("node_cpu", {}).values())
    node_cpus_sanitized = [_safe_float(v, default=0.0) for v in node_cpus]
    node_cpu_avg = sum(node_cpus_sanitized) / len(node_cpus_sanitized) if node_cpus_sanitized else 0.0

    # Map hostnames to node IDs for CSV columns
    hostname_to_id = {"edge-nodes-1": "n0", "edge-nodes-2": "n1", "edge-nodes-3": "n2"}

    row = {
        "timestamp":            datetime.now().isoformat(),
        "cycle":                cycle_count,
        "e2e_latency_ms":       round(e2e_latency, 2),
        "detection_accuracy":   round(detection_accuracy, 4),
        "active_variant":       metrics.get("active_variant", "unknown"),
        "sla_violated":         int(sla_violated),
        "migrations_this_cycle": migrations,
        "node_cpu_avg":         round(node_cpu_avg, 4),
        "energy_cost_n0":       energy.get("edge-nodes-1", 0),
        "energy_cost_n1":       energy.get("edge-nodes-2", 0),
        "energy_cost_n2":       energy.get("edge-nodes-3", 0),
        "solver_result":        solver_result if not solver_ok else "ok",
    }
    with open(EXPERIMENT_LOG, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

    if sla_violated:
        log.warning(f"SLA VIOLATED — e2e={row['e2e_latency_ms']}ms > {SLA_LATENCY_MS}ms")


def get_metrics_snapshot() -> dict:
    """Pull a metrics snapshot from the dynamic dataset module."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from solver.metrics_collector import (
            get_e2e_latency_ms, get_detection_accuracy,
            get_active_variant, get_node_cpu_util
        )
        nodes = k8s_client.get_all_worker_nodes() or ["edge-nodes-1", "edge-nodes-2", "edge-nodes-3"]
        return {
            "e2e_latency_ms":     get_e2e_latency_ms(),
            "detection_accuracy": get_detection_accuracy(),
            "active_variant":     get_active_variant(),
            "node_cpu":           {h: get_node_cpu_util(h) for h in nodes},
        }
    except Exception as e:
        log.warning(f"Metrics snapshot failed: {e}")
        return {"e2e_latency_ms": 0, "detection_accuracy": 0,
                "active_variant": "unknown", "node_cpu": {}}


# ── Add these helper functions to metrics_collector.py as well ──
# (get_e2e_latency_ms, get_detection_accuracy, get_active_variant)
# They are simple _prom() wrappers — see below

# ─────────────────────────── Main loop ────────────────────────────────────────

if __name__ == "__main__":
    _rdb = _init_redis()
    last_placement = init_cluster_state()
    init_experiment_log()
    log.info("Edge controller started. Logs → results/experiment_log.csv")
    log.info(f"Active Weights: Energy(w_c)={W_COST_ENERGY} | Disruption(w_d)={W_COST_DISRUPTION} | Accuracy(w_a)={W_GAIN_ACCURACY}")

    while True:
        try:
            cycle_count += 1
            log.info("")
            log.info(f"{'='*20} CYCLE {cycle_count} {'='*20}")

            # 1. Snapshot metrics for logging
            metrics = get_metrics_snapshot()
            energy  = get_energy_snapshot()

            placement_state_path = RESULTS_DIR / "last_placement.json"

            # 2. Write last_placement BEFORE solver runs so x_prev is accurate
            with open(placement_state_path, "w") as f:
                json.dump(last_placement, f, indent=2)

            # 3. Run solver — absolute --output-dir so CSV lands in project results/
            solver_result = "ok"
            fallback_ratio = None
            try:
                solver_proc = subprocess.run(
                    f"python3 run_solver.py --placement {placement_state_path} "
                    f"--output-dir {RESULTS_DIR.resolve()} "
                    f"--w-c {W_COST_ENERGY} --w-d {W_COST_DISRUPTION} --w-a {W_GAIN_ACCURACY} "
                    f"--max-fallback-ratio {MAX_FALLBACK_RATIO}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=os.path.expanduser("~/KLTN_project/src/solver"),
                    timeout=SOLVER_TIMEOUT_S,
                )
                solver_ok = solver_proc.returncode == 0

                if solver_ok:
                    import re
                    status_match = re.search(r"Status: (.*)", solver_proc.stdout)
                    solver_status = status_match.group(1).strip() if status_match else "Completed"
                    log.info(f"MILP Solver finished | Status: {solver_status}")
                    ratio_match = re.search(r"fallback_ratio=([0-9.]+)", solver_proc.stdout)
                    if ratio_match:
                        fallback_ratio = float(ratio_match.group(1))
                else:
                    solver_result = "quality_gate" if solver_proc.returncode == 2 else "error"
                    log.error(f"Solver failed! Error log:\n{solver_proc.stderr}")
            except subprocess.TimeoutExpired:
                solver_ok = False
                solver_result = "timeout"
                log.error(f"Solver timed out after {SOLVER_TIMEOUT_S}s. Keeping last known placement.")

            # 4. Apply placement — respects system:mode (milp / hybrid / drl)
            # In hybrid mode: compare drl:placement J vs milp:placement J from
            # Redis and apply whichever wins, using the same staleness/ref guards
            # as milp_agent._choose_hybrid_payload().
            migrations = 0
            df = get_latest_csv()
            mode = _get_system_mode()

            if mode == "hybrid" and solver_ok:
                milp_redis = _read_redis_placement("milp:placement")
                drl_redis  = _read_redis_placement("drl:placement")
                source, reason = _choose_hybrid_source(milp_redis, drl_redis)
                log.info("[HYBRID] source=%s reason=%s", source, reason)
                if source == "drl" and drl_redis:
                    migrations = apply_drl_placement(drl_redis)
                elif df is not None:
                    apply_placement(df)
                    migrations = int((df["migration_type"] != "Stayed").sum())
            elif mode == "drl" and solver_ok:
                drl_redis = _read_redis_placement("drl:placement")
                if drl_redis:
                    log.info("[DRL] applying drl:placement")
                    migrations = apply_drl_placement(drl_redis)
                elif df is not None:
                    log.info("[DRL] drl:placement absent — falling back to MILP CSV")
                    apply_placement(df)
                    migrations = int((df["migration_type"] != "Stayed").sum())
            elif df is not None and solver_ok:
                apply_placement(df)
                migrations = int((df["migration_type"] != "Stayed").sum())
            elif not solver_ok:
                log.warning(f"Solver {solver_result} — keeping current placement.")

            # 5. Log cycle
            log_cycle(metrics, migrations, solver_ok, energy, solver_result=solver_result)
            log_decision_event(cycle_count, solver_result, solver_ok, migrations, metrics, fallback_ratio)
            log.info(f"Cycle {cycle_count} complete. Total migrations applied: {migrations}")

        except Exception as e:
            log.error(f"Cycle error: {e}", exc_info=True)

        log.info(f"Sleeping {CONTROL_INTERVAL}s...")
        time.sleep(CONTROL_INTERVAL)
