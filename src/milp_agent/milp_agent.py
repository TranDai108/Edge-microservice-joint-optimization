"""Long-running MILP agent with Kubernetes scheduler-extender endpoints.

HA additions:
  - LeaderElector gates all milp:placement writes; standby pods run solve
    loop but discard results until elected.
  - Heartbeat written each cycle to milp:heartbeat (TTL = CONTROL_INTERVAL*3).
  - Prometheus metrics exported on port 8080/metrics via prometheus_client.
"""


from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import redis as redis_lib
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIR = PROJECT_ROOT / "solver"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from metrics_collector import build_dataset_from_cluster, get_e2e_latency_ms
from milp_model import solve_placement
try:
    from milp_agent.redis_state import (
        get_mode,
        get_client,
        read_confirmed_placement,
        read_drl_placement,
        read_placement,
        read_weights,
        write_expert_trajectory,
        write_confirmed_placement,
        write_heartbeat,
        write_placement,
        write_weights,
    )
except ImportError:
    from redis_state import (
        get_mode,
        get_client,
        read_confirmed_placement,
        read_drl_placement,
        read_placement,
        read_weights,
        write_expert_trajectory,
        write_confirmed_placement,
        write_heartbeat,
        write_placement,
        write_weights,
    )
from variant_catalog import DEFAULT_DETECTION_VARIANT
import k8s_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("milp-agent")

CONTROL_INTERVAL = int(os.getenv("CONTROL_INTERVAL", "30"))
NODE_WATCHDOG_INTERVAL = int(os.getenv("NODE_WATCHDOG_INTERVAL", "10"))
from config import MILP_W_C, MILP_W_D, MILP_W_A
W_C = MILP_W_C
W_D = MILP_W_D
W_A = MILP_W_A
THETA_MAX = float(os.getenv("THETA_MAX", "1.3"))
REDIS_HOST = os.getenv("REDIS_HOST", "redis.default.svc.cluster.local")
HYBRID_MAX_MILP_AGE_S = float(os.getenv("HYBRID_MAX_MILP_AGE_S", str(max(60, CONTROL_INTERVAL * 2))))
HYBRID_MAX_DRL_AGE_S = float(os.getenv("HYBRID_MAX_DRL_AGE_S", "20"))
HYBRID_MAX_REF_SKEW_S = float(os.getenv("HYBRID_MAX_REF_SKEW_S", str(max(35, CONTROL_INTERVAL + 5))))
HYBRID_REQUIRE_SAME_REF = os.getenv("HYBRID_REQUIRE_SAME_REF", "true").lower() in {
    "1", "true", "yes", "on"
}

# ── Prometheus metrics ────────────────────────────────────────────────────────
try:
    from prometheus_client import Gauge, Counter, make_asgi_app as _prom_asgi
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

if _PROM_AVAILABLE:
    try:
        _milp_leader_gauge = Gauge(
            "milp_leader_status", "1 = current MILP leader pod", ["pod"]
        )
    except ValueError:
        from prometheus_client import REGISTRY as _PROM_REGISTRY
        _milp_leader_gauge = _PROM_REGISTRY._names_to_collectors.get("milp_leader_status")
    try:
        _placement_epoch_counter = Counter(
            "milp_placement_epoch", "Monotonically increasing placement epoch"
        )
    except ValueError:
        from prometheus_client import REGISTRY as _PROM_REGISTRY
        _placement_epoch_counter = _PROM_REGISTRY._names_to_collectors.get("milp_placement_epoch")
else:
    class _Noop:  # minimal no-op shim
        def labels(self, **_): return self
        def set(self, _): pass
        def inc(self): pass
    _milp_leader_gauge = _Noop()
    _placement_epoch_counter = _Noop()

_POD_NAME = os.getenv("POD_NAME", "unknown")

MILP_ID_TO_DEPLOY = {
    "m0": "api-gateway",
    "m1": "ingest",
    "m2": "preprocess",
    "m3": "detection",
    "m4": "gen-ai",
    "m5": "postprocess",
}
SERVICE_ORDER = ["m0", "m1", "m2", "m3", "m4", "m5"]

# ── Module-level state ────────────────────────────────────────────────────────
_elector = None  # LeaderElector instance, set at startup


class ExtenderArgs(BaseModel):
    Pod: dict
    Nodes: Optional[dict] = None
    NodeNames: Optional[list[str]] = None


class HostPriority(BaseModel):
    Host: str
    Score: int


app = FastAPI(title="MILP Scheduler Agent")

# Mount Prometheus /metrics endpoint if available
if _PROM_AVAILABLE:
    try:
        app.mount("/metrics", _prom_asgi())
    except Exception:
        pass

rdb: Optional[redis_lib.Redis] = None

_force_resolve = threading.Event()   # set by watchdog to skip sleep and re-solve immediately
_healthy_nodes: set[str] = set()     # last-known healthy node set for change detection
_all_known_nodes: set[str] = set()   # accumulates every worker ever seen — used to track persistent down nodes


@app.on_event("startup")
async def startup() -> None:
    global rdb, _elector
    rdb = get_client(REDIS_HOST)
    log.info("Redis connected: %s", REDIS_HOST)
    write_weights(rdb, W_C, W_D, W_A)

    # ── Leader election ───────────────────────────────────────────────────────
    try:
        from ha.leader_election import LeaderElector  # type: ignore[import]
        _elector = LeaderElector(lease_name="milp-leader")
        _elector.start()
        log.info("LeaderElector started: milp-leader (pod=%s)", _POD_NAME)
    except Exception as exc:
        log.warning("Leader election unavailable (%s) — single-pod mode", exc)
        _elector = None

    threading.Thread(target=_solve_loop, daemon=True).start()
    threading.Thread(target=_pod_watcher_loop, daemon=True).start()
    threading.Thread(target=_node_health_watchdog, daemon=True).start()
    log.info("Solve loop, pod watcher, and node health watchdog started")


@app.post("/filter")
def filter_nodes(args: ExtenderArgs) -> dict:
    """Return all candidates unchanged; preference is expressed in /prioritize."""
    node_names = _extract_node_names(args)
    return {
        "Nodes": {"items": [{"metadata": {"name": n}} for n in node_names]},
        "FailedNodes": {},
        "Error": "",
    }


@app.post("/prioritize")
def prioritize_nodes(args: ExtenderArgs) -> list[HostPriority]:
    """Return node scores from latest MILP output, or equal scores on cache miss."""
    node_names = _extract_node_names(args)
    if not rdb:
        return [HostPriority(Host=n, Score=50) for n in node_names]

    mode = get_mode(rdb)
    if mode in ("milp", "shadow"):
        placement = read_placement(rdb)
    elif mode == "hybrid":
        milp_p = read_placement(rdb)
        drl_p = read_drl_placement(rdb)
        placement, reason = _choose_hybrid_payload(milp_p, drl_p)
        log.debug("hybrid choose source=%s", reason)
    elif mode == "drl":
        placement = read_drl_placement(rdb)
        if not placement:
            placement = read_placement(rdb)  # Twin-rejected or TTL expired — MILP retains control
            if placement:
                log.info("drl:placement absent; falling back to milp:placement for node scoring")
    else:
        placement = read_placement(rdb)

    if not placement:
        log.warning("No placement in Redis for mode=%s; returning equal scores", mode)
        return [HostPriority(Host=n, Score=50) for n in node_names]

    scores = placement.get("node_scores", {})
    out: list[HostPriority] = []
    for node_name in node_names:
        score = max(0, min(100, int(scores.get(node_name, 50))))
        out.append(HostPriority(Host=node_name, Score=score))
    return out


@app.get("/health")
def health() -> dict:
    ts = rdb.get("milp:solve_timestamp") if rdb else None
    is_leader = _elector.is_leader() if _elector else True
    return {"status": "ok", "last_solve": ts or "never", "leader": is_leader}


@app.get("/leader")
def get_leader_status() -> dict:
    """Return leader election status for this pod."""
    if _elector is None:
        return {"leader": True, "mode": "single-pod", "epoch": 0, "pod": _POD_NAME}
    return {
        "leader": _elector.is_leader(),
        "epoch": _elector.current_epoch(),
        "pod": _POD_NAME,
    }


def _extract_node_names(args: ExtenderArgs) -> list[str]:
    node_names = args.NodeNames or []
    if args.Nodes and isinstance(args.Nodes, dict):
        node_names = [
            n.get("metadata", {}).get("name", "")
            for n in args.Nodes.get("items", [])
            if n.get("metadata", {}).get("name")
        ]
    return node_names


def _parse_iso_utc(ts: object) -> Optional[datetime]:
    if not isinstance(ts, str) or not ts.strip():
        return None
    s = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_seconds(ts: object, now_utc: datetime) -> Optional[float]:
    dt = _parse_iso_utc(ts)
    if dt is None:
        return None
    return max(0.0, (now_utc - dt).total_seconds())


def _choose_hybrid_payload(milp_p: dict | None, drl_p: dict | None) -> tuple[dict | None, str]:
    """Choose payload for hybrid mode with fairness + anti-drift checks.

    DRL is considered only when:
      1) DRL and MILP payloads are fresh enough.
      2) DRL explicitly references MILP snapshot used for its evaluation.
      3) (optional strict) drl.milp_ref_timestamp == current milp.timestamp.
    """
    if not milp_p:
        return drl_p, "drl_only"
    if not drl_p:
        return milp_p, "milp_only"

    now_utc = datetime.now(timezone.utc)
    milp_age = _age_seconds(milp_p.get("timestamp"), now_utc)
    drl_age = _age_seconds(drl_p.get("timestamp"), now_utc)
    milp_ts = milp_p.get("timestamp")
    drl_ref_ts = drl_p.get("milp_ref_timestamp")

    if milp_age is None or milp_age > HYBRID_MAX_MILP_AGE_S:
        log.info("hybrid: MILP payload stale/invalid timestamp age=%s; keep MILP for safety", milp_age)
        return milp_p, "milp_stale_guard"
    if drl_age is None or drl_age > HYBRID_MAX_DRL_AGE_S:
        log.info("hybrid: DRL payload stale/invalid timestamp age=%s; keep MILP", drl_age)
        return milp_p, "drl_stale_guard"
    if not isinstance(drl_ref_ts, str):
        log.info("hybrid: DRL payload missing milp_ref_timestamp; keep MILP")
        return milp_p, "drl_missing_ref"

    ref_dt = _parse_iso_utc(drl_ref_ts)
    milp_dt = _parse_iso_utc(milp_ts)
    if ref_dt is None or milp_dt is None:
        log.info("hybrid: invalid ref timestamp (drl_ref=%s milp_ts=%s); keep MILP", drl_ref_ts, milp_ts)
        return milp_p, "drl_invalid_ref"

    ref_skew = abs((milp_dt - ref_dt).total_seconds())
    if HYBRID_REQUIRE_SAME_REF and drl_ref_ts != milp_ts:
        log.info("hybrid: ref mismatch drl_ref=%s current_milp=%s; keep MILP", drl_ref_ts, milp_ts)
        return milp_p, "ref_mismatch"
    if ref_skew > HYBRID_MAX_REF_SKEW_S:
        log.info("hybrid: ref skew too high (%.2fs > %.2fs); keep MILP", ref_skew, HYBRID_MAX_REF_SKEW_S)
        return milp_p, "ref_skew_guard"

    drl_j = drl_p.get("objective")
    milp_j = milp_p.get("objective")
    if isinstance(drl_j, (float, int)) and isinstance(milp_j, (float, int)) and float(drl_j) < float(milp_j):
        log.info(
            "hybrid: DRL wins on aligned snapshot (J=%.4f < MILP J=%.4f, ref=%s)",
            float(drl_j), float(milp_j), drl_ref_ts,
        )
        return drl_p, "drl_win_aligned"

    return milp_p, "milp_win_aligned"


def _build_state_vector(ds, result) -> list[float]:
    """Build fixed 44-dim state vector from live dataset + latest solution."""
    nodes = list(ds.nodes)
    node_ids = [n.node_id for n in nodes]
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # 1) Node CPU utilization ratio (4)
    cpu_util = [0.0, 0.0, 0.0, 0.0]
    for node in nodes[:4]:
        idx = node_index[node.node_id]
        cap = float(node.cap_cpu) if float(node.cap_cpu) > 0 else 1.0
        used = float(result.resource_usage.get(node.node_id, 0.0))
        cpu_util[idx] = max(0.0, min(1.0, used / cap))

    # 2) Node memory usage in GB (4)
    mem_gb = [0.0, 0.0, 0.0, 0.0]
    for node in nodes[:4]:
        idx = node_index[node.node_id]
        mem_gb[idx] = float(result.mem_usage.get(node.node_id, 0.0))

    # 3) Node CPU energy unit (4)
    e_cpu_unit = [0.0, 0.0, 0.0, 0.0]
    for node in nodes[:4]:
        idx = node_index[node.node_id]
        e_cpu_unit[idx] = float(node.energy_cost)

    # 4) Detection variant one-hot (3)
    det_variants = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
    det_onehot = [0.0, 0.0, 0.0]
    det_variant = result.placement.get("m3", ("yolo26-nano", "n0"))[0]
    if det_variant in det_variants:
        det_onehot[det_variants.index(det_variant)] = 1.0

    # 5) Confirmed placement one-hot: 6 services x 4 nodes (24)
    placement_onehot = [0.0] * 24
    confirmed = read_confirmed_placement(rdb) if rdb else {}

    for svc_idx, svc_id in enumerate(SERVICE_ORDER):
        deploy = MILP_ID_TO_DEPLOY.get(svc_id, svc_id)
        slot = svc_idx * 4
        node_id = None
        if deploy in confirmed:
            host = confirmed[deploy].get("node", "")
            # build_dataset_from_cluster uses sorted worker hostnames => n{i}
            for i, node in enumerate(nodes[:4]):
                if host == getattr(node, "hostname", None):
                    node_id = node.node_id
                    break
            if node_id is None and host.startswith("edge-nodes-"):
                try:
                    host_idx = int(host.rsplit("-", 1)[-1]) - 1
                    if 0 <= host_idx < min(4, len(nodes)):
                        node_id = nodes[host_idx].node_id
                except ValueError:
                    node_id = None
        if node_id is None:
            node_id = result.placement.get(svc_id, ("standard", "n0"))[1]

        idx = node_index.get(node_id, 0)
        if 0 <= idx < 4:
            placement_onehot[slot + idx] = 1.0

    # 6) e2e latency (1)
    raw_e2e = float(get_e2e_latency_ms())
    if math.isfinite(raw_e2e):
        e2e_latency = [raw_e2e]
    else:
        # Keep trajectory numerically stable even when upstream metric is missing.
        e2e_latency = [0.0]
        log.warning("Non-finite e2e latency detected; fallback to 0.0 in trajectory state")

    # 7) migrations in last cycle (1)
    migrations = [float(sum(1 for m in result.migration_types.values() if m != "Stayed"))]

    # 8) Objective weights (3) keep vector fixed at 44 dimensions
    weights = [float(ds.w_c), float(ds.w_d), float(ds.w_a)]

    vec = cpu_util + mem_gb + e_cpu_unit + det_onehot + placement_onehot + e2e_latency + migrations + weights
    if len(vec) < 44:
        vec.extend([0.0] * (44 - len(vec)))
    vec = vec[:44]
    # Final guard rail against NaN/Inf leakage from any upstream feature.
    if not all(math.isfinite(float(v)) for v in vec):
        log.warning("Non-finite value in state vector; replacing invalid entries with 0.0")
        vec = [float(v) if math.isfinite(float(v)) else 0.0 for v in vec]
    return vec


def _build_action_vector(ds, result) -> list[int]:
    """Encode MILP decision into fixed per-service action vector for DRL BC."""
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


def _solve_loop() -> None:
    global W_C, W_D, W_A
    log.info(
        "Solve loop starting: interval=%ss w_c=%s w_d=%s w_a=%s",
        CONTROL_INTERVAL,
        W_C,
        W_D,
        W_A,
    )
    while True:
        t0 = time.perf_counter()
        try:
            if rdb:
                live = read_weights(rdb)
                if live:
                    new_c = float(live.get("w_c", W_C))
                    new_d = float(live.get("w_d", W_D))
                    new_a = float(live.get("w_a", W_A))
                    if (new_c, new_d, new_a) != (W_C, W_D, W_A):
                        log.info(
                            "Weights updated from Redis: w_c=%.4f w_d=%.4f w_a=%.4f",
                            new_c, new_d, new_a,
                        )
                        W_C, W_D, W_A = new_c, new_d, new_a
            last_placement = read_confirmed_placement(rdb) if rdb else {}
            # Emergency evacuation: services on a now-unavailable node have x_prev=0
            # for all remaining nodes, forcing v[m]=1 for each.  If more services are
            # displaced than v_storm_max, the MILP is infeasible.  Raise the limit to
            # cover all displaced services so evacuation always produces a solution.
            available_nodes = set(k8s_client.get_all_worker_nodes())
            displaced = sum(
                1 for info in last_placement.values()
                if info.get("node") and info["node"] not in available_nodes
            )
            base_storm = 5 if not last_placement else 2
            v_storm_max = max(base_storm, displaced)
            if displaced:
                log.warning(
                    "Emergency evacuation: %d service(s) displaced from unavailable "
                    "node(s) — raising v_storm_max to %d",
                    displaced, v_storm_max,
                )
            ds = build_dataset_from_cluster(
                last_placement=last_placement,
                w_c=W_C,
                w_d=W_D,
                w_a=W_A,
                theta_max=THETA_MAX,
                v_storm_max=v_storm_max,
            )
            result = solve_placement(ds, verbose=False)
            if result and rdb:
                # ── Leader gate: only write if this pod is the elected leader ─────
                is_leader = _elector.is_leader() if _elector else True
                _milp_leader_gauge.labels(pod=_POD_NAME).set(1 if is_leader else 0)

                if is_leader:
                    log.info(
                        "Status=%s J=%.4f t=%.2fs",
                        result.status,
                        result.objective_value,
                        result.solve_time,
                    )
                    _ds_nodes_sorted = sorted(ds.nodes, key=lambda n: n.node_id)
                    _e_cpu_unit = [
                        float(_ds_nodes_sorted[i].energy_cost) if i < len(_ds_nodes_sorted) else 0.0
                        for i in range(4)
                    ]
                    write_placement(rdb, result, e_cpu_unit=_e_cpu_unit, elector=_elector)
                    _placement_epoch_counter.inc()
                    state_vec = _build_state_vector(ds, result)
                    action_vec = _build_action_vector(ds, result)
                    reward = -float(result.objective_value)
                    write_expert_trajectory(rdb, state_vec, action_vec, reward, state_vec)
                    log.info("expert_traj pushed len=%d", len(state_vec))
                else:
                    log.debug("Standby: solve OK but not leader — discarding result")

                # Heartbeat written by every pod (not just leader) so monitor can
                # distinguish "no pods running" from "standby not writing".
                write_heartbeat(rdb, "milp", ttl=CONTROL_INTERVAL * 3)

            elif not result:
                log.error("Solver returned no feasible solution")
        except Exception as exc:
            log.error("Solve loop error: %s", exc, exc_info=True)

        elapsed = time.perf_counter() - t0
        wait_s = max(0.0, CONTROL_INTERVAL - elapsed)
        triggered = _force_resolve.wait(timeout=wait_s)
        _force_resolve.clear()
        if triggered:
            log.info("Solve loop woken early by node health watchdog")


def _node_health_watchdog() -> None:
    """
    Poll Kubernetes node conditions every NODE_WATCHDOG_INTERVAL seconds.

    When a previously-healthy node transitions to NotReady, the watchdog:
      1. Logs a warning with the affected node name(s).
      2. Sets _force_resolve so _solve_loop() skips its sleep and re-solves
         immediately, allowing the MILP to replan without the failed node.
      3. Writes milp:node_down (TTL = 3× poll interval) to Redis so the DRL
         agent and edge_controller can react to the event.

    The key is refreshed every poll cycle as long as any node remains down
    (not just on transition) so it never expires while the outage persists.
    Recovery (node comes back) also triggers an immediate re-solve and clears
    the milp:node_down key.
    """
    global _healthy_nodes, _all_known_nodes
    log.info("Node health watchdog started (poll_interval=%ss)", NODE_WATCHDOG_INTERVAL)
    # Seed with ALL worker nodes (including NotReady) so nodes that are already
    # down at pod startup are included in currently_down from the first poll.
    _all_known_nodes.update(k8s_client.get_all_worker_nodes_any_status())
    while True:
        try:
            current_healthy = set(k8s_client.get_all_worker_nodes())
            _all_known_nodes.update(current_healthy)
            currently_down = _all_known_nodes - current_healthy

            if _healthy_nodes:
                newly_down = _healthy_nodes - current_healthy
                if newly_down:
                    log.warning(
                        "Node(s) became unavailable: %s — triggering immediate re-solve",
                        ", ".join(sorted(newly_down)),
                    )
                    _force_resolve.set()
                newly_up = current_healthy - _healthy_nodes
                if newly_up:
                    log.info(
                        "Node(s) recovered: %s — triggering re-solve",
                        ", ".join(sorted(newly_up)),
                    )
                    _force_resolve.set()

            # Always maintain milp:node_down to reflect current reality.
            # Refresh TTL every poll so the key never expires during a
            # persistent outage, and delete it the moment all nodes recover.
            if rdb:
                if currently_down:
                    rdb.setex(
                        "milp:node_down",
                        NODE_WATCHDOG_INTERVAL * 3,
                        ",".join(sorted(currently_down)),
                    )
                else:
                    rdb.delete("milp:node_down")

            _healthy_nodes = current_healthy
        except Exception as exc:
            log.warning("Node health watchdog error: %s", exc)
        time.sleep(NODE_WATCHDOG_INTERVAL)


def _pod_watcher_loop() -> None:
    tracked = set(MILP_ID_TO_DEPLOY.values())
    while True:
        try:
            for deploy_name in tracked:
                pod_node = k8s_client.get_pod_node(deploy_name)
                if not pod_node or not rdb:
                    continue
                variant = DEFAULT_DETECTION_VARIANT if deploy_name == "detection" else "standard"
                for env_item in k8s_client.get_pod_env(deploy_name):
                    if env_item.get("name") == "VARIANT_ID":
                        variant = env_item.get("value", variant)
                        break
                write_confirmed_placement(rdb, deploy_name, pod_node, variant)
        except Exception as exc:
            log.warning("Pod watcher error: %s", exc)
        time.sleep(15)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
