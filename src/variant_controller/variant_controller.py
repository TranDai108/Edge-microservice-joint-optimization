"""Event-driven AI variant controller — HA edition.

Subscribes to Redis events published by the MILP agent and applies
variant changes for AI services.

HA additions (Controller HA and Failover Plan):
  • Leader election via K8s Lease (only one pod applies rollouts at a time).
  • Distributed rollout lock (Redis SET NX) prevents duplicate concurrent
    rollouts when two pods transiently believe they are leader.
  • Idempotent rollout: desired state fingerprint stored in Redis; skip
    apply_variant_change if the cluster already matches desired state.
  • Sentinel-aware Redis client via src.ha.redis_client.make_redis_client.
"""


from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hashlib
import uuid

import redis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import k8s_client
from variant_catalog import (
    DEFAULT_DETECTION_VARIANT,
    DEFAULT_GEN_AI_VARIANT,
    DETECTION_ROLLOUT_TIMEOUT_S,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("variant-controller")

REDIS_HOST = os.getenv("REDIS_HOST", "redis.default.svc.cluster.local")
REGISTRY = os.getenv("REGISTRY", "192.168.100.3:5000")
RECONCILE_INTERVAL_S = int(os.getenv("RECONCILE_INTERVAL_S", "5"))
# Rollout lock TTL — long enough for the slowest rollout; prevent stale locks.
ROLLOUT_LOCK_TTL_S = int(os.getenv("ROLLOUT_LOCK_TTL_S", "120"))
DEFAULT_ROLLOUT_TIMEOUT_S = int(os.getenv("DEFAULT_ROLLOUT_TIMEOUT_S", "120"))
VC_REQUIRE_LEADER = os.getenv("VC_REQUIRE_LEADER", "true").lower() in {"1", "true", "yes", "on"}
HYBRID_MAX_MILP_AGE_S = float(os.getenv("HYBRID_MAX_MILP_AGE_S", "60"))
HYBRID_MAX_DRL_AGE_S = float(os.getenv("HYBRID_MAX_DRL_AGE_S", "20"))
HYBRID_MAX_REF_SKEW_S = float(os.getenv("HYBRID_MAX_REF_SKEW_S", "35"))
HYBRID_REQUIRE_SAME_REF = os.getenv("HYBRID_REQUIRE_SAME_REF", "true").lower() in {
    "1", "true", "yes", "on"
}

AI_SERVICES = {"detection", "gen-ai"}
MANAGED_SERVICES = {
    svc.strip()
    for svc in os.getenv(
        "MANAGED_SERVICES",
        "api-gateway,ingest,preprocess,detection,gen-ai,postprocess",
    ).split(",")
    if svc.strip()
}
DEFAULT_VARIANT_BY_SERVICE = {
    "api-gateway": "standard",
    "ingest": "standard",
    "preprocess": "standard",
    "detection": DEFAULT_DETECTION_VARIANT,
    "gen-ai": DEFAULT_GEN_AI_VARIANT,
    "postprocess": "standard",
}


def _get_mode(rdb: redis.Redis) -> str:
    mode = (rdb.get("system:mode") or "milp").strip().lower()
    if mode in {"milp", "shadow", "hybrid", "drl"}:
        return mode
    return "milp"


def _can_act(elector) -> bool:
    """Gate action execution.

    Default is strict leader-only for standard HA behavior. A fail-open mode
    can be enabled by setting VC_REQUIRE_LEADER=false for emergency recovery.
    """
    if elector is None:
        return True
    if elector.is_leader():
        return True
    return not VC_REQUIRE_LEADER


def _read_payload(rdb: redis.Redis, key: str) -> dict | None:
    raw = rdb.get(key)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Invalid %s payload: %s", key, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _as_float(v: object) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _parse_iso_utc(ts: object) -> datetime | None:
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


def _age_seconds(ts: object, now_utc: datetime) -> float | None:
    dt = _parse_iso_utc(ts)
    if dt is None:
        return None
    return max(0.0, (now_utc - dt).total_seconds())


def _hybrid_can_compare(milp_payload: dict | None, drl_payload: dict | None) -> tuple[bool, str]:
    if not milp_payload or not drl_payload:
        return False, "missing_payload"
    now_utc = datetime.now(timezone.utc)
    milp_age = _age_seconds(milp_payload.get("timestamp"), now_utc)
    drl_age = _age_seconds(drl_payload.get("timestamp"), now_utc)
    if milp_age is None or milp_age > HYBRID_MAX_MILP_AGE_S:
        return False, f"milp_stale:{milp_age}"
    if drl_age is None or drl_age > HYBRID_MAX_DRL_AGE_S:
        return False, f"drl_stale:{drl_age}"

    drl_ref_ts = drl_payload.get("milp_ref_timestamp")
    milp_ts = milp_payload.get("timestamp")
    ref_dt = _parse_iso_utc(drl_ref_ts)
    milp_dt = _parse_iso_utc(milp_ts)
    if ref_dt is None or milp_dt is None:
        return False, "invalid_ref_ts"
    if HYBRID_REQUIRE_SAME_REF and drl_ref_ts != milp_ts:
        return False, "ref_mismatch"
    if abs((milp_dt - ref_dt).total_seconds()) > HYBRID_MAX_REF_SKEW_S:
        return False, "ref_skew"
    return True, "ok"


def _choose_desired_payload(rdb: redis.Redis) -> tuple[dict | None, str, str]:
    """Return (payload, source_key, mode) used for controller reconciliation."""
    mode = _get_mode(rdb)
    milp_payload = _read_payload(rdb, "milp:placement")
    drl_payload = _read_payload(rdb, "drl:placement")

    if mode in {"milp", "shadow"}:
        return milp_payload, "milp:placement", mode

    if mode == "drl":
        if drl_payload:
            return drl_payload, "drl:placement", mode
        return milp_payload, "milp:placement(fallback)", mode

    # hybrid: compare only on aligned snapshots to avoid drift bias.
    comparable, cmp_reason = _hybrid_can_compare(milp_payload, drl_payload)
    if not comparable:
        if drl_payload:
            log.info("hybrid desired-source guard: %s -> MILP base", cmp_reason)
        return milp_payload, f"milp:placement(hybrid_guard:{cmp_reason})", mode

    drl_j = _as_float((drl_payload or {}).get("objective"))
    milp_j = _as_float((milp_payload or {}).get("objective"))
    if drl_payload and (milp_j is None or (drl_j is not None and drl_j < milp_j)):
        return drl_payload, "drl:placement(hybrid_win)", mode
    return milp_payload, "milp:placement(hybrid_base)", mode


def _rollout_lock_key(service: str) -> str:
    return f"variant-controller:lock:{service}"


def _applied_state_key(service: str) -> str:
    return f"variant-controller:applied:{service}"


def _desired_fingerprint(service: str, variant: str, node: str = "") -> str:
    """Stable hash of the desired state to detect idempotent re-triggers."""
    raw = f"{service}|{variant}|{node}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _acquire_rollout_lock(rdb: redis.Redis, service: str) -> tuple[bool, str]:
    """Acquire a distributed Redis lock for one service rollout.

    Returns (acquired, lock_token).  The caller must release the lock by
    deleting the key only if the stored token matches (prevents a slow pod
    from releasing a lock taken by another pod after expiry).
    """
    key = _rollout_lock_key(service)
    token = str(uuid.uuid4())
    # SET NX PX — atomic acquire with expiry in milliseconds
    acquired = rdb.set(key, token, nx=True, px=ROLLOUT_LOCK_TTL_S * 1000)
    return bool(acquired), token


_RELEASE_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


def _release_rollout_lock(rdb: redis.Redis, service: str, token: str) -> None:
    rdb.eval(_RELEASE_LOCK_LUA, 1, _rollout_lock_key(service), token)


def apply_variant_change(service: str, variant: str, rdb: redis.Redis | None = None) -> None:
    """Apply a variant rollout, guarded by a distributed lock for idempotency.

    Steps:
      1. Check if desired state is already applied (fingerprint match) — skip
         if so (idempotent re-trigger protection).
      2. Acquire distributed Redis lock so only one controller instance runs
         the rollout at a time (split-brain fencing at the action level).
      3. Perform the rollout via k8s_client.
      4. Record applied fingerprint in Redis on success.
      5. Always release the lock.
    """
    if service not in AI_SERVICES:
        log.debug("[%s] Non-AI service; skip variant change", service)
        return

    # ── Idempotency check ─────────────────────────────────────────────────────
    fingerprint = _desired_fingerprint(service, variant)
    if rdb is not None:
        applied = rdb.get(_applied_state_key(service))
        if applied == fingerprint:
            log.info(
                "[%s] Desired variant=%s already applied (fingerprint=%s); skipping rollout",
                service, variant, fingerprint,
            )
            return

    # ── Distributed lock ──────────────────────────────────────────────────────
    lock_token: str | None = None
    if rdb is not None:
        acquired, lock_token = _acquire_rollout_lock(rdb, service)
        if not acquired:
            log.warning(
                "[%s] Rollout lock held by another instance; skipping variant=%s",
                service, variant,
            )
            return
        log.debug("[%s] Acquired rollout lock (token=%s)", service, lock_token[:8])

    try:
        timeout = DETECTION_ROLLOUT_TIMEOUT_S.get(variant, 45)
        k8s_client.set_deployment_env(service, "VARIANT_ID", variant, container_name=service)

        if service == "detection":
            image = f"{REGISTRY}/{service}:{variant}"
            log.info("[%s] Applying variant=%s image=%s", service, variant, image)
            k8s_client.set_deployment_image(service, service, image)
        else:
            log.info("[%s] Applying variant=%s via VARIANT_ID env", service, variant)

        k8s_client.rollout_restart_deployment(service)
        ok = k8s_client.wait_rollout_complete(service, timeout_s=timeout)
        if not ok:
            raise RuntimeError(
                f"[{service}] Rollout did not complete for variant={variant}. "
                "Check pod events/logs for image pull or probe failures."
            )

        # ── Record applied state on success ───────────────────────────────────
        if rdb is not None:
            rdb.setex(_applied_state_key(service), ROLLOUT_LOCK_TTL_S * 10, fingerprint)
            log.debug("[%s] Applied state recorded (fingerprint=%s)", service, fingerprint)

    finally:
        if rdb is not None and lock_token is not None:
            _release_rollout_lock(rdb, service, lock_token)
            log.debug("[%s] Released rollout lock", service)


def _read_desired_state(rdb: redis.Redis) -> tuple[dict[str, dict[str, str]], str | None, str, str]:
    """Read desired placement for managed services from mode-selected payload."""
    payload, source_key, mode = _choose_desired_payload(rdb)
    if not payload:
        return {}, None, source_key, mode
    desired: dict[str, dict[str, str]] = {}
    placement = payload.get("placement", {})
    for service in MANAGED_SERVICES:
        data = placement.get(service, {})
        desired[service] = {
            "variant": data.get("variant", DEFAULT_VARIANT_BY_SERVICE[service]),
            "node": data.get("node", ""),
        }
    return desired, payload.get("timestamp"), source_key, mode


def _get_actual_service_state(service: str) -> dict[str, str]:
    """Read current running pod node + VARIANT_ID for one managed service."""
    node = k8s_client.get_pod_node(service) or ""
    variant = DEFAULT_VARIANT_BY_SERVICE.get(service, "standard")
    for env_item in k8s_client.get_pod_env(service):
        if env_item.get("name") == "VARIANT_ID":
            variant = env_item.get("value", variant)
            break
    return {"node": node, "variant": variant}


def _rollout_timeout(service: str, variant: str) -> int:
    if service == "detection":
        return DETECTION_ROLLOUT_TIMEOUT_S.get(variant, DEFAULT_ROLLOUT_TIMEOUT_S)
    return DEFAULT_ROLLOUT_TIMEOUT_S


def _apply_node_change(service: str, target_node: str, variant: str) -> None:
    """Apply nodeSelector migration for any managed service and wait rollout."""
    # Pre-step: force-delete pods stuck on unavailable nodes so the rolling
    # update controller is not blocked waiting for a graceful shutdown that
    # will never arrive from an unreachable kubelet.
    stuck = k8s_client.force_delete_stuck_pods(service)
    if stuck:
        log.info(
            "[%s] Force-deleted %d stuck pod(s) on unavailable node(s) before migration",
            service, stuck,
        )
    log.info("[%s] Applying node migration to %s", service, target_node)
    k8s_client.patch_deployment_node_selector(service, target_node)
    k8s_client.rollout_restart_deployment(service)
    timeout = _rollout_timeout(service, variant)
    ok = k8s_client.wait_rollout_complete(service, timeout_s=timeout)
    if not ok:
        raise RuntimeError(
            f"[{service}] Rollout did not complete during node migration to {target_node}."
        )


def _reconcile_service(service: str, desired: dict[str, str], actual: dict[str, str]) -> tuple[bool, list[str]]:
    """Enforce desired state for one managed service when drift is detected."""
    drift_reasons: list[str] = []
    desired_variant = desired.get("variant", DEFAULT_VARIANT_BY_SERVICE[service])
    desired_node = desired.get("node", "")
    actual_variant = actual.get("variant", DEFAULT_VARIANT_BY_SERVICE[service])
    actual_node = actual.get("node", "")

    variant_drift = desired_variant != actual_variant
    node_drift = bool(desired_node and desired_node != actual_node)

    if service in AI_SERVICES and variant_drift:
        drift_reasons.append("variant")
    if node_drift:
        drift_reasons.append("node")

    if not drift_reasons:
        return False, drift_reasons

    log.warning(
        "[%s] Drift detected (%s): desired=(variant=%s,node=%s) actual=(variant=%s,node=%s)",
        service,
        ",".join(drift_reasons),
        desired_variant,
        desired_node or "?",
        actual_variant,
        actual_node or "?",
    )

    if node_drift:
        _apply_node_change(service, desired_node, desired_variant)

    if service in AI_SERVICES and variant_drift:
        apply_variant_change(service, desired_variant)
    elif service in AI_SERVICES and not node_drift:
        timeout = _rollout_timeout(service, desired_variant)
        ok = k8s_client.wait_rollout_complete(service, timeout_s=timeout)
        if not ok:
            raise RuntimeError(
                f"[{service}] Rollout did not complete during node reconciliation "
                f"(variant={desired_variant})."
            )

    return True, drift_reasons


def reconcile_desired_state(rdb: redis.Redis) -> None:
    """Compare MILP desired state and live cluster state, then self-heal drift."""
    desired_by_service, desired_ts, desired_source, mode = _read_desired_state(rdb)
    if not desired_by_service:
        log.warning("No desired placement payload for reconciliation (mode=%s source=%s)", mode, desired_source)
        return

    drifted_services: list[dict[str, object]] = []
    in_sync = True

    for service, desired in desired_by_service.items():
        actual = _get_actual_service_state(service)
        try:
            reconciled, reasons = _reconcile_service(service, desired, actual)
        except Exception as exc:
            log.error("[%s] Reconcile failed (continuing with other services): %s", service, exc)
            in_sync = False
            continue
        if reconciled:
            in_sync = False
            drifted_services.append(
                {
                    "service": service,
                    "reasons": reasons,
                    "desired": desired,
                    "actual": actual,
                }
            )

    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "desired_timestamp": desired_ts,
        "desired_source": desired_source,
        "mode": mode,
        "in_sync": in_sync,
        "drift_count": len(drifted_services),
        "drifted_services": drifted_services,
    }
    rdb.setex("milp:controller_sync", RECONCILE_INTERVAL_S * 3, json.dumps(status))

    if in_sync:
        log.info("Controller sync OK: mode=%s source=%s matches live state", mode, desired_source)
    else:
        log.warning(
            "Controller sync repaired: mode=%s source=%s drift=%d",
            mode,
            desired_source,
            len(drifted_services),
        )


def run_listener() -> None:
    # ── Use Sentinel-aware client if available ────────────────────────────────
    try:
        from ha.redis_client import make_redis_client  # type: ignore[import]
        rdb = make_redis_client()
    except Exception:
        rdb = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

    # ── Leader election ───────────────────────────────────────────────────────
    try:
        from ha.leader_election import LeaderElector  # type: ignore[import]
        elector = LeaderElector(lease_name="variant-controller-leader")
        elector.start()
    except Exception as exc:
        log.warning("Leader election unavailable (%s) — running as sole instance", exc)
        elector = None  # type: ignore[assignment]

    pubsub = rdb.pubsub()
    pubsub.subscribe("milp:events")
    log.info("Subscribed to milp:events on %s", REDIS_HOST)

    last_reconcile = 0.0
    while True:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if message and message.get("type") == "message":
            # Only the leader handles incoming events; standby pods skip to
            # avoid duplicate rollouts when both pods hear the same pub.
            if _can_act(elector):
                try:
                    event = json.loads(message.get("data", "{}"))
                    service = event.get("service", "")
                    variant = event.get("variant", "standard")
                    mode = _get_mode(rdb)
                    log.info("Event received: service=%s variant=%s mode=%s -> reconcile", service, variant, mode)
                    reconcile_desired_state(rdb)
                    last_reconcile = time.time()
                except Exception as exc:
                    log.error("Event handler error: %s", exc, exc_info=True)
            else:
                log.debug("Standby: skipping event (not leader)")

        now = time.time()
        if now - last_reconcile >= RECONCILE_INTERVAL_S:
            if _can_act(elector):
                try:
                    reconcile_desired_state(rdb)
                except Exception as exc:
                    log.error("Reconcile loop error: %s", exc, exc_info=True)
            else:
                log.debug("Standby: skipping reconcile (not leader)")
            last_reconcile = now


if __name__ == "__main__":
    while True:
        try:
            run_listener()
        except Exception as exc:
            log.error("Redis connection lost: %s; retrying in 10s", exc)
            time.sleep(10)
