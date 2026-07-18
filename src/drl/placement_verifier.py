"""Tier-2 placement verifier for Digital Twin Async Veto flow.

The verifier watches `drl:placement:proposed` and validates schedulability on
KWOK twin nodes by creating one short-lived test pod **per target node**, each
pinned to that node via twin-node labels, so the K8s scheduler is forced to
admit or reject the placement exactly as the real cluster would.

Verification verdict is published back to Redis as either:
  - drl:placement:verified  (all per-node test-pods were Bound)
  - drl:placement:revoked   (any pod Unschedulable / timeout)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

import redis as redis_lib

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
except Exception:  # pragma: no cover - runtime dependency in cluster
    k8s_client = None
    k8s_config = None

try:
    from src.drl.twin_sync import _NODE_MAPPING, _make_k8s_v1api, _placement_to_resource, sync_once
except ImportError:
    from drl.twin_sync import _NODE_MAPPING, _make_k8s_v1api, _placement_to_resource, sync_once

try:
    from src.ha.redis_client import make_redis_client
except ImportError:
    from ha.redis_client import make_redis_client


log = logging.getLogger("placement-verifier")

_DEFAULT_REDIS_HOST = os.getenv("REDIS_HOST", "redis.default.svc.cluster.local")
_DEFAULT_REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
_POLL_INTERVAL_S = float(os.getenv("VERIFIER_POLL_INTERVAL_S", "1"))
_VERIFY_TIMEOUT_S = float(os.getenv("VERIFIER_TIMEOUT_S", "8"))
_POST_SYNC_SLEEP_S = float(os.getenv("VERIFIER_POST_SYNC_SLEEP_S", "0.5"))
_RESULT_TTL_S = int(os.getenv("VERIFIER_RESULT_TTL_S", "60"))
_POST_TIMEOUT_SLEEP_S = float(os.getenv("VERIFIER_POST_TIMEOUT_SLEEP_S", "33"))
_VERIFY_ACTIVE_MODES = {
    x.strip().lower()
    for x in os.getenv("VERIFIER_ACTIVE_MODES", "drl,hybrid").split(",")
    if x.strip()
}
_LATEST_REHYDRATE_MAX_AGE_S = float(os.getenv("VERIFIER_LATEST_REHYDRATE_MAX_AGE_S", "120"))
_REVERIFY_SAME_FP_INTERVAL_S = float(os.getenv("VERIFIER_REVERIFY_SAME_FP_INTERVAL_S", "30"))
_NAMESPACE = os.getenv("POD_NAMESPACE", "default")
_POD_NAME = os.getenv("POD_NAME", "placement-verifier")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fingerprint(proposed_raw: str) -> str:
    """Hash stable placement content, not volatile stats/timestamps.

    `drl:placement:proposed` may include changing telemetry fields each cycle.
    Fingerprinting whole payload would force unnecessary re-verification and
    create test-pod churn. We fingerprint only canonical placement mapping.
    """
    try:
        data = json.loads(proposed_raw)
        placement = data.get("placement", {})
        if isinstance(placement, dict):
            canonical = json.dumps(
                placement,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            return hashlib.md5(canonical.encode("utf-8")).hexdigest()[:12]
    except Exception:
        pass
    return hashlib.md5(proposed_raw.encode("utf-8")).hexdigest()[:12]


def _age_seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds())
    except Exception:
        return None


def _resources_per_twin_node(placement: dict) -> dict[str, tuple[int, int]]:
    """Return {twin_node_name: (cpu_mc, mem_mi)} for each twin-node that has at least
    one service assigned, based on the DRL proposed placement."""
    cpu_by_nid, mem_by_nid = _placement_to_resource(placement)
    result: dict[str, tuple[int, int]] = {}
    for nid, _real, twin in _NODE_MAPPING:
        cpu_cores = cpu_by_nid.get(nid, 0.0)
        mem_gb = mem_by_nid.get(nid, 0.0)
        if cpu_cores > 0 or mem_gb > 0:
            cpu_mc = max(1, int(round(cpu_cores * 1000 * 0.85)))
            mem_mi = max(1, int(round(mem_gb * 1024 * 0.85)))
            log.debug(
                "verifier: test-pod on %s 85%% budget: %dm CPU %dMi mem",
                twin, cpu_mc, mem_mi,
            )
            result[twin] = (cpu_mc, mem_mi)
    return result


def _make_test_pod_spec_for_node(name_prefix: str, cpu_mc: int, mem_mi: int, twin_node: str) -> dict:
    """Build a test-pod spec pinned to a specific KWOK twin-node."""
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            # Use generateName instead of a deterministic fingerprint-derived
            # name.  Re-verifying the same fingerprint can otherwise recreate
            # e.g. placement-test-abc123-1 while kube-scheduler still has an
            # assumed pod cache entry for the previous UID, producing
            # DefaultBinder "pod not found" / UID precondition races.
            "generateName": name_prefix,
            "namespace": _NAMESPACE,
            "labels": {
                "verifier": "placement-verifier",
                "verifier-id": _POD_NAME,
                "proposed-fingerprint": name_prefix.removeprefix("placement-test-").split("-")[0],
                "target-twin-node": twin_node,
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "tolerations": [
                {
                    "key": "kwok.x-k8s.io/node",
                    "operator": "Equal",
                    "value": "fake",
                    "effect": "NoSchedule",
                }
            ],
            "nodeSelector": {
                "type": "kwok",
                "twin-of": twin_node.removeprefix("twin-"),
            },
            "containers": [
                {
                    "name": "tester",
                    "image": "busybox",
                    "command": ["sleep", "3600"],
                    "resources": {
                        "requests": {
                            "cpu": f"{cpu_mc}m",
                            "memory": f"{mem_mi}Mi",
                        }
                    },
                }
            ],
        },
    }


class PlacementVerifier:
    def __init__(self) -> None:
        # Resolve the writable primary through Sentinel.  A plain long-lived
        # connection can remain pinned to a demoted primary after failover and
        # then fail every verdict/heartbeat write with READONLY.
        self._rdb = make_redis_client()
        self._v1 = self._init_k8s_client()
        self._last_fingerprint: str | None = None
        self._last_mode: str = "unknown"
        self._last_verify_at: float = 0.0
        if self._v1 is not None:
            self._cleanup_stale_test_pods()

    @staticmethod
    def _init_k8s_client():
        if k8s_client is None or k8s_config is None:
            log.warning("kubernetes client unavailable")
            return None
        try:
            return _make_k8s_v1api()
        except Exception as exc:
            log.warning("failed to initialize kubernetes client: %s", exc)
            return None

    def _publish(self, key: str, payload: dict) -> None:
        payload_raw = json.dumps(payload)
        self._rdb.setex(key, _RESULT_TTL_S, payload_raw)
        # Durable shadow key for dashboards/ops when short TTL key expires.
        if key in {"drl:placement:verified", "drl:placement:revoked"}:
            self._delete_opposite_verdict(key)
            self._rdb.set(f"{key}:latest", payload_raw)
            # Backward-compatible Tier-2 keys still used by some dashboards/tools.
            self._rdb.setex("placement:verifier:last", _RESULT_TTL_S, payload_raw)
            self._rdb.set("placement:verifier:latest", payload_raw)
            compat_key = (
                "tier2:placement:verified"
                if key == "drl:placement:verified"
                else "tier2:placement:revoked"
            )
            self._rdb.setex(compat_key, _RESULT_TTL_S, payload_raw)
            self._rdb.setex("tier2:placement:last_result", _RESULT_TTL_S, payload_raw)
            self._rdb.setex("drl:tier2_stats", _RESULT_TTL_S, payload_raw)

    def _delete_opposite_verdict(self, key: str) -> None:
        opposite = (
            "drl:placement:revoked"
            if key == "drl:placement:verified"
            else "drl:placement:verified"
        )
        compat_opposite = (
            "tier2:placement:revoked"
            if key == "drl:placement:verified"
            else "tier2:placement:verified"
        )
        self._rdb.delete(opposite, f"{opposite}:latest", compat_opposite)

    def _publish_heartbeat(self, fp: str) -> None:
        """Record that the verifier is actively observing the current proposal.

        Verdict timestamps intentionally represent when the KWOK check ran.
        This heartbeat lets dashboards distinguish an unchanged placement that is
        still being watched from a genuinely stale verifier process.
        """
        payload = {
            "verifier_id": _POD_NAME,
            "proposed_fingerprint": fp,
            "timestamp": _utc_now(),
        }
        self._rdb.setex(
            "drl:placement:verifier_heartbeat",
            max(_RESULT_TTL_S, int(_LATEST_REHYDRATE_MAX_AGE_S)),
            json.dumps(payload),
        )

    def _republish_latest_if_missing(self, fp: str) -> bool:
        """Rehydrate short-TTL verdict key from durable latest key.

        Returns True when a key is republished, False otherwise.
        """
        has_verified = bool(self._rdb.get("drl:placement:verified"))
        has_revoked = bool(self._rdb.get("drl:placement:revoked"))
        if has_verified or has_revoked:
            return False

        for base_key in ("drl:placement:verified", "drl:placement:revoked"):
            raw = self._rdb.get(f"{base_key}:latest")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if payload.get("proposed_fingerprint") != fp:
                continue
            age_s = _age_seconds(payload.get("timestamp"))
            if age_s is None or age_s > _LATEST_REHYDRATE_MAX_AGE_S:
                log.info(
                    "skip rehydrate %s for fp=%s (stale age=%.1fs max=%.1fs)",
                    base_key,
                    fp,
                    -1.0 if age_s is None else age_s,
                    _LATEST_REHYDRATE_MAX_AGE_S,
                )
                continue
            payload["rehydrated_at"] = _utc_now()
            payload["rehydrated"] = True
            payload["age_seconds"] = round(age_s, 3)
            self._publish(base_key, payload)
            log.info(
                "rehydrated %s from durable latest for fp=%s",
                base_key,
                fp,
            )
            return True
        return False

    def _should_reverify_same_fingerprint(self) -> bool:
        """Decide whether unchanged placement should be re-verified.

        This prevents a dead zone where:
          - fingerprint is unchanged,
          - short TTL verdict keys have expired,
          - durable latest verdict is too stale to rehydrate,
        and verifier would otherwise do nothing forever.
        """
        has_verified = bool(self._rdb.get("drl:placement:verified"))
        has_revoked = bool(self._rdb.get("drl:placement:revoked"))
        if has_verified or has_revoked:
            return False
        now = time.monotonic()
        return (now - self._last_verify_at) >= max(1.0, _REVERIFY_SAME_FP_INTERVAL_S)

    def _verify_placement_on_kwok(
        self, fp: str, node_resources: dict[str, tuple[int, int]]
    ) -> tuple[bool, str, list[str] | None]:
        """Create one test-pod per target twin-node and verify all are schedulable.

        Each pod is pinned to its specific twin-node via nodeSelector so the
        K8s scheduler cannot silently redirect it elsewhere.

        Returns:
            (ok, reason, verified_nodes) where verified_nodes lists the twin-node
            names that successfully received a binding, or None on failure.
        """
        if self._v1 is None:
            return False, "kwok_unavailable", None

        if not node_resources:
            return False, "empty_placement", None

        self._cleanup_stale_test_pods()

        for _twin, (_cpu_mc, _mem_mi) in node_resources.items():
            can_schedule, reason = self._check_node_capacity_available(_twin, _cpu_mc, _mem_mi)
            if not can_schedule:
                log.warning("capacity pre-check failed %s: %s", _twin, reason)
                return False, "insufficient_capacity", None

        created_pods: list[str] = []
        try:
            for i, (twin_node, (cpu_mc, mem_mi)) in enumerate(node_resources.items()):
                pod_name_prefix = f"placement-test-{fp}-{i}-"
                pod_body = _make_test_pod_spec_for_node(
                    pod_name_prefix, cpu_mc, mem_mi, twin_node
                )
                try:
                    created = self._v1.create_namespaced_pod(namespace=_NAMESPACE, body=pod_body)
                    metadata = getattr(created, "metadata", None)
                    pod_name = getattr(metadata, "name", None)
                    if not pod_name:
                        # Defensive fallback for tests or unusual clients.  In
                        # cluster, Kubernetes always returns the generated name.
                        pod_name = pod_name_prefix.rstrip("-")
                    created_pods.append(pod_name)
                    log.debug(
                        "created verifier pod %s for fp=%s target=%s",
                        pod_name,
                        fp,
                        twin_node,
                    )
                except Exception as exc:
                    log.warning("failed to create verifier pod %s*: %s", pod_name_prefix, exc)
                    return False, "kwok_unavailable", None

            deadline = time.monotonic() + _VERIFY_TIMEOUT_S
            pending: set[str] = set(created_pods)
            bound: list[str] = []

            while time.monotonic() < deadline and pending:
                time.sleep(0.5)
                still_pending: set[str] = set()
                for pod_name in pending:
                    try:
                        pod = self._v1.read_namespaced_pod(
                            name=pod_name, namespace=_NAMESPACE
                        )
                        spec = getattr(pod, "spec", None)
                        status = getattr(pod, "status", None)
                        node_name = getattr(spec, "node_name", None)
                        phase = getattr(status, "phase", None)
                        conditions = getattr(status, "conditions", None) or []

                        for cond in conditions:
                            if getattr(cond, "reason", None) == "Unschedulable":
                                log.warning(
                                    "pod %s unschedulable: %s",
                                    pod_name, getattr(cond, "message", ""),
                                )
                                return False, "pod_unschedulable", None

                        if phase == "Failed":
                            return False, "pod_unschedulable", None

                        if node_name:
                            bound.append(str(node_name))
                        else:
                            still_pending.add(pod_name)
                    except Exception as exc:
                        log.warning("error reading pod %s: %s", pod_name, exc)
                        still_pending.add(pod_name)
                pending = still_pending

            if pending:
                return False, "timeout", None
            return True, "ok", bound

        finally:
            for pod_name in created_pods:
                try:
                    self._v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=_NAMESPACE,
                        grace_period_seconds=0,
                    )
                    log.debug("cleanup: delete request sent for %s", pod_name)
                    self._wait_for_pod_deletion(pod_name, timeout_s=5.0)
                except Exception as exc:
                    if getattr(exc, "status", None) == 404:
                        log.debug("cleanup: pod %s already gone (404)", pod_name)
                    else:
                        log.error("cleanup: failed to delete pod %s: %s", pod_name, exc)

    def _cleanup_stale_test_pods(self) -> None:
        """Delete any lingering placement-test pods created by this verifier instance.

        Called before each new verification so that stale pods from timed-out or
        otherwise incomplete prior cycles do not occupy twin-node CPU/mem and
        cause the next test-pod to be Unschedulable.
        """
        try:
            pods = self._v1.list_namespaced_pod(
                namespace=_NAMESPACE,
                label_selector="verifier=placement-verifier",
            )
            for pod in pods.items:
                pod_name = pod.metadata.name
                try:
                    self._v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=_NAMESPACE,
                        grace_period_seconds=0,
                    )
                    log.debug("pre-cleanup: delete request sent for stale pod %s", pod_name)
                    self._wait_for_pod_deletion(pod_name, timeout_s=5.0)
                except Exception as exc:
                    log.warning("pre-cleanup: failed to delete stale pod %s: %s", pod_name, exc)
        except Exception as exc:
            log.warning("pre-cleanup: failed to list stale pods: %s", exc)

    def _wait_for_pod_deletion(self, pod_name: str, timeout_s: float = 5.0) -> bool:
        """Wait until pod is removed from the cluster or timeout expires."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                self._v1.read_namespaced_pod(name=pod_name, namespace=_NAMESPACE)
                time.sleep(0.2)
            except Exception:
                log.debug("cleanup: pod %s confirmed deleted", pod_name)
                return True
        log.warning("cleanup: pod %s still present after %.1fs", pod_name, timeout_s)
        return False

    @staticmethod
    def _parse_cpu_mc(val: str) -> int:
        """Parse CPU string to millicores ('500m' → 500, '2' → 2000)."""
        s = str(val)
        try:
            if s.endswith("m"):
                return int(s[:-1])
            return int(float(s) * 1000)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _parse_mem_mi(val: str) -> int:
        """Parse memory string to MiB ('512Mi' → 512, '1Gi' → 1024, '2048Ki' → 2)."""
        s = str(val)
        try:
            if s.endswith("Ki"):
                return max(0, int(s[:-2]) // 1024)
            if s.endswith("Mi"):
                return int(s[:-2])
            if s.endswith("Gi"):
                return int(s[:-2]) * 1024
            return max(0, int(s) // (1024 * 1024))
        except (ValueError, TypeError):
            return 0

    def _check_node_capacity_available(
        self, twin_node: str, cpu_mc: int, mem_mi: int
    ) -> tuple[bool, str]:
        """Check if twin node has enough available capacity for a test pod.

        Returns (True, 'ok') when capacity is sufficient, or when the check
        cannot be performed (K8s unavailable, API error) so pod creation is
        still attempted.  Terminating pods are excluded from used-resources.
        """
        if self._v1 is None:
            return True, "k8s_unavailable_skip"
        try:
            node = self._v1.read_node(twin_node)
            alloc = node.status.allocatable or {}
            total_cpu_mc = self._parse_cpu_mc(alloc.get("cpu", "4"))
            total_mem_mi = self._parse_mem_mi(alloc.get("memory", "3911Mi"))

            pods = self._v1.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={twin_node}"
            )
            used_cpu_mc = 0
            used_mem_mi = 0
            for pod in pods.items:
                phase = getattr(pod.status, "phase", None)
                if phase in ("Succeeded", "Failed"):
                    continue  # completed pods have released scheduler resources
                for container in pod.spec.containers or []:
                    req = (
                        container.resources.requests
                        if container.resources and container.resources.requests
                        else {}
                    )
                    used_cpu_mc += self._parse_cpu_mc(req.get("cpu", "0m"))
                    used_mem_mi += self._parse_mem_mi(req.get("memory", "0Mi"))

            available_cpu = max(0, total_cpu_mc - used_cpu_mc)
            available_mem = max(0, total_mem_mi - used_mem_mi)
            if available_cpu < cpu_mc or available_mem < mem_mi:
                return False, (
                    f"have cpu={available_cpu}m mem={available_mem}Mi,"
                    f" need cpu={cpu_mc}m mem={mem_mi}Mi"
                )
            return True, "ok"
        except Exception as exc:
            log.warning("capacity check error for %s, skipping: %s", twin_node, exc)
            return True, f"check_error_skip: {exc}"

    def verify_once(self, proposed_raw: str) -> None:
        self._last_verify_at = time.monotonic()
        fp = _fingerprint(proposed_raw)

        try:
            data = json.loads(proposed_raw)
            placement = data.get("placement", {})
            if not isinstance(placement, dict):
                raise ValueError("placement must be dict")
        except Exception:
            payload = {
                "ok": False,
                "reason": "invalid_payload",
                "proposed_fingerprint": fp,
                "timestamp": _utc_now(),
            }
            self._publish("drl:placement:revoked", payload)
            return

        sync_once(self._rdb)
        time.sleep(max(0.0, _POST_SYNC_SLEEP_S))

        node_resources = _resources_per_twin_node(placement)
        ok, reason, verified_nodes = self._verify_placement_on_kwok(fp, node_resources)

        if not ok and reason == "timeout":
            log.warning(
                "post-timeout sleep: %.0fs — waiting for kube-scheduler assumed-pod cache to expire",
                _POST_TIMEOUT_SLEEP_S,
            )
            time.sleep(_POST_TIMEOUT_SLEEP_S)

        if ok:
            payload = {
                "ok": True,
                "verified_nodes": verified_nodes,
                "verifier_id": _POD_NAME,
                "proposed_fingerprint": fp,
                "timestamp": _utc_now(),
            }
            self._publish("drl:placement:verified", payload)
            log.info("verification passed fp=%s nodes=%s", fp, verified_nodes)
            return

        payload = {
            "ok": False,
            "reason": reason,
            "proposed_fingerprint": fp,
            "timestamp": _utc_now(),
        }
        self._publish("drl:placement:revoked", payload)
        log.warning("verification revoked fp=%s reason=%s", fp, reason)

    def run_forever(self) -> None:
        log.info(
            "placement verifier started: redis=%s:%s poll=%.1fs timeout=%.1fs active_modes=%s",
            _DEFAULT_REDIS_HOST,
            _DEFAULT_REDIS_PORT,
            _POLL_INTERVAL_S,
            _VERIFY_TIMEOUT_S,
            sorted(_VERIFY_ACTIVE_MODES),
        )
        while True:
            try:
                mode = (self._rdb.get("system:mode") or "milp").strip().lower()
                if mode != self._last_mode:
                    log.info(
                        "verifier mode changed: %s -> %s (active=%s)",
                        self._last_mode,
                        mode,
                        mode in _VERIFY_ACTIVE_MODES,
                    )
                    if mode in _VERIFY_ACTIVE_MODES and self._last_mode not in _VERIFY_ACTIVE_MODES:
                        # Force one validation after mode enters active set.
                        self._last_fingerprint = None
                    self._last_mode = mode

                if mode not in _VERIFY_ACTIVE_MODES:
                    time.sleep(_POLL_INTERVAL_S)
                    continue

                proposed_raw = self._rdb.get("drl:placement:proposed")
                if not proposed_raw:
                    time.sleep(_POLL_INTERVAL_S)
                    continue

                fp = _fingerprint(proposed_raw)
                self._publish_heartbeat(fp)
                if fp != self._last_fingerprint:
                    self.verify_once(proposed_raw)
                    self._last_fingerprint = fp
                else:
                    # Placement unchanged: keep recent verdict key available
                    # by republishing from durable latest when needed.
                    rehydrated = self._republish_latest_if_missing(fp)
                    if not rehydrated and self._should_reverify_same_fingerprint():
                        log.info(
                            "re-verifying unchanged fingerprint fp=%s after stale/missing verdict",
                            fp,
                        )
                        self.verify_once(proposed_raw)
            except redis_lib.RedisError as exc:
                log.warning("redis error in verifier loop: %s", exc)
            except Exception as exc:
                log.error("unexpected verifier loop error: %s", exc, exc_info=True)

            time.sleep(_POLL_INTERVAL_S)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    verifier = PlacementVerifier()
    verifier.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
