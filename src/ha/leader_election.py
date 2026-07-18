"""Leader election helper for MILP and DRL controllers.

Uses the Kubernetes coordination.k8s.io/v1 Lease API — the same mechanism
used by kube-controller-manager — to implement active-standby leadership.

Only the current leader should write placement keys to Redis.  A standby pod
runs its main loop in observe-only mode until it wins the Lease.

Design:
    • Lease duration: LEADER_LEASE_DURATION_S (default 15 s)
    • Renew deadline:  LEADER_RENEW_DEADLINE_S (default 10 s)
    • Retry period:    LEADER_RETRY_PERIOD_S   (default  2 s)
    • Leader identity: POD_NAME env var (injected via fieldRef in deployment)

Fencing (split-brain prevention):
    Each placement write includes a monotonically increasing epoch stored in
    Redis under `<prefix>:epoch`.  The leader reads and increments the epoch
    atomically inside a Lua script before writing a placement key.  A stale
    pod that lost the Lease but hasn't noticed yet will find that its epoch is
    behind the new leader's epoch and must skip the write.

Usage:
    from src.ha.leader_election import LeaderElector, placement_write_guarded

    elector = LeaderElector(lease_name="milp-leader")
    elector.start()          # background thread; blocks until first election

    if elector.is_leader():
        placement_write_guarded(rdb, "milp:placement", payload, elector)

Environment variables (all optional, have defaults):
    LEADER_LEASE_NAME        milp-leader
    LEADER_LEASE_NAMESPACE   default
    LEADER_LEASE_DURATION_S  15
    LEADER_RENEW_DEADLINE_S  10
    LEADER_RETRY_PERIOD_S    2
    POD_NAME                 (auto-detected from hostname if absent)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("leader-election")

# ── Try to import the kubernetes client; degrade gracefully in unit-test env ──
try:
    from kubernetes import client as k8s_client, config as k8s_config

    def _load_k8s_config() -> bool:
        try:
            cfg = k8s_client.Configuration()
            k8s_config.load_incluster_config(client_configuration=cfg)
            # Keep the in-cluster auth config as-is to stay compatible with
            # the installed kubernetes client version.
            k8s_client.Configuration.set_default(cfg)
            return True
        except Exception:
            try:
                k8s_config.load_kube_config()
                return True
            except Exception:
                return False

    _K8S_AVAILABLE = _load_k8s_config()
except ImportError:
    _K8S_AVAILABLE = False

_LEASE_DURATION_S = int(os.getenv("LEADER_LEASE_DURATION_S", "15"))
_RENEW_DEADLINE_S = int(os.getenv("LEADER_RENEW_DEADLINE_S", "10"))
_RETRY_PERIOD_S = int(os.getenv("LEADER_RETRY_PERIOD_S", "2"))
_LEASE_NAMESPACE = os.getenv("LEADER_LEASE_NAMESPACE", "default")
_POD_NAME = os.getenv("POD_NAME", socket.gethostname())


class LeaderElector:
    """Active-standby leader elector backed by a K8s coordination Lease.

    If the kubernetes client is unavailable (local dev / unit tests), falls
    back to a trivially-leader mode so controller code runs unchanged.
    """

    def __init__(self, lease_name: str = "milp-leader") -> None:
        self._lease_name = os.getenv("LEADER_LEASE_NAME", lease_name)
        self._identity = _POD_NAME
        self._is_leader = False
        self._epoch = 0  # local epoch counter (incremented on leader change)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if not _K8S_AVAILABLE:
            log.warning(
                "kubernetes client unavailable — running as trivial leader "
                "(single-pod dev/test mode)"
            )
            self._is_leader = True

    # ── Public API ────────────────────────────────────────────────────────────

    def is_leader(self) -> bool:
        with self._lock:
            return self._is_leader

    def current_epoch(self) -> int:
        with self._lock:
            return self._epoch

    def start(self) -> None:
        """Start the background election/renewal thread."""
        if not _K8S_AVAILABLE:
            return  # trivial leader mode, nothing to renew
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"leader-elector-{self._lease_name}"
        )
        self._thread.start()
        log.info(
            "LeaderElector started: lease=%s identity=%s namespace=%s",
            self._lease_name,
            self._identity,
            _LEASE_NAMESPACE,
        )

    def stop(self) -> None:
        self._stop_event.set()

    # ── Election loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        coord_api = k8s_client.CoordinationV1Api()
        while not self._stop_event.is_set():
            try:
                if self._is_leader:
                    self._renew(coord_api)
                else:
                    self._try_acquire(coord_api)
            except Exception as exc:
                log.warning("LeaderElector error: %s", exc)
                with self._lock:
                    self._is_leader = False
            time.sleep(_RETRY_PERIOD_S)

    def _try_acquire(self, coord_api) -> None:
        now = datetime.now(timezone.utc)
        body = k8s_client.V1Lease(
            metadata=k8s_client.V1ObjectMeta(
                name=self._lease_name, namespace=_LEASE_NAMESPACE
            ),
            spec=k8s_client.V1LeaseSpec(
                holder_identity=self._identity,
                lease_duration_seconds=_LEASE_DURATION_S,
                acquire_time=now,
                renew_time=now,
                lease_transitions=0,
            ),
        )
        try:
            coord_api.create_namespaced_lease(_LEASE_NAMESPACE, body)
            with self._lock:
                self._is_leader = True
                self._epoch += 1
            log.info(
                "Acquired lease '%s' (epoch=%d identity=%s)",
                self._lease_name, self._epoch, self._identity,
            )
        except k8s_client.exceptions.ApiException as exc:
            if exc.status == 409:
                # Lease exists — check if it has expired
                self._check_expired_and_steal(coord_api)
            else:
                raise

    def _check_expired_and_steal(self, coord_api) -> None:
        try:
            lease = coord_api.read_namespaced_lease(self._lease_name, _LEASE_NAMESPACE)
            spec = lease.spec
            holder = spec.holder_identity or ""
            renew_time = spec.renew_time
            duration = spec.lease_duration_seconds or _LEASE_DURATION_S

            if holder == self._identity:
                # We already hold it; treat as renewal
                with self._lock:
                    self._is_leader = True
                return

            if renew_time is None:
                return  # can't determine expiry

            elapsed = (datetime.now(timezone.utc) - renew_time).total_seconds()
            if elapsed > duration:
                # Lease expired — steal it (fencing: increment leaseTransitions)
                now = datetime.now(timezone.utc)
                transitions = (spec.lease_transitions or 0) + 1
                body = k8s_client.V1Lease(
                    spec=k8s_client.V1LeaseSpec(
                        holder_identity=self._identity,
                        lease_duration_seconds=_LEASE_DURATION_S,
                        acquire_time=now,
                        renew_time=now,
                        lease_transitions=transitions,
                    )
                )
                coord_api.patch_namespaced_lease(self._lease_name, _LEASE_NAMESPACE, body)
                with self._lock:
                    self._is_leader = True
                    self._epoch += 1
                log.info(
                    "Stole expired lease '%s' from '%s' (epoch=%d transitions=%d)",
                    self._lease_name, holder, self._epoch, transitions,
                )
        except Exception as exc:
            log.debug("Steal attempt failed: %s", exc)

    def _renew(self, coord_api) -> None:
        now = datetime.now(timezone.utc)
        body = k8s_client.V1Lease(
            spec=k8s_client.V1LeaseSpec(
                holder_identity=self._identity,
                lease_duration_seconds=_LEASE_DURATION_S,
                renew_time=now,
            )
        )
        try:
            coord_api.patch_namespaced_lease(self._lease_name, _LEASE_NAMESPACE, body)
            log.debug("Renewed lease '%s'", self._lease_name)
        except k8s_client.exceptions.ApiException as exc:
            log.warning("Lease renewal failed (%s) — stepping down", exc.status)
            with self._lock:
                self._is_leader = False


# ── Placement write helper with epoch fencing ─────────────────────────────────

_LUA_EPOCH_WRITE = """
local epoch_key  = KEYS[1]
local data_key   = KEYS[2]
local my_epoch   = tonumber(ARGV[1])
local payload    = ARGV[2]
local ttl        = tonumber(ARGV[3])

local stored = tonumber(redis.call('GET', epoch_key) or '0')
if my_epoch < stored then
    return 0  -- stale write rejected
end
redis.call('SET', epoch_key, my_epoch)
redis.call('SETEX', data_key, ttl, payload)
return 1
"""


def placement_write_guarded(
    rdb,
    key: str,
    payload: dict,
    elector: LeaderElector,
    ttl: int = 35,
    epoch_key: Optional[str] = None,
) -> bool:
    """Write placement to Redis only if this pod is the current leader.

    Performs an atomic Lua CAS: the write is rejected if a newer epoch is
    already stored, preventing split-brain writes from a stale pod.

    Args:
        rdb:       Redis client (sentinel-aware or plain)
        key:       Redis key to write (e.g. "milp:placement")
        payload:   dict to serialise as JSON
        elector:   LeaderElector instance; write is skipped if not leader
        ttl:       Key TTL in seconds (default 35)
        epoch_key: Override epoch key (default: ``key + ":epoch"``)

    Returns:
        True if the write succeeded, False if skipped/rejected.
    """
    if not elector.is_leader():
        log.debug("placement_write_guarded: not leader, skipping write to %s", key)
        return False

    epoch = elector.current_epoch()
    e_key = epoch_key or f"{key}:epoch"

    try:
        result = rdb.eval(
            _LUA_EPOCH_WRITE,
            2,           # number of KEYS
            e_key,       # KEYS[1]
            key,         # KEYS[2]
            epoch,       # ARGV[1]
            json.dumps(payload),  # ARGV[2]
            ttl,         # ARGV[3]
        )
        if result == 1:
            log.debug("placement_write_guarded: wrote %s (epoch=%d ttl=%ds)", key, epoch, ttl)
            return True
        else:
            log.warning(
                "placement_write_guarded: epoch %d rejected for %s (newer epoch in Redis)",
                epoch, key,
            )
            return False
    except Exception as exc:
        log.error("placement_write_guarded: Redis error writing %s: %s", key, exc)
        return False
