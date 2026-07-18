"""Digital Twin state sync — mirrors real cluster load into KWOK virtual nodes.

Tier 2 of the Digital Twin stack: keeps twin-edge-nodes-1..4 synchronised
with the actual cluster resource usage so the K8s scheduler makes accurate
admission decisions when DRL test-pods are scheduled on virtual nodes.

Sync cycle (default every 15 s, aligned with Prometheus scrape interval):
    1. Read resource_usage and placement from milp:placement (Redis)
    2. Compute background load = total_load − service_load_from_placement
    3. PATCH each twin-edge-nodes-X with annotations:
           kwok.x-k8s.io/usage-cpu    = "<millicores>m"
           kwok.x-k8s.io/usage-memory = "<MiB>Mi"
    4. KWOK virtual nodes now reflect *background* load only (no service load)
       so that placement_verifier test-pods represent DRL placements without
       double-counting the current service allocation.

Node mapping:
    edge-nodes-1 → twin-edge-nodes-1   (n0 in MILP model)
    edge-nodes-2 → twin-edge-nodes-2   (n1)
    edge-nodes-3 → twin-edge-nodes-3   (n2)
    edge-nodes-4 → twin-edge-nodes-4   (n3)

Usage
─────
    # As a one-shot sync:
    python3 -m src.drl.twin_sync --redis-host localhost --once

    # As a continuous background loop (run inside K8s pod or systemd):
    python3 -m src.drl.twin_sync --redis-host localhost --interval 15

    # Import and run as a background thread from drl_agent.py:
    from src.drl.twin_sync import TwinSyncLoop
    loop = TwinSyncLoop(redis_host="localhost")
    loop.start_background()
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from typing import Optional

import redis as redis_lib

log = logging.getLogger("twin-sync")


def _make_k8s_v1api():
    """Return a CoreV1Api client using standard in-cluster auth first.

    Falls back to kubeconfig when running outside a cluster (local dev / CI).
    """
    from kubernetes import client as _kc, config as _kcfg  # noqa: PLC0415

    try:
        cfg = _kc.Configuration()
        _kcfg.load_incluster_config(client_configuration=cfg)
        return _kc.CoreV1Api(api_client=_kc.ApiClient(configuration=cfg))
    except Exception:
        _kcfg.load_kube_config()
        return _kc.CoreV1Api()


# ── Node mapping: MILP node_id → real hostname → twin hostname ────────────────
_NODE_MAPPING: list[tuple[str, str, str]] = [
    ("n0", "edge-nodes-1", "twin-edge-nodes-1"),
    ("n1", "edge-nodes-2", "twin-edge-nodes-2"),
    ("n2", "edge-nodes-3", "twin-edge-nodes-3"),
    ("n3", "edge-nodes-4", "twin-edge-nodes-4"),
]

# Minimum usage values to report (prevents KWOK from showing 0 on idle nodes)
_MIN_CPU_MILLICORES: int = 50    # 50m
_MIN_MEM_MI: int = 128            # 128Mi

_VARIANT_CPU_REQ: dict[str, float] = {
    "standard": 0.20,
    "yolo26-nano": 0.50,
    "yolo26-small": 1.00,
    "yolo26-medium": 2.00,
    "qwen-1.5b-nano": 0.30,
    "llama-3b-small": 0.60,
    "gemma2-2b-medium": 0.40,
}
_VARIANT_MEM_REQ: dict[str, float] = {
    "standard": 0.25,
    "yolo26-nano": 0.78,
    "yolo26-small": 1.10,
    "yolo26-medium": 1.65,
    "qwen-1.5b-nano": 0.50,
    "llama-3b-small": 1.00,
    "gemma2-2b-medium": 0.70,
}
_NODE_MAP: dict[str, str] = {
    "edge-nodes-1": "n0",
    "edge-nodes-2": "n1",
    "edge-nodes-3": "n2",
    "edge-nodes-4": "n3",
}


# ── Core sync logic ───────────────────────────────────────────────────────────

def _placement_to_resource(placement: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Convert placement dict to per-node CPU/mem totals.

    Expected payload shape:
        {"svc-name": {"node": "edge-nodes-X", "variant": "..."}, ...}
    """
    cpu: dict[str, float] = {f"n{i}": 0.0 for i in range(4)}
    mem: dict[str, float] = {f"n{i}": 0.0 for i in range(4)}
    for info in placement.values():
        if not isinstance(info, dict):
            continue
        node_id = _NODE_MAP.get(str(info.get("node", "")), "n0")
        variant = str(info.get("variant", "standard"))
        cpu[node_id] += _VARIANT_CPU_REQ.get(variant, 0.0)
        mem[node_id] += _VARIANT_MEM_REQ.get(variant, 0.0)
    return cpu, mem


def _read_background_load(
    rdb: redis_lib.Redis,
) -> tuple[dict[str, float], dict[str, float]]:
    """Read *background* CPU (cores) and memory (GB) load per node.

    Background load = total node load (milp:placement resource_usage)
                    − load from the 6 tracked services (milp:placement.placement).

    KWOK twin-nodes are annotated with background-only load so that test-pods
    created by placement_verifier represent DRL-proposed services without
    double-counting the current service allocation.

    Returns:
        (cpu_bg, mem_bg) — both keyed by MILP node_id ("n0".."n3").
        Falls back to zero values when Redis key is missing.
    """
    cpu_bg: dict[str, float] = {f"n{i}": 0.0 for i in range(4)}
    mem_bg: dict[str, float] = {f"n{i}": 0.0 for i in range(4)}

    try:
        raw = rdb.get("milp:placement")
        if not raw:
            return cpu_bg, mem_bg
        data = json.loads(raw)
        for node_id, val in data.get("resource_usage", {}).items():
            if node_id in cpu_bg:
                cpu_bg[node_id] = max(0.0, float(val))
        for node_id, val in data.get("mem_usage", {}).items():
            if node_id in mem_bg:
                mem_bg[node_id] = max(0.0, float(val))
        placement = data.get("placement", {})
        if isinstance(placement, dict):
            svc_cpu, svc_mem = _placement_to_resource(placement)
            for node_id in cpu_bg:
                cpu_bg[node_id] = max(0.0, cpu_bg[node_id] - svc_cpu.get(node_id, 0.0))
                mem_bg[node_id] = max(0.0, mem_bg[node_id] - svc_mem.get(node_id, 0.0))
    except (redis_lib.RedisError, json.JSONDecodeError, ValueError) as exc:
        log.warning("twin_sync: failed to read milp:placement — %s", exc)

    return cpu_bg, mem_bg


def _patch_twin_node(
    node_name: str,
    cpu_millicores: int,
    mem_mi: int,
) -> bool:
    """PATCH a KWOK virtual node's usage annotations via the kubernetes client.

    Returns True on success, False on any error.
    """
    try:
        v1 = _make_k8s_v1api()
        body = {
            "metadata": {
                "annotations": {
                    "kwok.x-k8s.io/usage-cpu":    f"{cpu_millicores}m",
                    "kwok.x-k8s.io/usage-memory": f"{mem_mi}Mi",
                }
            }
        }
        v1.patch_node(node_name, body)
        return True
    except ImportError:
        log.warning(
            "twin_sync: kubernetes library not installed. "
            "Run: pip install kubernetes"
        )
        return False
    except Exception as exc:
        log.warning("twin_sync: failed to patch node %s — %s", node_name, exc)
        return False


def sync_once(
    rdb: redis_lib.Redis,
) -> dict[str, dict]:
    """Run one synchronisation cycle.  Returns a summary dict for logging/monitoring.

    Reads background load from Redis and patches all 4 twin nodes with it.
    Safe to call even if KWOK nodes don't exist yet (failures are logged, not raised).
    """
    cpu_used, mem_used = _read_background_load(rdb)

    summary: dict[str, dict] = {}
    for node_id, real_name, twin_name in _NODE_MAPPING:
        cpu_cores = cpu_used.get(node_id, 0.0)
        mem_gb = mem_used.get(node_id, 0.0)

        cpu_mc = max(_MIN_CPU_MILLICORES, int(cpu_cores * 1000))
        mem_mi = max(_MIN_MEM_MI, int(mem_gb * 1024))

        success = _patch_twin_node(twin_name, cpu_mc, mem_mi)
        summary[twin_name] = {
            "real_node": real_name,
            "cpu_millicores": cpu_mc,
            "mem_mi": mem_mi,
            "patched": success,
        }
        log.debug(
            "twin_sync: %s ← %s  cpu=%dm mem=%dMi patched=%s",
            twin_name, real_name, cpu_mc, mem_mi, success,
        )

    return summary


# ── Continuous background loop ────────────────────────────────────────────────

class TwinSyncLoop:
    """Continuous background thread that syncs twin node annotations every N seconds.

    Designed to run alongside drl_agent.py so twin nodes always reflect
    the latest cluster state before Digital Twin validation runs.

    Example::
        loop = TwinSyncLoop(redis_host="localhost", interval=15)
        loop.start_background()
        # ... drl_agent continues running
        loop.stop()
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        interval: int = 15,
    ) -> None:
        self.interval = interval
        self._rdb = redis_lib.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_background(self) -> None:
        """Start the sync loop as a daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="twin-sync",
        )
        self._thread.start()
        log.info("TwinSyncLoop started (interval=%ds)", self.interval)

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                summary = sync_once(self._rdb)
                n_ok = sum(1 for v in summary.values() if v["patched"])
                log.info(
                    "twin_sync: cycle complete — %d/%d nodes patched",
                    n_ok, len(summary),
                )
            except Exception as exc:
                log.error("twin_sync: unexpected error in sync cycle — %s", exc)
            elapsed = time.perf_counter() - t0
            self._stop_event.wait(max(0.0, self.interval - elapsed))


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync real cluster load into KWOK twin-node annotations."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--interval", type=int, default=15,
        help="Seconds between sync cycles (default 15).",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Run a single sync cycle and exit.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    rdb = redis_lib.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)

    if args.once:
        summary = sync_once(rdb)
        for name, info in summary.items():
            print(f"{name}: cpu={info['cpu_millicores']}m mem={info['mem_mi']}Mi patched={info['patched']}")
        return 0

    loop = TwinSyncLoop(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        interval=args.interval,
    )
    loop.start_background()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        loop.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
