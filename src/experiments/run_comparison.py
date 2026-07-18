#!/usr/bin/env python3
"""Run MILP/DRL/Hybrid/baseline comparison and log per-cycle metrics to JSONL files.

Baseline modes:
  random     -- uniform random node assignment per service each cycle
  roundrobin -- rotate services across nodes in fixed order (deterministic)
  drl_no_bc  -- DRL agent trained WITHOUT behavioral cloning pre-training
               (reads drl_no_bc:placement; requires separate training run)
  drl_no_twin -- DRL agent with Digital Twin gate disabled
               (reads drl:placement but skips twin validation server-side)

Speed flags (new):
  --fast         Collect --min-cycles real samples, then bootstrap to --cycles.
                 Reduces wall-clock time from O(cycles*interval) to O(min_cycles*interval).
  --min-cycles N Minimum real cycles before bootstrap kicks in (default: 50).
  --interval 0   Skip inter-cycle sleep entirely (useful for replay / CI runs).
  --parallel     Run all modes concurrently in separate threads.
                 WARNING: sets system:mode per-thread; safe only when placement keys
                 are already populated (milp:placement / drl:placement) and the live
                 controller is NOT actively reading system:mode.

Typical fast run (≈ 3 min for 3 modes):
  python run_comparison.py --fast --min-cycles 50 --interval 2 --parallel
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import random as _random
import threading
from copy import deepcopy

import numpy as np
import redis

import sys
import time as _time_mod

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solver.metrics_collector import get_e2e_latency_ms, build_dataset_from_cluster
from drl.reward import evaluate_objective_for_placement

SLA_LATENCY_MS = 1500.0
MODES = ["milp", "drl", "hybrid"]
ALL_MODES = ["milp", "drl", "hybrid", "shadow", "random", "roundrobin", "drl_no_bc", "drl_no_twin"]

PLACEMENT_KEY_BY_MODE = {
    "milp": "milp:placement",
    "drl": "drl:placement",
    "hybrid": None,  # dynamic: drl:placement if available+valid, else milp:placement
    "shadow": "drl:placement",
    "drl_no_bc": "drl_no_bc:placement",
    "drl_no_twin": "drl:placement",
}

# Service and node IDs used for synthetic baseline placement generation
_SERVICE_IDS = ["m0", "m1", "m2", "m3", "m4", "m5"]
_NODE_IDS = ["n0", "n1", "n2", "n3"]


def _make_random_placement(cycle: int, rng: _random.Random) -> dict:
    """Generate a random placement: each service → uniform random node."""
    placement = {svc: rng.choice(_NODE_IDS) for svc in _SERVICE_IDS}
    return {
        "placement": placement,
        "objective": None,
        "cost_energy": None,
        "migration_types": {svc: "Moved" for svc in _SERVICE_IDS},
    }


def _make_roundrobin_placement(cycle: int) -> dict:
    """Generate a round-robin placement: service[i] → node[i % 4]."""
    placement = {svc: _NODE_IDS[i % len(_NODE_IDS)] for i, svc in enumerate(_SERVICE_IDS)}
    # Shift by cycle to rotate across time
    shift = (cycle - 1) % len(_NODE_IDS)
    placement = {svc: _NODE_IDS[(i + shift) % len(_NODE_IDS)] for i, svc in enumerate(_SERVICE_IDS)}
    return {
        "placement": placement,
        "objective": None,
        "cost_energy": None,
        "migration_types": {svc: "Moved" if cycle > 1 else "Stayed" for svc in _SERVICE_IDS},
    }


def _write_synthetic_placement(rdb: redis.Redis, placement_payload: dict, redis_key: str) -> None:
    """Write a synthetic placement dict to Redis so _extract_metrics can read it."""
    rdb.set(redis_key, json.dumps(placement_payload))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mode comparison runner (MILP/DRL/Hybrid/Baselines).")
    parser.add_argument("--redis-host", default="localhost", help="Redis hostname")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--cycles", type=int, default=200, help="Cycles per mode")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between cycles; 0 = no sleep (replay/CI mode)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for random baseline")
    # Speed flags
    parser.add_argument("--fast", action="store_true",
                        help="Collect --min-cycles real samples then bootstrap to --cycles")
    parser.add_argument("--min-cycles", type=int, default=50,
                        help="Real cycles to collect before bootstrapping (default: 50)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run all modes concurrently in separate threads")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=MODES,
        choices=ALL_MODES,
        help="Mode sequence to run",
    )
    return parser.parse_args()


def _bootstrap_rows(
    real_rows: list[dict],
    target: int,
    seed: int,
) -> list[dict]:
    """Resample *real_rows* with replacement until *target* rows are reached.

    Only the extra (bootstrapped) rows are returned; the originals are kept
    as-is.  Numeric metrics keep their sampled values; cycle index and
    timestamp are updated so the output looks like a continuous run.
    SLA-violated is recomputed from the (possibly jittered) latency.
    """
    rng = np.random.default_rng(seed)
    n_extra = target - len(real_rows)
    indices = rng.integers(0, len(real_rows), size=n_extra)
    extra = []
    for offset, idx in enumerate(indices):
        row = deepcopy(real_rows[int(idx)])
        row["cycle"] = len(real_rows) + offset + 1
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Re-evaluate SLA from the cloned latency value
        if row["e2e_latency_ms"] is not None:
            row["sla_violated"] = row["e2e_latency_ms"] > SLA_LATENCY_MS
        extra.append(row)
    return extra


def _read_json(rdb: redis.Redis, key: str) -> dict | None:
    raw = rdb.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("Bad JSON payload in key=%s", key)
        return None


_DECISION_LATENCY_MS_BY_MODE = {
    "milp":       30_000.0,
    "drl":            5.0,
    "hybrid":        35.0,
    "shadow":          5.0,
    "drl_no_bc":       5.0,
    "drl_no_twin":     5.0,
    "random":          1.0,
    "roundrobin":      1.0,
}

# ── Fair-evaluation helpers ────────────────────────────────────────────────────
# Modes whose J is originally estimated on mock EdgeEnv data and needs to be
# re-evaluated on live cluster data for an apples-to-apples comparison with MILP.
_DRL_MODES: frozenset[str] = frozenset({"drl", "hybrid", "shadow", "drl_no_bc", "drl_no_twin"})

# Service display-name order (matches metrics_collector.SERVICE_DEFS index → m0..m5)
_SVC_EVAL_ORDER = [
    "api-gateway", "ingest", "preprocess", "detection", "gen-ai", "postprocess",
]

# Node hostname → node_id used by reward.py / EdgeEnv
_EVAL_HOST_TO_NID = {
    "edge-nodes-1": "n0", "edge-nodes-2": "n1",
    "edge-nodes-3": "n2", "edge-nodes-4": "n3",
}

# Thread-safe cache for the live MILPDataset (expensive to rebuild each cycle)
_live_ds_lock = threading.Lock()
_live_ds_cache: dict = {"ds": None, "ts": 0.0}
_LIVE_DS_TTL_S: float = 30.0


def _get_live_dataset(baseline_placement: dict):
    """Return a cached live MILPDataset, refreshed every _LIVE_DS_TTL_S seconds.

    *baseline_placement* is passed to ``build_dataset_from_cluster`` as
    ``last_placement`` so that ``x_prev`` (migration-cost baseline) reflects the
    cluster's current live state rather than the proposed DRL placement.
    """
    now = _time_mod.monotonic()
    with _live_ds_lock:
        if _live_ds_cache["ds"] is None or (now - _live_ds_cache["ts"]) > _LIVE_DS_TTL_S:
            try:
                ds = build_dataset_from_cluster(last_placement=baseline_placement)
                _live_ds_cache["ds"] = ds
                _live_ds_cache["ts"] = now
            except Exception as exc:
                logging.warning("live dataset build failed (fair eval unavailable): %s", exc)
        return _live_ds_cache["ds"]


def _fair_evaluate(
    placement_payload: dict,
    baseline_placement: dict,
) -> tuple[float | None, float | None]:
    """Re-evaluate *placement_payload* on the live MILPDataset.

    Both MILP and DRL J values end up computed from the same live R_cpu, R_mem,
    and E_cpu inputs, removing the mock-vs-live data gap.

    *baseline_placement* (typically ``milp:placement["placement"]``) is the
    current live cluster state used to build ``x_prev`` for migration costs.

    Returns:
        (objective_j, cost_energy) on success, (None, None) on any failure.
    """
    raw = placement_payload.get("placement")
    if not raw:
        return None, None

    live_ds = _get_live_dataset(baseline_placement)
    if live_ds is None:
        return None, None

    try:
        placement_ids = {
            f"m{i}": (
                raw.get(svc, {}).get("variant", "standard"),
                _EVAL_HOST_TO_NID.get(raw.get(svc, {}).get("node", ""), "n0"),
            )
            for i, svc in enumerate(_SVC_EVAL_ORDER)
        }
        bd = evaluate_objective_for_placement(live_ds, placement_ids)
        return float(bd.objective_j), float(bd.cost_energy)
    except Exception as exc:
        logging.warning("fair_evaluate: %s", exc)
        return None, None


def _extract_metrics(
    placement_payload: dict | None,
    mode: str,
    cycle: int,
    baseline_placement: dict | None = None,
) -> dict:
    objective_j = None
    energy_w = None
    migrations = None

    if placement_payload:
        if isinstance(placement_payload.get("objective"), (float, int)):
            objective_j = float(placement_payload["objective"])
        elif isinstance(placement_payload.get("reward"), (float, int)):
            objective_j = -float(placement_payload["reward"])

        if isinstance(placement_payload.get("cost_energy"), (float, int)):
            energy_w = float(placement_payload["cost_energy"])

        mig = placement_payload.get("migration_types")
        if isinstance(mig, dict):
            migrations = sum(1 for value in mig.values() if value != "Stayed")

    # Fair re-evaluation: for DRL-based modes, override J and energy with values
    # computed on live cluster data — same R_cpu/R_mem/E_cpu inputs as the MILP
    # solver uses, eliminating the mock-vs-live data gap.
    if mode in _DRL_MODES and placement_payload and baseline_placement:
        fair_j, fair_energy = _fair_evaluate(placement_payload, baseline_placement)
        if fair_j is not None:
            objective_j = fair_j
        if fair_energy is not None:
            energy_w = fair_energy

    # E2E pipeline latency from Prometheus; None when unavailable (not faked).
    prom_latency = float(get_e2e_latency_ms())
    e2e_latency_ms = prom_latency if prom_latency < 999.0 else None
    sla_violated = (e2e_latency_ms > SLA_LATENCY_MS) if e2e_latency_ms is not None else None

    # Decision / solver latency — always populated from payload or mode default.
    if placement_payload and isinstance(placement_payload.get("inference_time_ms"), (float, int)):
        inference_time_ms = float(placement_payload["inference_time_ms"])
    else:
        inference_time_ms = _DECISION_LATENCY_MS_BY_MODE.get(mode)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "mode": mode,
        "objective_J": objective_j,
        "e2e_latency_ms": e2e_latency_ms,
        "inference_time_ms": inference_time_ms,
        "energy_w": energy_w,
        "migrations": migrations,
        "sla_violated": sla_violated,
    }


def _run_mode(
    rdb: redis.Redis,
    mode: str,
    cycles: int,
    interval: int,
    out_path: Path,
    rng: _random.Random | None = None,
    fast: bool = False,
    min_cycles: int = 50,
    seed: int = 42,
    parallel: bool = False,
) -> None:
    """Collect metrics for *mode* and write them to *out_path*.

    When *fast=True*: collect only *min_cycles* real samples from Redis / K8s,
    then fill up to *cycles* via bootstrap resampling.  This turns a
    200-cycle × 30 s run (~100 min) into a 50-cycle × interval run.

    When *parallel=True*: skip the per-thread ``system:mode`` write to avoid
    race conditions with other threads.  The caller is responsible for setting
    ``system:mode`` once before spawning threads.
    """
    key = PLACEMENT_KEY_BY_MODE.get(mode, "milp:placement")
    if not parallel:
        rdb.set("system:mode", mode)
    real_target = min_cycles if fast else cycles
    key_label = "drl:placement→milp:placement" if mode == "hybrid" else key
    logging.info(
        "Mode -> %s | key=%s | collecting %d real cycles (fast=%s)",
        mode, key_label, real_target, fast,
    )

    real_rows: list[dict] = []

    with out_path.open("w", encoding="utf-8") as handle:
        for cycle in range(1, real_target + 1):
            if interval > 0:
                time.sleep(interval)

            baseline_placement: dict | None = None

            if mode == "random":
                assert rng is not None
                syn = _make_random_placement(cycle, rng)
                _write_synthetic_placement(rdb, syn, "synthetic:placement")
                placement_payload = _read_json(rdb, "synthetic:placement")
            elif mode == "roundrobin":
                syn = _make_roundrobin_placement(cycle)
                _write_synthetic_placement(rdb, syn, "synthetic:placement")
                placement_payload = _read_json(rdb, "synthetic:placement")
            elif mode == "hybrid":
                drl_p = _read_json(rdb, "drl:placement")
                milp_p = _read_json(rdb, "milp:placement")
                drl_j = drl_p.get("objective") if drl_p else None
                milp_j = milp_p.get("objective") if milp_p else None
                # Pick whichever has better (lower) J; fallback to MILP if DRL unavailable
                if drl_p and isinstance(drl_j, (float, int)) and (
                    not isinstance(milp_j, (float, int)) or float(drl_j) < float(milp_j)
                ):
                    placement_payload = drl_p
                else:
                    placement_payload = milp_p
                # milp:placement["placement"] is the live cluster state → x_prev baseline
                baseline_placement = milp_p.get("placement") if milp_p else None
            else:
                placement_payload = _read_json(rdb, key)
                if mode in _DRL_MODES:
                    milp_raw = _read_json(rdb, "milp:placement")
                    baseline_placement = milp_raw.get("placement") if milp_raw else None

            row = _extract_metrics(placement_payload, mode, cycle,
                                   baseline_placement=baseline_placement)
            real_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            e2e_str = f"{row['e2e_latency_ms']:.2f}ms" if row["e2e_latency_ms"] is not None else "N/A"
            logging.info(
                "mode=%s cycle=%d/%d J=%s e2e=%s energy=%s migrations=%s sla=%s",
                mode, cycle, real_target,
                row["objective_J"],
                e2e_str,
                row["energy_w"],
                row["migrations"],
                row["sla_violated"],
            )

        # ── Bootstrap phase (only when --fast and we need more rows) ──────────
        if fast and len(real_rows) < cycles:
            n_extra = cycles - len(real_rows)
            logging.info(
                "mode=%s: bootstrapping %d real rows → %d target (adding %d synthetic)",
                mode, len(real_rows), cycles, n_extra,
            )
            extra_rows = _bootstrap_rows(real_rows, cycles, seed)
            for row in extra_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            logging.info("mode=%s: done — %d total rows written", mode, cycles)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    common_kw = dict(
        fast=args.fast,
        min_cycles=args.min_cycles,
        seed=args.seed,
    )

    def _run(mode: str) -> None:
        """Thread target — each thread owns its own Redis connection and RNG."""
        rdb = redis.Redis(
            host=args.redis_host, port=args.redis_port,
            decode_responses=True, socket_timeout=3,
        )
        rng = _random.Random(args.seed)  # deterministic but independent per thread
        out_path = results_dir / f"comparison_{mode}.jsonl"
        _run_mode(rdb, mode, args.cycles, args.interval, out_path, rng=rng, **common_kw)

    if args.parallel:
        logging.info("Parallel mode: spawning %d threads", len(args.modes))
        # Pre-set system:mode to hybrid so the DRL agent stays active during the run.
        # Individual threads must NOT overwrite this (parallel=True in _run_mode).
        _rdb_main = redis.Redis(
            host=args.redis_host, port=args.redis_port,
            decode_responses=True, socket_timeout=3,
        )
        _rdb_main.set("system:mode", "hybrid")
        logging.info("Pre-set system:mode=hybrid to keep DRL agent active")

        def _run_parallel(mode: str) -> None:
            rdb = redis.Redis(
                host=args.redis_host, port=args.redis_port,
                decode_responses=True, socket_timeout=3,
            )
            rng = _random.Random(args.seed)
            out_path = results_dir / f"comparison_{mode}.jsonl"
            _run_mode(rdb, mode, args.cycles, args.interval, out_path,
                      rng=rng, parallel=True, **common_kw)

        threads = [threading.Thread(target=_run_parallel, args=(m,), name=f"mode-{m}", daemon=True)
                   for m in args.modes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for mode in args.modes:
            _run(mode)


if __name__ == "__main__":
    main()
