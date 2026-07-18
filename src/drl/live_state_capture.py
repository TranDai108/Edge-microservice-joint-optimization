"""Capture live cluster states from Redis for offline DRL fine-tuning.

Reads the live 44-dim state vector via EdgeEnv._build_live_state() every
INTERVAL seconds for N cycles and writes them as JSONL to OUTPUT_FILE.

Each line:
    {"state": [...44 floats...], "milp_placement": {...}, "milp_j": float,
     "timestamp": "ISO8601", "cycle": int}

Usage:
    PYTHONPATH=src python3 src/drl/live_state_capture.py \
        --cycles 200 --interval 5 --output results/live_states.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import redis as redis_lib  # noqa: E402

try:
    from src.drl.edge_env import EdgeEnv  # noqa: E402
except ImportError:
    from drl.edge_env import EdgeEnv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("live-capture")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture live cluster states from Redis.")
    p.add_argument("--cycles", type=int, default=200, help="Number of state samples to capture.")
    p.add_argument("--interval", type=float, default=5.0, help="Seconds between samples.")
    p.add_argument(
        "--output",
        type=str,
        default="results/live_states.jsonl",
        help="Output JSONL file path.",
    )
    p.add_argument("--redis-host", type=str, default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rdb = redis_lib.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        rdb.ping()
        log.info("Redis connected at %s:%s", args.redis_host, args.redis_port)
    except redis_lib.RedisError as exc:
        log.error("Cannot reach Redis: %s", exc)
        sys.exit(1)

    env = EdgeEnv(mock=False)
    env.rdb = rdb

    written = 0
    skipped = 0
    with open(out_path, "w") as fh:
        for cycle in range(1, args.cycles + 1):
            t0 = time.perf_counter()
            try:
                state = env._build_live_state()

                milp_j: float | None = None
                milp_placement: dict = {}
                raw_milp = rdb.get("milp:placement")
                if raw_milp:
                    m = json.loads(raw_milp)
                    milp_j = m.get("objective")
                    milp_placement = m.get("placement", {})

                record = {
                    "cycle": cycle,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "state": state.tolist(),
                    "milp_placement": milp_placement,
                    "milp_j": milp_j,
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                written += 1

                if cycle % 20 == 0 or cycle == 1:
                    log.info(
                        "cycle %d/%d — state[41:44]=[%.3f, %.3f, %.3f]  milp_j=%.4f",
                        cycle, args.cycles,
                        state[41], state[42], state[43],
                        milp_j if milp_j is not None else float("nan"),
                    )

            except Exception as exc:
                log.warning("cycle %d: error reading state: %s", cycle, exc)
                skipped += 1

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, args.interval - elapsed))

    log.info("Done. Written %d samples (%d skipped) to %s", written, skipped, out_path)


if __name__ == "__main__":
    main()
