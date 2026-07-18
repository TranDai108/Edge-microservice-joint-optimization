"""Export DRL storm-projection correction samples from Redis to JSONL.

Use this after running adaptive agent in shadow/hybrid mode. The output JSONL
is suitable for downstream BC/fine-tune pipelines where each line is one
correction sample: raw action proposed by policy and projected safe action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import redis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DRL storm correction samples")
    p.add_argument("--redis-host", default="redis.default.svc.cluster.local")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--redis-db", type=int, default=0)
    p.add_argument("--key", default="drl:storm_corrections")
    p.add_argument("--limit", type=int, default=5000, help="max newest samples to export")
    p.add_argument(
        "--output",
        default="results/storm_corrections.jsonl",
        help="output JSONL path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rdb = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        decode_responses=True,
    )

    raw_items = rdb.lrange(args.key, 0, max(0, args.limit - 1))
    parsed: list[dict] = []
    for raw in raw_items:
        try:
            parsed.append(json.loads(raw))
        except Exception:
            continue

    # Redis LPUSH inserts newest first. Write chronological order for replay.
    parsed.reverse()
    with out_path.open("w", encoding="utf-8") as f:
        for item in parsed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if not parsed:
        print(f"[export] no valid records in key={args.key}")
        return 0

    avg_before = sum(float(x.get("migration_before", 0)) for x in parsed) / len(parsed)
    avg_after = sum(float(x.get("migration_after", 0)) for x in parsed) / len(parsed)
    print(
        "[export] wrote",
        len(parsed),
        "samples to",
        out_path,
        "| avg migration before/after =",
        f"{avg_before:.3f}/{avg_after:.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

