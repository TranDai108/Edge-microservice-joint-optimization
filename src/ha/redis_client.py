"""Redis Sentinel-aware client factory.

Provides a drop-in redis.Redis-compatible client that:
  • Connects through Redis Sentinel when REDIS_SENTINEL_HOSTS is set
  • Falls back to a plain Redis connection to REDIS_HOST if Sentinel is absent
  • Retries transient connection errors with exponential back-off
  • Supports idempotent placement writes via set-if-not-changed semantics

Usage:
    from src.ha.redis_client import make_redis_client

    rdb = make_redis_client()
    rdb.ping()   # works with both Sentinel and plain Redis

Environment variables:
    REDIS_SENTINEL_HOSTS   comma-separated host:port list  (optional)
    REDIS_SENTINEL_MASTER  Sentinel master name            (default: mymaster)
    REDIS_HOST             plain Redis host                (default: redis.default.svc.cluster.local)
    REDIS_PORT             plain Redis port                (default: 6379)
    REDIS_CONNECT_TIMEOUT  socket connect timeout (s)      (default: 5)
    REDIS_MAX_RETRIES      retries on startup              (default: 10)
    REDIS_RETRY_BACKOFF_S  base back-off between retries   (default: 2)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import redis
from redis.sentinel import Sentinel

log = logging.getLogger("redis-client")

_SENTINEL_HOSTS_RAW = os.getenv("REDIS_SENTINEL_HOSTS", "")
_SENTINEL_MASTER = os.getenv("REDIS_SENTINEL_MASTER", "mymaster")
_REDIS_HOST = os.getenv("REDIS_HOST", "redis.default.svc.cluster.local")
_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "5"))
_MAX_RETRIES = int(os.getenv("REDIS_MAX_RETRIES", "10"))
_RETRY_BACKOFF_S = float(os.getenv("REDIS_RETRY_BACKOFF_S", "2"))


def _parse_port(raw: str, default: int = 6379) -> int:
    """Parse REDIS_PORT safely.

    Kubernetes service-link injection sets REDIS_PORT='tcp://10.x.x.x:6379'
    instead of a bare integer.  Extract the numeric portion in that case.
    """
    raw = raw.strip()
    if raw.startswith("tcp://") or raw.startswith("http"):
        # e.g. 'tcp://10.43.44.134:6379' → 6379
        try:
            return int(raw.rsplit(":", 1)[-1])
        except ValueError:
            return default
    try:
        return int(raw)
    except ValueError:
        return default


_REDIS_PORT = _parse_port(os.getenv("REDIS_PORT", "6379"))


def _parse_sentinel_hosts(raw: str) -> list[tuple[str, int]]:
    """Parse 'host:port,host:port' into [(host, port), ...]."""
    hosts = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            h, p = entry.rsplit(":", 1)
            hosts.append((h.strip(), int(p)))
        else:
            hosts.append((entry, 26379))
    return hosts


def make_redis_client(decode_responses: bool = True) -> redis.Redis:
    """Return a Redis client connected via Sentinel or plain host.

    Retries connection with exponential back-off to survive pod startup races
    where Redis may not be reachable immediately.
    """
    sentinel_hosts = _parse_sentinel_hosts(_SENTINEL_HOSTS_RAW)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if sentinel_hosts:
                log.info(
                    "Connecting via Sentinel: master=%s sentinels=%s",
                    _SENTINEL_MASTER, sentinel_hosts,
                )
                sentinel = Sentinel(
                    sentinel_hosts,
                    socket_timeout=_CONNECT_TIMEOUT,
                    decode_responses=decode_responses,
                )
                # discover_master() raises if Sentinel quorum not reached
                rdb = sentinel.master_for(
                    _SENTINEL_MASTER,
                    socket_timeout=_CONNECT_TIMEOUT,
                    decode_responses=decode_responses,
                )
            else:
                log.info(
                    "Connecting to plain Redis: %s:%d", _REDIS_HOST, _REDIS_PORT
                )
                rdb = redis.Redis(
                    host=_REDIS_HOST,
                    port=_REDIS_PORT,
                    socket_timeout=_CONNECT_TIMEOUT,
                    decode_responses=decode_responses,
                )

            rdb.ping()
            log.info("Redis connected OK (attempt %d/%d)", attempt, _MAX_RETRIES)
            return rdb

        except (redis.RedisError, OSError) as exc:
            backoff = _RETRY_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                "Redis connect attempt %d/%d failed: %s — retrying in %.1fs",
                attempt, _MAX_RETRIES, exc, backoff,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(min(backoff, 30.0))
            else:
                raise RuntimeError(
                    f"Could not connect to Redis after {_MAX_RETRIES} attempts"
                ) from exc

    # unreachable, but satisfies type checkers
    raise RuntimeError("make_redis_client: exhausted retries")


def idempotent_setex(rdb: redis.Redis, key: str, value: str, ttl: int) -> bool:
    """Write key only if the value has changed, renew TTL regardless.

    Avoids unnecessary write amplification during reconciliation loops where
    the same placement is re-emitted every cycle.

    Returns True if a write was performed, False if value was unchanged.
    """
    existing = rdb.get(key)
    if existing == value:
        # Value unchanged — just touch the TTL to prevent expiry
        rdb.expire(key, ttl)
        return False
    rdb.setex(key, ttl, value)
    return True
