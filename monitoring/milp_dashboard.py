#!/usr/bin/env python3
"""
MILP Control System Dashboard — Enhanced Edition
────────────────────────────────────────────────
Real-time visibility into:
  • MILP placement & objective decomposition
  • Per-node CPU / RAM utilisation gauges (live cAdvisor / node_exporter)
  • Dual-energy Kepler + DRAM power breakdown per node
  • Detection QoS: accuracy vs variant, E2E latency vs SLA
  • Per-container resource usage table
  • MILP model pressure, Kubernetes overcommit, and runtime saturation per node
  • Migration events feed (current cycle)
  • Prometheus scrape-health / Redis connectivity health panel
"""

import json
import hashlib
import os
import sys
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

_TZ_HCM = ZoneInfo("Asia/Ho_Chi_Minh")  # GMT+7 — Hanoi / Ho Chi Minh City


def _fmt_ts(ts, fmt: str = "%H:%M:%S") -> str:
    """Convert a timestamp (Unix float OR ISO-8601 string) to GMT+7 formatted string."""
    try:
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=_TZ_HCM)
        else:
            # ISO string — may be UTC-naive or UTC-aware (with Z / +00:00)
            s = str(ts).replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s)
            except ValueError:
                return str(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(_TZ_HCM)
        return dt.strftime(fmt)
    except Exception:
        return str(ts)


def _parse_iso_utc(ts):
    try:
        if not ts:
            return None
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _age_seconds(ts):
    dt = _parse_iso_utc(ts)
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def _tier2_fingerprint(raw_payload: str) -> Optional[str]:
    """Match placement_verifier._fingerprint without importing runtime modules."""
    try:
        data = json.loads(raw_payload)
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
    return hashlib.md5(raw_payload.encode("utf-8")).hexdigest()[:12]


def _tier2_verdict_stale(
    verdict: Optional[dict],
    verdict_age_s: Optional[float],
    current_fp: Optional[str],
    heartbeat: Optional[dict],
    heartbeat_age_s: Optional[float],
) -> bool:
    if not verdict:
        return False

    verdict_fp = verdict.get("proposed_fingerprint")
    if current_fp and verdict_fp != current_fp:
        return True

    heartbeat_fp = heartbeat.get("proposed_fingerprint") if heartbeat else None
    heartbeat_current = (
        bool(current_fp)
        and heartbeat_fp == current_fp
        and isinstance(heartbeat_age_s, (int, float))
        and heartbeat_age_s <= TIER2_STALE_AFTER_S
    )
    if heartbeat_current and verdict_fp == current_fp:
        return False

    return verdict_age_s is None or verdict_age_s > TIER2_STALE_AFTER_S


def _tier2_drop_superseded_verdict(out: dict) -> None:
    """Hide an older opposite verdict for the same current proposal.

    Redis keeps separate durable `verified:latest` and `revoked:latest` keys. A
    later successful verification should supersede an older timeout/revocation
    for the same proposal fingerprint; otherwise the dashboard shows both boxes
    and the old one keeps the global stale warning alive.
    """
    current_fp = out.get("current_fingerprint")
    if not current_fp or not out.get("verified") or not out.get("revoked"):
        return

    candidates: list[tuple[datetime, str]] = []
    for key in ("verified", "revoked"):
        verdict = out.get(key)
        if verdict.get("proposed_fingerprint") != current_fp:
            return
        dt = _parse_iso_utc(verdict.get("timestamp"))
        if dt is None:
            return
        candidates.append((dt, key))

    if len(candidates) != 2:
        return

    _newest_dt, active_key = max(candidates, key=lambda item: item[0])
    inactive_key = "revoked" if active_key == "verified" else "verified"
    out[f"{inactive_key}_superseded"] = True
    out[inactive_key] = None
    out[f"{inactive_key}_age_s"] = None
    out[f"{inactive_key}_stale"] = False
    out["source"] = out.get(f"{active_key}_source") or out.get("source", "none")
from pathlib import Path
from collections import deque

import redis
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ── Project src path ─────────────────────────────────────────────────────────
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

K8S_IMPORT_ERROR: Optional[str] = None
try:
    import k8s_client
except Exception as exc:
    k8s_client = None
    K8S_IMPORT_ERROR = str(exc)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
REDIS_HOST          = os.getenv("REDIS_HOST", "redis.default.svc.cluster.local")
REDIS_PORT          = int(os.getenv("REDIS_PORT", "6379"))
# REDIS_PRIMARY_HOST: pod IP of redis-primary (reachable from host even when
# cluster DNS is not).  Set via env or detected at startup from kubectl.
REDIS_PRIMARY_HOST  = os.getenv("REDIS_PRIMARY_HOST", "")
REDIS_SENTINEL_HOST = os.getenv("REDIS_SENTINEL_HOST", "redis-sentinel.default.svc.cluster.local")
REDIS_SENTINEL_PORT = int(os.getenv("REDIS_SENTINEL_PORT", "26379"))
REDIS_SENTINEL_NAME = os.getenv("REDIS_SENTINEL_NAME", "mymaster")
NAMESPACE     = os.getenv("K8S_NAMESPACE", "default")
KUBECTL_BIN   = os.getenv("KUBECTL_BIN", "kubectl")
PROM_URL      = os.getenv("PROM_URL", "http://localhost:32090/api/v1/query")
SLA_LATENCY_MS      = float(os.getenv("SLA_LATENCY_MS", "1500.0"))
CONTROL_INTERVAL    = int(os.getenv("CONTROL_INTERVAL", "30"))
SOLVER_TIMEOUT_S    = int(os.getenv("SOLVER_TIMEOUT_S", "120"))
MAX_FALLBACK_RATIO  = float(os.getenv("MAX_FALLBACK_RATIO", "0.6"))
TIER2_STALE_AFTER_S = float(os.getenv("TIER2_STALE_AFTER_S", "120"))
NODES               = ["edge-nodes-1", "edge-nodes-2", "edge-nodes-3", "edge-nodes-4"]
SERVICES            = ["api-gateway", "ingest", "preprocess", "detection", "gen-ai", "postprocess"]
MILP_ID_MAP   = {"m0": "api-gateway", "m1": "ingest", "m2": "preprocess",
                  "m3": "detection", "m4": "gen-ai", "m5": "postprocess"}
SIM_RESULTS_FILE = os.getenv(
    "SIM_RESULTS_FILE",
    str(Path(__file__).resolve().parent.parent / "results" / "client_results.jsonl"),
)

# ── Palette ───────────────────────────────────────────────────────────────────
NODE_COLORS   = {
    "edge-nodes-1": "#4C9BE8",
    "edge-nodes-2": "#58C4A0",
    "edge-nodes-3": "#E8854C",
    "edge-nodes-4": "#A78BFA",
}
VARIANT_COLORS = {
    "standard":     "#90EE90",
    "yolo26-nano":  "#FFD700",
    "yolo26-small": "#FFA500",
    "yolo26-medium":"#FF6347",
}
SVC_COLORS = {
    "api-gateway": "#1f77b4", "ingest": "#ff7f0e",
    "preprocess":  "#2ca02c", "detection": "#d62728",
    "gen-ai": "#17a2b8", "postprocess": "#9467bd",
}
MIGRATION_COLORS = {
    "Stayed": "#4CAF50",
    "Node Migration": "#FF9800",
    "AI Model Redeployment": "#2196F3",
    "Node Migration + AI Redeployment": "#9C27B0",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MILP + DRL Edge Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-mode CSS injection ────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Metric delta colors */
  [data-testid="stMetricDelta"] svg { display: none; }
  .metric-ok   { color: #4CAF50 !important; font-weight: 600; }
  .metric-warn { color: #FF9800 !important; font-weight: 600; }
  .metric-err  { color: #F44336 !important; font-weight: 600; }

  /* Tighten tabs */
  [data-baseweb="tab-list"] { gap: 6px; }
  [data-baseweb="tab"] { padding: 6px 14px; border-radius: 6px; }

  /* Section header divider */
  .section-header {
    font-size: 1.05rem; font-weight: 700;
    letter-spacing: .04em; color: #a0c4ff;
    border-bottom: 1px solid #2d3748; padding-bottom: 4px; margin-bottom: 8px;
  }
  /* Card-like container */
  .card {
    background: #1e2535; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
    border: 1px solid #2d3748;
  }
  /* Badge pills */
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: .78rem; font-weight: 600; color: #fff; margin: 2px;
  }
  /* Mode banner */
  .mode-milp  { background:#2563EB; color:#fff; padding:4px 18px;
                border-radius:8px; font-weight:700; font-size:1rem; }
  .mode-shadow{ background:#D97706; color:#fff; padding:4px 18px;
                border-radius:8px; font-weight:700; font-size:1rem; }
  .mode-drl   { background:#16A34A; color:#fff; padding:4px 18px;
                border-radius:8px; font-weight:700; font-size:1rem; }
    .mode-hybrid{ background:#14B8A6; color:#fff; padding:4px 18px;
                                border-radius:8px; font-weight:700; font-size:1rem; }
  .mode-unknown{ background:#6B7280; color:#fff; padding:4px 18px;
                border-radius:8px; font-weight:700; font-size:1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data-fetching helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_redis_client():
    """Return a Redis client connected to the writable master.

    Strategy (in order):
    1. Try REDIS_PRIMARY_HOST (pod IP, bypasses DNS, works from host).
    2. Try REDIS_HOST cluster-DNS name (works inside cluster).
    3. Try localhost / 127.0.0.1 — but only if they are master, not replica.
    4. Ask redis-sentinel for the current master address.
    """
    def _is_master(rdb) -> bool:
        """Return True only if this connection is to a writable master."""
        try:
            info = rdb.info("replication")
            return info.get("role") == "master"
        except Exception:
            return False

    def _try_host(host: str, port: int = REDIS_PORT) -> "redis.Redis | None":
        try:
            rdb = redis.Redis(host=host, port=port,
                              decode_responses=True, socket_timeout=2)
            rdb.ping()
            if _is_master(rdb):
                return rdb
        except Exception:
            pass
        return None

    # ── Auto-detect primary pod IP via kubectl if not set ────────────────────
    primary_ip = REDIS_PRIMARY_HOST
    if not primary_ip:
        try:
            import subprocess  # already imported at top-level but safe here
            result = subprocess.run(
                "kubectl get pod redis-primary-0 -n default "
                "-o jsonpath='{.status.podIP}' 2>/dev/null",
                shell=True, capture_output=True, text=True, timeout=3,
            )
            primary_ip = result.stdout.strip().strip("'")
        except Exception:
            primary_ip = ""

    # ── 1-3. Direct candidates (master-only) ─────────────────────────────────
    for host in filter(None, [primary_ip, REDIS_HOST, "localhost", "127.0.0.1"]):
        rdb = _try_host(host)
        if rdb is not None:
            st.session_state["redis_host_ok"] = host
            return rdb

    # ── 4. Sentinel fallback — resolve master dynamically ────────────────────
    # Try sentinel pod IP if cluster DNS not reachable
    sentinel_candidates = [REDIS_SENTINEL_HOST]
    try:
        import subprocess
        result = subprocess.run(
            "kubectl get pod -n default -l app=redis-sentinel "
            "-o jsonpath='{.items[0].status.podIP}' 2>/dev/null",
            shell=True, capture_output=True, text=True, timeout=3,
        )
        sentinel_ip = result.stdout.strip().strip("'")
        if sentinel_ip:
            sentinel_candidates.insert(0, sentinel_ip)
    except Exception:
        pass

    for sentinel_host in sentinel_candidates:
        try:
            sentinel = redis.sentinel.Sentinel(
                [(sentinel_host, REDIS_SENTINEL_PORT)],
                socket_timeout=2,
                decode_responses=True,
            )
            rdb = sentinel.master_for(REDIS_SENTINEL_NAME, socket_timeout=2)
            rdb.ping()
            st.session_state["redis_host_ok"] = f"sentinel({sentinel_host})→master"
            return rdb
        except Exception:
            pass

    st.session_state["redis_host_ok"] = ""
    return None


def _kubectl(cmd: str, timeout: int = 6) -> str:
    try:
        r = subprocess.run(f"{KUBECTL_BIN} {cmd}", shell=True,
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_cpu_cores(value: Any) -> float:
    """Parse a Kubernetes CPU quantity into cores."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        if text.endswith("m"):
            return float(text[:-1]) / 1000.0
        return float(text)
    except Exception:
        return 0.0


def _parse_memory_gib(value: Any) -> float:
    """Parse a Kubernetes memory quantity into GiB."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    units = {
        "Ki": 1 / (1024 * 1024),
        "Mi": 1 / 1024,
        "Gi": 1.0,
        "Ti": 1024.0,
        "K": 1000 / (1024 ** 3),
        "M": 1000 ** 2 / (1024 ** 3),
        "G": 1000 ** 3 / (1024 ** 3),
        "T": 1000 ** 4 / (1024 ** 3),
    }
    try:
        for suffix, factor in units.items():
            if text.endswith(suffix):
                return float(text[:-len(suffix)]) * factor
        return float(text) / (1024 ** 3)
    except Exception:
        return 0.0


def _prom(query: str, timeout: int = 5) -> Optional[float]:
    """Single PromQL instant query → first scalar."""
    try:
        r = requests.get(PROM_URL, params={"query": query}, timeout=timeout)
        results = r.json().get("data", {}).get("result", [])
        return float(results[0]["value"][1]) if results else None
    except Exception:
        return None


def _prom_vector(query: str, timeout: int = 5) -> list[dict]:
    """PromQL instant query → list of {metric, value} dicts."""
    try:
        r = requests.get(PROM_URL, params={"query": query}, timeout=timeout)
        return r.json().get("data", {}).get("result", [])
    except Exception:
        return []


def _prom_range(query: str, start: float, end: float,
                step: int = 15, timeout: int = 8) -> list[dict]:
    """PromQL range query for sparklines."""
    try:
        r = requests.get(
            PROM_URL.replace("/query", "/query_range"),
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=timeout,
        )
        return r.json().get("data", {}).get("result", [])
    except Exception:
        return []


@st.cache_data(ttl=5, show_spinner=False)
def get_recent_k8s_warning_events() -> list[dict[str, str]]:
    """Return structured Warning events from the namespace."""
    raw = _kubectl(
        f"get events -n {NAMESPACE} --field-selector type=Warning -o json",
        timeout=8,
    )
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []

    rows: list[dict[str, str]] = []
    for ev in data.get("items", []):
        md = ev.get("metadata", {}) or {}
        obj = ev.get("involvedObject", {}) or {}
        ts = (
            ev.get("eventTime")
            or ev.get("lastTimestamp")
            or md.get("creationTimestamp")
            or ""
        )
        rows.append({
            "timestamp": str(ts),
            "reason": str(ev.get("reason", "")),
            "type": str(ev.get("type", "")),
            "object_kind": str(obj.get("kind", "")),
            "object_name": str(obj.get("name", "")),
            "message": str(ev.get("message", "")),
            "count": str(ev.get("count", 1)),
            "source": str((ev.get("source", {}) or {}).get("component", "")),
        })

    def _ts_key(x: dict[str, str]) -> str:
        return x.get("timestamp", "")

    rows.sort(key=_ts_key, reverse=True)
    return rows


# ── Redis placement readers ────────────────────────────────────────────────────

@st.cache_data(ttl=4, show_spinner=False)
def get_milp_placement() -> Optional[dict]:
    rdb = _get_redis_client()
    if rdb:
        try:
            raw = rdb.get("milp:placement")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    # kubectl fallback
    out = _kubectl(f"exec deploy/redis -n {NAMESPACE} -- redis-cli GET milp:placement")
    if out:
        try:
            return json.loads(out)
        except Exception:
            pass
    return None


@st.cache_data(ttl=4, show_spinner=False)
def get_confirmed_placement() -> Optional[dict]:
    rdb = _get_redis_client()
    if rdb:
        try:
            raw = rdb.get("milp:confirmed_placement")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    out = _kubectl(f"exec deploy/redis -n {NAMESPACE} -- redis-cli GET milp:confirmed_placement")
    if out:
        try:
            return json.loads(out)
        except Exception:
            pass
    return None


@st.cache_data(ttl=4, show_spinner=False)
def get_controller_sync_status() -> Optional[dict]:
    rdb = _get_redis_client()
    if rdb:
        try:
            raw = rdb.get("milp:controller_sync")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    out = _kubectl(f"exec deploy/redis -n {NAMESPACE} -- redis-cli GET milp:controller_sync")
    if out:
        try:
            return json.loads(out)
        except Exception:
            pass
    return None


# ── Live Prometheus metrics (all cached 10 s) ─────────────────────────────────

@st.cache_data(ttl=10, show_spinner=False)
def get_node_cpu_util(hostname: str) -> Optional[float]:
    q = f'1 - avg(rate(node_cpu_seconds_total{{mode="idle",node="{hostname}"}}[60s]))'
    v = _prom(q)
    if v is None:
        # Instance-IP fallback — discovered from kube API
        if k8s_client:
            ip = k8s_client.get_node_internal_ip(hostname)
            if ip:
                q2 = f'1 - avg(rate(node_cpu_seconds_total{{mode="idle",instance="{ip}:9100"}}[60s]))'
                v = _prom(q2)
    return round(max(0.0, min(1.0, v)), 4) if v is not None else None


@st.cache_data(ttl=10, show_spinner=False)
def get_node_capacity_cpu(hostname: str) -> float:
    if k8s_client:
        return k8s_client.get_node_capacity_cpu(hostname) or 2.0
    return 2.0


@st.cache_data(ttl=10, show_spinner=False)
def get_node_capacity_mem(hostname: str) -> float:
    if k8s_client:
        return k8s_client.get_node_allocatable_memory(hostname) or 2.0
    return 2.0


@st.cache_data(ttl=10, show_spinner=False)
def get_node_mem_used_gb(hostname: str) -> Optional[float]:
    """Sum all container working-set bytes on this node across namespaces."""
    q = f'sum(container_memory_working_set_bytes{{node="{hostname}"}}) / 1073741824'
    v = _prom(q)
    return round(v, 3) if v is not None else None


@st.cache_data(ttl=10, show_spinner=False)
def get_kepler_node_watts(hostname: str) -> Optional[float]:
    q = f'rate(kepler_node_platform_joules_total{{instance="{hostname}"}}[60s])'
    return _prom(q)


@st.cache_data(ttl=10, show_spinner=False)
def get_kepler_dram_watts(hostname: str) -> Optional[float]:
    q = f'rate(kepler_node_dram_joules_total{{instance="{hostname}"}}[60s])'
    return _prom(q)


@st.cache_data(ttl=10, show_spinner=False)
def get_container_cpu_cores(svc: str) -> Optional[float]:
    q = (f'sum(rate(container_cpu_usage_seconds_total{{container="{svc}",'
         f'namespace="{NAMESPACE}"}}[60s]))')
    v = _prom(q)
    return round(v, 4) if v is not None else None


@st.cache_data(ttl=10, show_spinner=False)
def get_container_mem_gb(svc: str) -> Optional[float]:
    q = (f'sum(container_memory_working_set_bytes{{container="{svc}",'
         f'namespace="{NAMESPACE}"}}) / 1073741824')
    v = _prom(q)
    return round(v, 3) if v is not None else None


@st.cache_data(ttl=10, show_spinner=False)
def get_e2e_latency() -> Optional[float]:
    v = _prom("rate(e2e_latency_ms_sum[60s]) / rate(e2e_latency_ms_count[60s])")
    if v is not None:
        return round(v, 2)
    # Redis fallback: api-gateway writes kltn:e2e_latency (TTL 30 s) after each
    # completed pipeline request — available even when Prometheus is unreachable.
    rdb = _get_redis_client()
    if rdb:
        try:
            raw = rdb.get("kltn:e2e_latency")
            if raw is not None:
                return round(float(raw), 2)
        except Exception:
            pass
    return None


@st.cache_data(ttl=10, show_spinner=False)
def get_detection_accuracy(variant: Optional[str] = None) -> Optional[float]:
    if variant:
        q = (f'rate(detection_accuracy_sum{{variant="{variant}"}}[300s]) / '
             f'rate(detection_accuracy_count{{variant="{variant}"}}[300s])')
    else:
        q = "rate(detection_accuracy_sum[60s]) / rate(detection_accuracy_count[60s])"
    v = _prom(q)
    return round(v, 4) if v is not None else None


@st.cache_data(ttl=10, show_spinner=False)
def get_active_variant() -> str:
    if k8s_client:
        env = k8s_client.get_pod_env("detection") or []
        for item in env:
            if item.get("name") == "VARIANT_ID":
                return item.get("value", "unknown")
    return "unknown"


@st.cache_data(ttl=5, show_spinner=False)
def get_milp_weights() -> dict[str, Any]:
    """Read live MILP objective weights — priority: Redis > Deployment env > Pod env > defaults."""
    defaults = {
        "w_c": _safe_float(os.getenv("MILP_W_C", os.getenv("W_C", "0.20")), 0.20),
        "w_d": _safe_float(os.getenv("MILP_W_D", os.getenv("W_D", "0.10")), 0.10),
        "w_a": _safe_float(os.getenv("MILP_W_A", os.getenv("W_A", "0.70")), 0.70),
        "source": "defaults",
    }

    # 1. Redis milp:weights — highest priority (set by dashboard sliders or milp_agent startup).
    rdb = _get_redis_client()
    if rdb:
        try:
            raw = rdb.get("milp:weights")
            if raw:
                data = json.loads(raw)
                return {
                    "w_c": _safe_float(data.get("w_c"), defaults["w_c"]),
                    "w_d": _safe_float(data.get("w_d"), defaults["w_d"]),
                    "w_a": _safe_float(data.get("w_a"), defaults["w_a"]),
                    "source": "redis:milp:weights",
                }
        except Exception:
            pass

    # 2. K8s Deployment env — reflects changes as soon as manifest is applied.
    out = _kubectl(f"get deployment milp-agent -n {NAMESPACE} -o json", timeout=8)
    if out:
        try:
            payload = json.loads(out)
            containers = payload.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
            target = None
            for c in containers:
                if c.get("name") == "milp-agent":
                    target = c
                    break
            if target is None and containers:
                target = containers[0]

            env_map = {}
            for item in (target or {}).get("env", []):
                k = item.get("name")
                v = item.get("value")
                if k:
                    env_map[k] = v

            # Accept new MILP_W_* names first, fall back to legacy W_* names.
            def _pick(new_key, old_key, default):
                return _safe_float(env_map.get(new_key, env_map.get(old_key)), default)

            if any(k in env_map for k in ("MILP_W_C", "MILP_W_D", "MILP_W_A", "W_C", "W_D", "W_A")):
                return {
                    "w_c": _pick("MILP_W_C", "W_C", defaults["w_c"]),
                    "w_d": _pick("MILP_W_D", "W_D", defaults["w_d"]),
                    "w_a": _pick("MILP_W_A", "W_A", defaults["w_a"]),
                    "source": "deployment/milp-agent",
                }
        except Exception:
            pass

    # 3. Running pod env fallback.
    if k8s_client:
        try:
            env = k8s_client.get_pod_env("milp-agent") or []
            env_map = {item.get("name", ""): item.get("value", "") for item in env}
            if any(k in env_map for k in ("MILP_W_C", "MILP_W_D", "MILP_W_A", "W_C", "W_D", "W_A")):
                def _pick(new_key, old_key, default):
                    return _safe_float(env_map.get(new_key, env_map.get(old_key)), default)
                return {
                    "w_c": _pick("MILP_W_C", "W_C", defaults["w_c"]),
                    "w_d": _pick("MILP_W_D", "W_D", defaults["w_d"]),
                    "w_a": _pick("MILP_W_A", "W_A", defaults["w_a"]),
                    "source": "pod/milp-agent",
                }
        except Exception:
            pass

    return defaults


@st.cache_data(ttl=10, show_spinner=False)
def get_pod_node_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for svc in SERVICES:
        if k8s_client:
            n = k8s_client.get_pod_node(svc)
            result[svc] = n or "unknown"
        else:
            result[svc] = "unknown"
    return result


@st.cache_data(ttl=5, show_spinner=False)
def get_cluster_state_snapshot() -> dict[str, list[dict] | float]:
    """Read live nodes/deployments/pods from Kubernetes once per refresh window."""
    snapshot: dict[str, list[dict] | float] = {
        "timestamp": time.time(),
        "nodes": [],
        "deployments": [],
        "pods": [],
    }

    out_nodes = _kubectl("get nodes -o json", timeout=8)
    if out_nodes:
        try:
            payload = json.loads(out_nodes)
            for item in payload.get("items", []):
                labels = item.get("metadata", {}).get("labels", {})
                is_worker = (
                    "node-role.kubernetes.io/control-plane" not in labels
                    and "node-role.kubernetes.io/master" not in labels
                )
                conditions = item.get("status", {}).get("conditions", [])
                ready = any(
                    c.get("type") == "Ready" and c.get("status") == "True"
                    for c in conditions
                )
                snapshot["nodes"].append({
                    "name": item.get("metadata", {}).get("name", ""),
                    "ready": ready,
                    "is_worker": is_worker,
                    "unschedulable": bool(item.get("spec", {}).get("unschedulable", False)),
                })
        except Exception:
            pass

    out_deploy = _kubectl(f"get deployments -n {NAMESPACE} -o json", timeout=8)
    if out_deploy:
        try:
            payload = json.loads(out_deploy)
            for item in payload.get("items", []):
                status = item.get("status", {})
                spec = item.get("spec", {})
                snapshot["deployments"].append({
                    "name": item.get("metadata", {}).get("name", ""),
                    "ready": _safe_int(status.get("readyReplicas", 0)),
                    "desired": _safe_int(spec.get("replicas", 0)),
                    "available": _safe_int(status.get("availableReplicas", 0)),
                    "updated": _safe_int(status.get("updatedReplicas", 0)),
                })
        except Exception:
            pass

    out_pods = _kubectl(f"get pods -n {NAMESPACE} -o json", timeout=8)
    if out_pods:
        try:
            payload = json.loads(out_pods)
            for item in payload.get("items", []):
                cstats = item.get("status", {}).get("containerStatuses", []) or []
                ready_cnt = sum(1 for c in cstats if c.get("ready"))
                total_cnt = len(cstats)
                snapshot["pods"].append({
                    "name": item.get("metadata", {}).get("name", ""),
                    "phase": item.get("status", {}).get("phase", "Unknown"),
                    "node": item.get("spec", {}).get("nodeName", ""),
                    "ip": item.get("status", {}).get("podIP", ""),
                    "app": item.get("metadata", {}).get("labels", {}).get("app", ""),
                    "ready": f"{ready_cnt}/{total_cnt}",
                    "restarts": sum(_safe_int(c.get("restartCount", 0)) for c in cstats),
                })
        except Exception:
            pass

    snapshot["nodes"] = sorted(snapshot["nodes"], key=lambda n: n.get("name", ""))
    snapshot["deployments"] = sorted(snapshot["deployments"], key=lambda d: d.get("name", ""))
    snapshot["pods"] = sorted(snapshot["pods"], key=lambda p: p.get("name", ""))
    return snapshot


@st.cache_data(ttl=8, show_spinner=False)
def get_cluster_worker_nodes() -> list[str]:
    snapshot = get_cluster_state_snapshot()
    workers = [
        n.get("name", "")
        for n in snapshot.get("nodes", [])
        if n.get("is_worker") and not n.get("unschedulable")
    ]
    workers = [w for w in workers if w]
    if workers:
        return workers

    if k8s_client:
        try:
            fallback = k8s_client.get_all_worker_nodes() or []
            if fallback:
                return sorted(fallback)
        except Exception:
            pass

    return NODES


@st.cache_data(ttl=8, show_spinner=False)
def get_node_pod_resource_totals() -> dict[str, dict[str, Any]]:
    """Aggregate Kubernetes requests/limits for all pods on each node."""
    totals: dict[str, dict[str, Any]] = {}

    def _ensure(node: str) -> dict[str, Any]:
        if node not in totals:
            totals[node] = {
                "cpu_request": 0.0,
                "cpu_limit": 0.0,
                "mem_request_gib": 0.0,
                "mem_limit_gib": 0.0,
                "pods": [],
            }
        return totals[node]

    raw = _kubectl("get pods -A -o json", timeout=8)
    if not raw:
        return totals
    try:
        payload = json.loads(raw)
    except Exception:
        return totals

    for item in payload.get("items", []):
        spec = item.get("spec", {}) or {}
        status = item.get("status", {}) or {}
        node = spec.get("nodeName") or ""
        if not node:
            continue

        cpu_req = cpu_lim = mem_req = mem_lim = 0.0
        for container in spec.get("containers", []) or []:
            resources = container.get("resources", {}) or {}
            requests = resources.get("requests", {}) or {}
            limits = resources.get("limits", {}) or {}
            cpu_req += _parse_cpu_cores(requests.get("cpu"))
            cpu_lim += _parse_cpu_cores(limits.get("cpu"))
            mem_req += _parse_memory_gib(requests.get("memory"))
            mem_lim += _parse_memory_gib(limits.get("memory"))

        bucket = _ensure(node)
        bucket["cpu_request"] += cpu_req
        bucket["cpu_limit"] += cpu_lim
        bucket["mem_request_gib"] += mem_req
        bucket["mem_limit_gib"] += mem_lim
        bucket["pods"].append({
            "namespace": item.get("metadata", {}).get("namespace", ""),
            "pod": item.get("metadata", {}).get("name", ""),
            "phase": status.get("phase", ""),
            "cpu_request": cpu_req,
            "cpu_limit": cpu_lim,
            "mem_request_gib": mem_req,
            "mem_limit_gib": mem_lim,
        })

    for node_data in totals.values():
        node_data["pods"].sort(
            key=lambda pod: (
                float(pod.get("cpu_limit") or 0.0),
                float(pod.get("cpu_request") or 0.0),
                str(pod.get("pod") or ""),
            ),
            reverse=True,
        )
    return totals


@st.cache_data(ttl=5, show_spinner=False)
def get_deployment_replica_map() -> dict[str, str]:
    replica_map: dict[str, str] = {}
    for dep in get_cluster_state_snapshot().get("deployments", []):
        name = dep.get("name")
        if not name:
            continue
        replica_map[name] = f"{dep.get('ready', 0)}/{dep.get('desired', 0)}"
    return replica_map


@st.cache_data(ttl=60, show_spinner=False)
def get_recent_logs(deploy: str, lines: int = 20) -> str:
    out = _kubectl(f"logs deployment/{deploy} -n {NAMESPACE} --tail={lines}", timeout=8)
    return out or "(No logs available)"


# ── DRL / Digital-Twin Redis readers ─────────────────────────────────────────

@st.cache_data(ttl=4, show_spinner=False)
def get_system_mode_details() -> tuple[str, str]:
    """Read system:mode from Redis with a safe default when key is missing."""
    rdb = _get_redis_client()
    if rdb:
        try:
            v = rdb.get("system:mode")
            if v:
                mode = v.strip().lower()
                if mode in {"milp", "shadow", "drl", "hybrid"}:
                    return mode, "redis"
                return "unknown", "redis (invalid value)"
            return "milp", "default (redis empty)"
        except Exception:
            return "unknown", "redis error"
    return "unknown", "redis unavailable"


def get_system_mode() -> str:
    """Read system:mode from Redis (milp | shadow | drl | hybrid)."""
    return get_system_mode_details()[0]


@st.cache_data(ttl=4, show_spinner=False)
def get_drl_placement() -> Optional[dict]:
    """Read drl:placement from Redis.

    Falls back to drl:placement:proposed when drl:placement is absent
    (e.g. system running in milp mode where the leader-gated write is
    skipped but the DRL agent still records its proposed decision).
    """
    rdb = _get_redis_client()
    if not rdb:
        return None
    try:
        raw = rdb.get("drl:placement")
        if raw:
            live = json.loads(raw)
            if isinstance(live, dict):
                live["_source"] = "live"
            return live
        proposed_raw = rdb.get("drl:placement:proposed")
        if proposed_raw:
            proposed = json.loads(proposed_raw)
            if isinstance(proposed.get("placement"), dict):
                return {
                    "placement": proposed["placement"],
                    "_source": "proposed",
                    "_accepted": bool(proposed.get("accepted", False)),
                    "_twin_stats": proposed.get("twin_stats", {}),
                }
    except Exception:
        pass
    return None


@st.cache_data(ttl=5, show_spinner=False)
def get_tier2_status() -> dict:
    """Read Tier-2 verifier verdicts and mark staleness against current proposal."""
    rdb = _get_redis_client()
    out = {
        "verified": None,
        "revoked": None,
        "source": "none",
        "verified_source": None,
        "revoked_source": None,
        "current_fingerprint": None,
        "heartbeat": None,
        "heartbeat_age_s": None,
        "verified_age_s": None,
        "revoked_age_s": None,
        "verified_stale": False,
        "revoked_stale": False,
    }
    if not rdb:
        return out
    try:
        proposed_raw = rdb.get("drl:placement:proposed")
        if proposed_raw:
            out["current_fingerprint"] = _tier2_fingerprint(proposed_raw)
    except Exception:
        pass
    try:
        hb = rdb.get("drl:placement:verifier_heartbeat")
        if hb:
            out["heartbeat"] = json.loads(hb)
            out["heartbeat_age_s"] = _age_seconds(out["heartbeat"].get("timestamp"))
    except Exception:
        pass
    try:
        v = rdb.get("drl:placement:verified")
        if v:
            out["verified"] = json.loads(v)
            out["verified_source"] = "recent"
            out["source"] = "recent"
        elif (vl := rdb.get("drl:placement:verified:latest")):
            out["verified"] = json.loads(vl)
            out["verified_source"] = "latest"
            out["source"] = "latest"
    except Exception:
        pass
    try:
        r = rdb.get("drl:placement:revoked")
        if r:
            out["revoked"] = json.loads(r)
            out["revoked_source"] = "recent"
            out["source"] = "recent"
        elif (rl := rdb.get("drl:placement:revoked:latest")):
            out["revoked"] = json.loads(rl)
            out["revoked_source"] = "latest"
            out["source"] = "latest"
    except Exception:
        pass
    if out["verified"]:
        age = _age_seconds(out["verified"].get("timestamp"))
        out["verified_age_s"] = age
        out["verified_stale"] = _tier2_verdict_stale(
            out["verified"],
            age,
            out["current_fingerprint"],
            out["heartbeat"],
            out["heartbeat_age_s"],
        )
    if out["revoked"]:
        age = _age_seconds(out["revoked"].get("timestamp"))
        out["revoked_age_s"] = age
        out["revoked_stale"] = _tier2_verdict_stale(
            out["revoked"],
            age,
            out["current_fingerprint"],
            out["heartbeat"],
            out["heartbeat_age_s"],
        )
    _tier2_drop_superseded_verdict(out)
    return out


@st.cache_data(ttl=5, show_spinner=False)
def get_twin_stats() -> Optional[dict]:
    """Read drl:twin_stats (TTL 35 s) from Redis."""
    rdb = _get_redis_client()
    if rdb:
        try:
            raw = rdb.get("drl:twin_stats")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return None


@st.cache_data(ttl=10, show_spinner=False)
def get_expert_traj_count() -> int:
    """Return LLEN of milp:expert_trajectories."""
    rdb = _get_redis_client()
    if rdb:
        try:
            return int(rdb.llen("milp:expert_trajectories"))
        except Exception:
            pass
    return 0


@st.cache_data(ttl=8, show_spinner=False)
def get_simulator_tail(max_lines: int = 400) -> list[dict[str, Any]]:
    """Read recent simulator JSONL rows from disk for pipeline observability."""
    path = Path(SIM_RESULTS_FILE)
    if not path.exists():
        return []

    rows: deque[str] = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(line)
    except Exception:
        return []

    parsed: list[dict[str, Any]] = []
    for line in rows:
        try:
            parsed.append(json.loads(line))
        except Exception:
            continue
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gauge(value: float, max_val: float, label: str, unit: str = "%",
           warn: float = 0.70, crit: float = 0.90,
           height: int = 180, invert: bool = False) -> go.Figure:
    """Gauge indicator.

    invert=False (default): green when low, red when high (e.g. CPU, latency).
    invert=True:            green when high, red when low (e.g. safe ratio).
    """
    pct = value / max_val if max_val else 0.0
    if invert:
        color = "#4CAF50" if pct >= warn else ("#FF9800" if pct >= crit else "#F44336")
        steps = [
            {"range": [0, max_val * crit],          "color": "#2a1010"},
            {"range": [max_val * crit, max_val * warn], "color": "#2a2010"},
            {"range": [max_val * warn, max_val],     "color": "#1e2a1e"},
        ]
        threshold_val = max_val * crit
    else:
        color = "#4CAF50" if pct < warn else ("#FF9800" if pct < crit else "#F44336")
        steps = [
            {"range": [0, max_val * warn],              "color": "#1e2a1e"},
            {"range": [max_val * warn, max_val * crit], "color": "#2a2010"},
            {"range": [max_val * crit, max_val],        "color": "#2a1010"},
        ]
        threshold_val = max_val * crit
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": unit, "font": {"size": 20}},
        title={"text": label, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, max_val], "tickfont": {"size": 10}},
            "bar":  {"color": color},
            "steps": steps,
            "threshold": {
                "line": {"color": "#F44336", "width": 3},
                "thickness": 0.75,
                "value": threshold_val,
            },
        },
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
    )
    return fig


def _bar_usage(used: float, cap: float, label: str, unit: str,
               color: str = "#4C9BE8", height: int = 80) -> go.Figure:
    pct = min(used / cap * 100, 100) if cap else 0
    bar_color = color if pct < 70 else ("#FF9800" if pct < 90 else "#F44336")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[pct], y=[label], orientation="h",
        marker_color=bar_color, width=0.6,
        text=[f"{used:.2f} / {cap:.2f} {unit} ({pct:.1f}%)"],
        textposition="inside", insidetextanchor="start",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False),
        yaxis=dict(showticklabels=False),
        height=height, margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.6)",
        font_color="#e0e0e0",
    )
    return fig


def _migration_badge(mtype: str) -> str:
    bg = MIGRATION_COLORS.get(mtype, "#607D8B")
    return (f'<span class="badge" style="background:{bg}">{mtype}</span>')


def _status_dot(ok: bool) -> str:
    return "🟢" if ok else "🔴"


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard sections
# ─────────────────────────────────────────────────────────────────────────────

def render_header(placement: Optional[dict]) -> None:
    """Top KPI row: system mode + MILP + DRL key metrics."""
    # ── System mode banner ────────────────────────────────────────────────────
    mode, mode_source = get_system_mode_details()
    mode_labels = {
        "milp":    ("MILP MODE — MILP governs all placement",       "mode-milp"),
        "shadow":  ("SHADOW MODE — DRL runs silently, MILP governs", "mode-shadow"),
        "drl":     ("DRL MODE — DRL governs, MILP is fallback",      "mode-drl"),
        "hybrid":  ("HYBRID MODE — DRL gated by MILP safety",        "mode-hybrid"),
        "unknown": ("MODE UNKNOWN — check Redis system:mode",        "mode-unknown"),
    }
    label, css_cls = mode_labels.get(mode, mode_labels["unknown"])

    ts_raw = placement.get("timestamp", None) if placement else None
    if ts_raw:
        ts_time = _fmt_ts(ts_raw, "%H:%M:%S")
        ts_date = _fmt_ts(ts_raw, "%Y-%m-%d")
        last_solve = f"Last solve: {ts_time} · {ts_date} GMT+7"
    else:
        last_solve = "Last solve: —"

    st.markdown(
        f'<div style="margin-bottom:10px; display:flex; gap:10px; align-items:center;">'
        f'<span class="{css_cls}">⚙ {label}</span>'
        f'<span style="color:#a0aec0; font-size:0.85rem;">{last_solve} · source: {mode_source}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    # MILP objective
    with col1:
        obj = placement.get("objective", None) if placement else None
        status = placement.get("status", "—") if placement else "—"
        st.metric("🎯 MILP J", f"{obj:.4f}" if obj is not None else "N/A",
                  delta=f"status: {status}", delta_color="off")

    # E2E latency
    with col2:
        lat = get_e2e_latency()
        delta_str = f"SLA: {SLA_LATENCY_MS:.0f} ms"
        if lat is not None:
            if lat > SLA_LATENCY_MS:
                delta_str = f"⚠️ SLA VIOLATED"
            st.metric("🌐 E2E Latency", f"{lat:.1f} ms", delta=delta_str,
                      delta_color="inverse" if lat > SLA_LATENCY_MS else "off")
        else:
            st.metric("🌐 E2E Latency", "N/A", delta=delta_str, delta_color="off")

    # Detection accuracy
    with col3:
        active_var = get_active_variant()
        acc = get_detection_accuracy(active_var if active_var != "unknown" else None)
        acc_pct = f"{acc*100:.1f}%" if acc is not None else "N/A"
        st.metric("🎯 Det. Accuracy", acc_pct, delta=f"Variant: {active_var}",
                  delta_color="off")

    # Migrations this cycle
    with col4:
        mig_types = placement.get("migration_types", {}) if placement else {}
        n_mig = sum(1 for v in mig_types.values() if v != "Stayed")
        st.metric("🔀 Migrations", str(n_mig), delta="this cycle", delta_color="off")

    with st.expander("Header details", expanded=False):
        d1, d2, d3 = st.columns(3)
        with d1:
            st_val = placement.get("solve_time", None) if placement else None
            st.metric("⏱ Solve Time", f"{st_val:.3f}s" if st_val is not None else "—",
                      delta="Target < 5s", delta_color="off")
        with d2:
            status = placement.get("status", "—") if placement else "—"
            icon = "🟢" if status == "optimal" else ("🟡" if status == "feasible" else "🔴")
            st.metric("🔄 Solver", status, delta=icon, delta_color="off")
        with d3:
            drl_pl = get_drl_placement() or {}
            drl_j = drl_pl.get("comparable_objective", drl_pl.get("objective"))
            milp_j = (placement.get("objective") if placement else None)
            if drl_j is not None and milp_j is not None and milp_j != 0:
                gap_pct = 100 * (float(drl_j) - float(milp_j)) / abs(float(milp_j))
                delta = f"{'+' if gap_pct >= 0 else ''}{gap_pct:.1f}% vs MILP"
            else:
                delta = "comparable"
            st.metric("🤖 DRL Comparable J", f"{float(drl_j):.4f}" if drl_j is not None else "N/A",
                      delta=delta, delta_color="off")


def render_health_bar() -> None:
    """Compact connectivity status bar."""
    st.markdown('<p class="section-header">🔌 System Connectivity</p>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        rdb = _get_redis_client()
        host = st.session_state.get("redis_host_ok", "")
        if rdb:
            st.success(f"✅ Redis `{host}:{REDIS_PORT}`")
        else:
            st.error("❌ Redis disconnected")
            st.caption("kubectl port-forward svc/redis 6379:6379")

    with c2:
        prom_ok = _prom("up") is not None or len(_prom_vector("up")) > 0
        if prom_ok:
            st.success(f"✅ Prometheus `{PROM_URL.split('/api')[0].split('/')[-1]}`")
        else:
            st.error("❌ Prometheus unreachable")

    with c3:
        if k8s_client is None:
            st.error(f"❌ k8s SDK: {K8S_IMPORT_ERROR[:50]}")
        else:
            st.success("✅ Kubernetes SDK")

    with c4:
        kctl = _kubectl("version --client", timeout=3)
        if kctl:
            st.success("✅ kubectl")
        else:
            st.warning("⚠️ kubectl unavailable")


def render_cluster_state_summary() -> None:
    """Compact live counts sourced directly from the Kubernetes API."""
    st.markdown('<p class="section-header">🧭 Live Cluster State</p>',
                unsafe_allow_html=True)

    snapshot = get_cluster_state_snapshot()
    worker_nodes = [n for n in snapshot.get("nodes", []) if n.get("is_worker")]
    ready_workers = sum(1 for n in worker_nodes if n.get("ready"))
    total_workers = len(worker_nodes)

    deployments = snapshot.get("deployments", [])
    dep_ready = sum(dep.get("ready", 0) for dep in deployments)
    dep_desired = sum(dep.get("desired", 0) for dep in deployments)

    pods = snapshot.get("pods", [])
    pod_total = len(pods)
    pod_running = sum(1 for p in pods if p.get("phase") == "Running")
    pod_pending = sum(1 for p in pods if p.get("phase") in {"Pending", "Unknown"})
    pod_failed = sum(1 for p in pods if p.get("phase") in {"Failed", "CrashLoopBackOff"})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🖥 Worker Nodes", f"{ready_workers}/{total_workers}",
                  delta="Ready/Total")
    with c2:
        st.metric("📦 Deployments", f"{dep_ready}/{dep_desired}",
                  delta="Ready/Desired replicas")
    with c3:
        st.metric("🧩 Pods", f"{pod_running}/{pod_total}",
                  delta="Running/Total")
    with c4:
        st.metric("⚠️ Non-Running Pods", str(pod_pending + pod_failed),
                  delta=f"Pending:{pod_pending} Failed:{pod_failed}")

    synced_at = _fmt_ts(snapshot.get("timestamp", time.time()), "%H:%M:%S (GMT+7)")
    st.caption(f"Synced from cluster API at {synced_at} (namespace={NAMESPACE})")


def render_node_resources() -> None:
    st.markdown('<p class="section-header">🖥️ Node Resource Utilisation & Capacity</p>',
                unsafe_allow_html=True)

    nodes = get_cluster_worker_nodes()
    for node in nodes:
        cap_cpu   = get_node_capacity_cpu(node)
        cap_mem   = get_node_capacity_mem(node)
        cpu_util  = get_node_cpu_util(node)
        mem_used  = get_node_mem_used_gb(node)
        kepler_w  = get_kepler_node_watts(node)
        dram_w    = get_kepler_dram_watts(node)

        cpu_used_val = round((cpu_util or 0.0) * cap_cpu, 3)
        cpu_free_val = max(round(cap_cpu - cpu_used_val, 3), 0.0)
        cpu_pct      = min(100.0, (cpu_util or 0.0) * 100.0)

        mem_used_val = mem_used or 0.0
        mem_free_val = max(round(cap_mem - mem_used_val, 3), 0.0)
        mem_pct      = min(100.0, (mem_used_val / cap_mem * 100.0) if cap_mem else 0.0)

        node_color = NODE_COLORS.get(node, "#4C9BE8")
        cpu_color  = "#F44336" if cpu_pct > 85 else ("#FF9800" if cpu_pct > 70 else node_color)
        mem_color  = "#F44336" if mem_pct > 85 else ("#FF9800" if mem_pct > 70 else "#58C4A0")

        # Node header (safe single-line HTML, no conditional logic inside)
        st.markdown(
            '<div style="background:#1a2035; border-left:4px solid ' + node_color + '; '
            'border-radius:8px; padding:10px 18px; margin-bottom:8px; display:flex; '
            'justify-content:space-between; align-items:center;">' 
            '<b style="color:' + node_color + '; font-size:1.1rem;">⬡ ' + node + '</b>'
            '<span style="color:#a0aec0; font-size:0.82rem;">Capacity: '
            '<b style="color:#fff">' + str(int(cap_cpu)) + ' cores</b>&nbsp;|&nbsp;'
            '<b style="color:#fff">' + f"{cap_mem:.1f}" + ' GB</b> RAM</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 2])

        # Left: KPI metric tiles
        with left:
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.metric(
                    "CPU Utilization",
                    f"{cpu_pct:.1f}%",
                    delta=f"{cpu_used_val:.2f} / {cap_cpu:.0f} cores",
                    delta_color="off",
                )
            with r1c2:
                st.metric(
                    "RAM Used",
                    f"{mem_pct:.1f}%",
                    delta=f"{mem_used_val:.2f} / {cap_mem:.1f} GB",
                    delta_color="off",
                )
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.metric(
                    "Platform W",
                    f"{kepler_w:.1f} W" if kepler_w is not None else "N/A",
                    delta="CPU+Uncore" if kepler_w is not None else None,
                    delta_color="off",
                )
            with r2c2:
                st.metric(
                    "DRAM W",
                    f"{dram_w:.1f} W" if dram_w is not None else "N/A",
                    delta="Memory power" if dram_w is not None else None,
                    delta_color="off",
                )

        # Right: horizontal stacked bar — Used vs Free for CPU and RAM
        with right:
            fig = go.Figure()

            # ── CPU row ──
            fig.add_trace(go.Bar(
                name="CPU Used",
                y=["CPU (cores)"],
                x=[cpu_used_val],
                orientation="h",
                marker_color=cpu_color,
                text=[f"{cpu_used_val:.2f} cores ({cpu_pct:.1f}%)"],
                textposition="inside",
                insidetextanchor="start",
                textfont=dict(color="white", size=11, family="monospace"),
                width=0.5,
                legendgroup="used",
            ))
            fig.add_trace(go.Bar(
                name="CPU Free",
                y=["CPU (cores)"],
                x=[cpu_free_val],
                orientation="h",
                marker=dict(
                    color="rgba(76,155,232,0.18)",
                    line=dict(color="#4C9BE8", width=1),
                ),
                text=[f"{cpu_free_val:.2f} free"],
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(color="#7090b0", size=10),
                width=0.5,
                legendgroup="free",
            ))

            # ── RAM row ──
            fig.add_trace(go.Bar(
                name="RAM Used",
                y=["RAM (GB)"],
                x=[mem_used_val],
                orientation="h",
                marker_color=mem_color,
                text=[f"{mem_used_val:.2f} GB ({mem_pct:.1f}%)"],
                textposition="inside",
                insidetextanchor="start",
                textfont=dict(color="white", size=11, family="monospace"),
                width=0.5,
                legendgroup="used",
                showlegend=False,
            ))
            fig.add_trace(go.Bar(
                name="RAM Free",
                y=["RAM (GB)"],
                x=[mem_free_val],
                orientation="h",
                marker=dict(
                    color="rgba(88,196,160,0.18)",
                    line=dict(color="#58C4A0", width=1),
                ),
                text=[f"{mem_free_val:.2f} free"],
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(color="#4a9070", size=10),
                width=0.5,
                legendgroup="free",
                showlegend=False,
            ))

            # Dotted capacity marker lines
            fig.add_vline(
                x=cap_cpu,
                line=dict(color=node_color, width=2, dash="dot"),
                annotation_text=f"Max {int(cap_cpu)}c",
                annotation_font=dict(color=node_color, size=10),
                annotation_position="top",
            )
            fig.add_vline(
                x=cap_mem,
                line=dict(color="#58C4A0", width=2, dash="dot"),
                annotation_text=f"Max {cap_mem:.1f}GB",
                annotation_font=dict(color="#58C4A0", size=10),
                annotation_position="bottom",
            )

            fig.update_layout(
                barmode="stack",
                height=155,
                margin=dict(l=5, r=30, t=25, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,14,28,0.6)",
                font_color="#e0e0e0",
                legend=dict(
                    orientation="h", y=1.35, x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.06)",
                    zeroline=False,
                    tickfont=dict(size=10),
                    # x-axis upper bound = max of CPU cap or mem cap (they're different units
                    # but on one shared axis — so we scale CPU bar to its own cap mark)
                ),
                yaxis=dict(
                    tickfont=dict(size=12, family="monospace"),
                ),
                bargap=0.4,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"node_res_bar_{node}", config={"displayModeBar": False})

        st.divider()



def render_placement_table(placement: Optional[dict], confirmed: Optional[dict], pod_node_map: dict) -> None:
    """Current MILP placement with confirmed and live K8s location."""
    st.markdown('<p class="section-header">📍 Service Placement</p>',
                unsafe_allow_html=True)

    if not placement:
        st.warning("⚠️ No MILP placement data in Redis")
        return

    pd_rows = []
    pl = placement.get("placement", {})
    mt = placement.get("migration_types", {})
    replica_map = get_deployment_replica_map()
    confirmed = confirmed or {}

    for svc in SERVICES:
        info   = pl.get(svc, {})
        mig    = mt.get(svc, "—")
        milp_node = info.get("node", "?")
        variant   = info.get("variant", "standard")
        confirmed_node = confirmed.get(svc, {}).get("node", "unknown")
        live_node = pod_node_map.get(svc, "unknown")
        if milp_node == confirmed_node == live_node:
            match = "✅ in-sync"
        elif milp_node == live_node and confirmed_node != live_node:
            match = "⚠️ confirmed lag"
        elif milp_node == confirmed_node and live_node != confirmed_node:
            match = "⚠️ pod lag"
        else:
            match = "⚠️ mismatch"

        pd_rows.append({
            "Service":      svc,
            "MILP Node":    milp_node,
            "Confirmed Node": confirmed_node,
            "Live K8s Node":live_node,
            "Replicas":     replica_map.get(svc, "N/A"),
            "Sync":         match,
            "Variant":      variant,
            "Migration":    mig,
        })

    # Add emoji indicator for migration type (no jinja2 needed)
    def _mig_icon(val: str) -> str:
        icons = {
            "Stayed": "✅ Stayed",
            "Node Migration": "🔄 Node Mig.",
            "AI Model Redeployment": "🧠 AI Redeploy",
            "Node Migration + AI Redeployment": "🔄🧠 Node+AI",
        }
        return icons.get(val, val)

    df_display = pd.DataFrame(pd_rows)
    df_display["Migration"] = df_display["Migration"].apply(_mig_icon)
    st.dataframe(df_display, width="stretch", hide_index=True)


def render_objective_panel(placement: Optional[dict]) -> None:
    """MILP objective decomposition: energy, disruption, accuracy."""
    st.markdown('<p class="section-header">📐 Objective Decomposition  J = w_c·C_norm + w_d·D_norm − w_a·A_norm</p>',
                unsafe_allow_html=True)

    if not placement:
        st.info("No placement data.")
        return

    c_norm = placement.get("norm_cost_energy", 0)
    d_norm = placement.get("norm_cost_disruption", 0)
    a_norm = placement.get("norm_gain_accuracy", 0)
    c_raw  = placement.get("cost_energy", 0)
    d_raw  = placement.get("cost_disruption", 0)
    a_raw  = placement.get("gain_accuracy", 0)
    obj    = placement.get("objective", 0)
    weights = get_milp_weights()
    w_c = _safe_float(weights.get("w_c"), 0.15)
    w_d = _safe_float(weights.get("w_d"), 0.10)
    w_a = _safe_float(weights.get("w_a"), 0.75)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⚡ C_norm (Energy)", f"{c_norm:.4f}",
                  delta=f"Raw C = {c_raw:.2f} W",
                  help=f"Normalized energy cost. Live w_c = {w_c:.4f}.")
    with col2:
        st.metric("🔄 D_norm (Disruption)", f"{d_norm:.4f}",
                  delta=f"Raw D = {d_raw:.2f} s",
                  help="Normalized migration disruption cost.")
    with col3:
        st.metric("🎯 A_norm (Accuracy)", f"{a_norm:.4f}",
                  delta=f"Raw A = {a_raw:.4f}",
                  help="Normalized accuracy gain (higher = better).")
    with col4:
        st.metric("🏆 Objective J", f"{obj:.4f}",
                  delta="↓ lower is better",
                  help="Final MILP objective value. Range [-1, 1].")

    st.caption(
        f"Live weights: w_c={w_c:.4f}, w_d={w_d:.4f}, w_a={w_a:.4f} "
        f"(source: {weights.get('source', 'unknown')})"
    )

    # Stacked bar decomposition
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="w_c · C_norm (Energy)",
        x=["Objective J"], y=[w_c * c_norm],
        marker_color="#F44336", text=[f"{w_c*c_norm:.4f}"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="w_d · D_norm (Disruption)",
        x=["Objective J"], y=[w_d * d_norm],
        marker_color="#FF9800", text=[f"{w_d*d_norm:.4f}"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="-w_a · A_norm (Accuracy gain, minimised)",
        x=["Objective J"], y=[-w_a * a_norm],
        marker_color="#4CAF50", text=[f"-{w_a*a_norm:.4f}"],
        textposition="inside",
    ))
    fig.update_layout(
        barmode="relative", height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", legend=dict(orientation="h", y=1.1),
        yaxis_title="Contribution to J",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Weight pie
    lc, rc = st.columns(2)
    with lc:
        pie_values = [max(w_c, 0.0), max(w_d, 0.0), max(w_a, 0.0)]
        if sum(pie_values) <= 0:
            pie_values = [0.15, 0.10, 0.75]
        fig2 = go.Figure(go.Pie(
            labels=[
                f"Energy (w_c={w_c:.3f})",
                f"Disruption (w_d={w_d:.3f})",
                f"Accuracy (w_a={w_a:.3f})",
            ],
            values=pie_values,
            marker=dict(colors=["#F44336", "#FF9800", "#4CAF50"]),
            hole=0.45,
        ))
        fig2.update_layout(
            height=220, margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Weight Distribution", font=dict(size=13)),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
        )
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    with rc:
        node_scores = placement.get("node_scores", {})
        if node_scores:
            fig3 = go.Figure(go.Bar(
                x=list(node_scores.keys()),
                y=list(node_scores.values()),
                marker_color=[NODE_COLORS.get(n, "#607D8B") for n in node_scores],
                text=[str(v) for v in node_scores.values()],
                textposition="outside",
            ))
            fig3.update_layout(
                height=220, margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Node Scheduler Scores", font=dict(size=13)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", yaxis_range=[0, 110],
            )
            st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})


def render_energy_panel() -> None:
    """Live per-node energy panel: platform watts, DRAM watts, E_unit."""
    st.markdown('<p class="section-header">⚡ Energy Breakdown (Kepler)</p>',
                unsafe_allow_html=True)

    rows = []
    nodes = get_cluster_worker_nodes()
    for node in nodes:
        cap_cpu = get_node_capacity_cpu(node)
        cap_mem = get_node_capacity_mem(node)
        plat_w  = get_kepler_node_watts(node)
        dram_w  = get_kepler_dram_watts(node)
        e_cpu   = round(plat_w / cap_cpu, 4) if plat_w and cap_cpu else None
        e_mem   = round(dram_w / cap_mem, 4) if dram_w and cap_mem else None

        rows.append({
            "Node":        node,
            "Platform W":  f"{plat_w:.2f}" if plat_w else "N/A",
            "DRAM W":      f"{dram_w:.2f}" if dram_w else "N/A",
            "E_cpu (W/core)": f"{e_cpu:.4f}" if e_cpu else "N/A",
            "E_mem (W/GB)":   f"{e_mem:.4f}" if e_mem else "N/A",
            "CPU cap (cores)": f"{cap_cpu:.0f}",
            "RAM cap (GB)":    f"{cap_mem:.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    # Stacked energy bar: CPU fraction vs DRAM fraction per node
    plat_vals  = [get_kepler_node_watts(n) or 0 for n in nodes]
    dram_vals  = [get_kepler_dram_watts(n) or 0 for n in nodes]
    cpu_only   = [max(p - d, 0) for p, d in zip(plat_vals, dram_vals)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="CPU+Uncore W", x=nodes, y=cpu_only,
        marker_color="#4C9BE8",
    ))
    fig.add_trace(go.Bar(
        name="DRAM W", x=nodes, y=dram_vals,
        marker_color="#E8854C",
    ))
    fig.update_layout(
        barmode="stack", height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Node Power Consumption (W)", font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", legend=dict(orientation="h", y=1.15),
        yaxis_title="Watts",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_container_resources() -> None:
    """Per-service container CPU cores and RAM usage."""
    st.markdown('<p class="section-header">📦 Container Resource Usage (live cAdvisor)</p>',
                unsafe_allow_html=True)

    rows = []
    for svc in SERVICES:
        cpu = get_container_cpu_cores(svc)
        mem = get_container_mem_gb(svc)
        rows.append({
            "Service": svc,
            "CPU (cores)": f"{cpu:.4f}" if cpu is not None else "N/A",
            "RAM (GB)":    f"{mem:.3f}" if mem is not None else "N/A",
        })

    df = pd.DataFrame(rows)
    lcol, rcol = st.columns(2)
    with lcol:
        st.dataframe(df, width="stretch", hide_index=True)

    with rcol:
        cpu_vals = [get_container_cpu_cores(s) or 0 for s in SERVICES]
        mem_vals = [get_container_mem_gb(s) or 0 for s in SERVICES]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("CPU (cores)", "RAM (GB)"))
        colors = [SVC_COLORS.get(s, "#607D8B") for s in SERVICES]
        fig.add_trace(go.Bar(x=SERVICES, y=cpu_vals,
                             marker_color=colors, showlegend=False),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=SERVICES, y=mem_vals,
                             marker_color=colors, showlegend=False),
                      row=1, col=2)
        fig.update_layout(
            height=240, margin=dict(l=5, r=5, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
        )
        fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_detection_qos() -> None:
    """Detection accuracy (live), all variants strip, E2E latency vs SLA."""
    st.markdown('<p class="section-header">🎯 Detection QoS</p>',
                unsafe_allow_html=True)

    active_var = get_active_variant()
    lat        = get_e2e_latency()
    acc_live   = get_detection_accuracy(active_var if active_var != "unknown" else None)

    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        st.metric("🤖 Active Variant", active_var)
    with qcol2:
        acc_str = f"{acc_live*100:.2f}%" if acc_live else "N/A"
        st.metric("📊 Live Accuracy", acc_str)
    with qcol3:
        if lat is not None:
            color_lat = "normal" if lat <= SLA_LATENCY_MS else "inverse"
            sla_str = f"SLA {SLA_LATENCY_MS:.0f}ms {'✅' if lat <= SLA_LATENCY_MS else '⚠️ VIOLATED'}"
            st.metric("🌐 E2E Latency", f"{lat:.1f} ms", delta=sla_str)
        else:
            st.metric("🌐 E2E Latency", "N/A")

    # Offline accuracy reference bar for all variants
    from variant_catalog import DETECTION_OFFLINE_ACCURACY
    variants = list(DETECTION_OFFLINE_ACCURACY.keys())
    offline  = list(DETECTION_OFFLINE_ACCURACY.values())
    v_colors = [VARIANT_COLORS.get(v, "#607D8B") for v in variants]

    # Overlay live accuracy if known
    live_vals = []
    for v in variants:
        lv = get_detection_accuracy(v)
        live_vals.append(lv if lv is not None else None)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Offline mAP", x=variants, y=offline,
        marker_color=v_colors, opacity=0.55,
        text=[f"{v*100:.0f}%" for v in offline], textposition="outside",
    ))

    # Mark active variant with a highlighted bar outline
    marker_colors_outline = ["#FFD700" if v == active_var else "rgba(0,0,0,0)"
                             for v in variants]
    fig.add_trace(go.Bar(
        name="Active Variant", x=variants,
        y=[max(offline) * 1.05] * len(variants),   # full-height invisible bar
        marker_color="rgba(0,0,0,0)",
        marker_line_color=marker_colors_outline,
        marker_line_width=3,
        showlegend=False, hoverinfo="skip",
    ))

    # Live accuracy scatter overlay
    live_text = []
    for iv, v in enumerate(live_vals):
        tag = " ← active" if variants[iv] == active_var else ""
        live_text.append(f"{v*100:.1f}%{tag}" if v else "N/A")

    fig.add_trace(go.Scatter(
        name="Live Accuracy", x=variants,
        y=[v if v is not None else 0 for v in live_vals],
        mode="markers+text",
        marker=dict(size=12, color="#FFFFFF", symbol="diamond",
                    line=dict(color="#FFD700", width=2)),
        text=live_text,
        textposition="top center", textfont=dict(color="#FFD700"),
    ))

    fig.update_layout(
        height=280, barmode="overlay",
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Variant Accuracy: Offline mAP vs Live", font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", legend=dict(orientation="h", y=1.15),
        yaxis=dict(range=[0, 1.2], tickformat=".0%"),
    )
    st.plotly_chart(fig, width="stretch", key="qos_accuracy_chart",
                    config={"displayModeBar": False})

    # Latency vs SLA gauge
    if lat is not None:
        fig_lat = _gauge(lat, SLA_LATENCY_MS * 2, "E2E Latency", " ms",
                         warn=SLA_LATENCY_MS / (SLA_LATENCY_MS * 2),
                         crit=0.80, height=200)
        fig_lat.update_traces(gauge_threshold_value=SLA_LATENCY_MS)
        st.plotly_chart(fig_lat, key="qos_latency_gauge",
                        config={"displayModeBar": False})


def render_oversubscription(placement: Optional[dict], pod_node_map: dict[str, str]) -> None:
    """Separate MILP model pressure from Kubernetes overcommit and runtime load."""
    st.markdown('<p class="section-header">🔁 CPU Pressure & Oversubscription</p>',
                unsafe_allow_html=True)
    st.caption(
        "Model view stays tied to MILP placement. Runtime view counts every pod "
        "on the node, including agents, Redis, monitoring, and stress pods."
    )

    nodes = get_cluster_worker_nodes()
    pod_totals = get_node_pod_resource_totals()
    resource_usage = (placement or {}).get("resource_usage", {}) or {}

    rows: list[dict[str, Any]] = []
    for node in nodes:
        cap = float(get_node_capacity_cpu(node) or 0.0)
        util = get_node_cpu_util(node)
        node_data = pod_totals.get(node, {})

        try:
            model_idx = NODES.index(node)
            model_key = f"n{model_idx}"
        except ValueError:
            model_key = node
        model_cpu = _safe_float(resource_usage.get(model_key, resource_usage.get(node)), 0.0)

        req_cpu = float(node_data.get("cpu_request", 0.0) or 0.0)
        limit_cpu = float(node_data.get("cpu_limit", 0.0) or 0.0)
        actual_cpu = float(util * cap) if util is not None and cap else 0.0

        model_ratio = model_cpu / cap if cap else 0.0
        request_ratio = req_cpu / cap if cap else 0.0
        limit_ratio = limit_cpu / cap if cap else 0.0
        actual_ratio = actual_cpu / cap if cap else 0.0

        overcommitted = limit_ratio > 1.0
        saturated = actual_ratio >= 0.90
        if overcommitted and saturated:
            state = "overcommitted + saturated"
        elif overcommitted:
            state = "overcommitted"
        elif saturated:
            state = "saturated"
        else:
            state = "normal"

        rows.append({
            "Node": node,
            "Status": state,
            "MILP modeled CPU": model_cpu,
            "θ_model": model_ratio,
            "K8s requests": req_cpu,
            "Request ratio": request_ratio,
            "K8s limits": limit_cpu,
            "Limit ratio": limit_ratio,
            "Actual node CPU": actual_cpu,
            "Runtime ratio": actual_ratio,
            "Pod count": len(node_data.get("pods", []) or []),
        })

    if not rows:
        st.info("No worker-node resource data available.")
        return

    df = pd.DataFrame(rows)
    fig = go.Figure()
    series = [
        ("θ_model (MILP)", "θ_model", "#A78BFA"),
        ("K8s requests/cap", "Request ratio", "#60A5FA"),
        ("K8s limits/cap", "Limit ratio", "#F59E0B"),
        ("Actual CPU/cap", "Runtime ratio", "#EF4444"),
    ]
    for label, col, color in series:
        fig.add_trace(go.Bar(
            name=label,
            x=df["Node"],
            y=df[col],
            marker_color=color,
            text=[f"{v:.0%}" for v in df[col]],
            textposition="auto",
        ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="#FBBF24",
                  annotation_text="Capacity 100%", annotation_position="top left")
    fig.add_hline(y=1.3, line_dash="dash", line_color="#F87171",
                  annotation_text="Θmax 130%", annotation_position="top left")
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=10, r=10, t=40, b=35),
        yaxis_title="Ratio to CPU capacity",
        yaxis_range=[0, max(1.55, float(df[["θ_model", "Request ratio", "Limit ratio", "Runtime ratio"]].max().max()) * 1.12)],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch", key="cpu_pressure_runtime_overcommit",
                    config={"displayModeBar": False})

    table = df.copy()
    for col in ["MILP modeled CPU", "K8s requests", "K8s limits", "Actual node CPU"]:
        table[col] = table[col].map(lambda v: f"{v:.2f}c")
    for col in ["θ_model", "Request ratio", "Limit ratio", "Runtime ratio"]:
        table[col] = table[col].map(lambda v: f"{v:.1%}")
    st.dataframe(table, width="stretch", hide_index=True)

    st.caption(
        "MILP modeled CPU comes from `milp:placement.resource_usage`. "
        "K8s requests/limits count all pods on each node. Actual node CPU comes from node-exporter."
    )

    runtime_pods: list[dict[str, Any]] = []
    for node, node_data in pod_totals.items():
        for pod in node_data.get("pods", []) or []:
            if (pod.get("cpu_limit") or 0.0) <= 0 and (pod.get("cpu_request") or 0.0) <= 0:
                continue
            runtime_pods.append({
                "Node": node,
                "Namespace": pod.get("namespace", ""),
                "Pod": pod.get("pod", ""),
                "Phase": pod.get("phase", ""),
                "CPU request": f"{float(pod.get('cpu_request') or 0.0):.2f}c",
                "CPU limit": f"{float(pod.get('cpu_limit') or 0.0):.2f}c",
                "RAM request": f"{float(pod.get('mem_request_gib') or 0.0):.2f}Gi",
                "RAM limit": f"{float(pod.get('mem_limit_gib') or 0.0):.2f}Gi",
            })
    runtime_pods.sort(key=lambda row: _safe_float(str(row["CPU limit"]).rstrip("c")), reverse=True)
    with st.expander("Runtime pods counted in Kubernetes overcommit view", expanded=False):
        if runtime_pods:
            st.dataframe(pd.DataFrame(runtime_pods[:80]), width="stretch", hide_index=True)
        else:
            st.info("No pod requests/limits available from Kubernetes.")


def render_migration_feed(placement: Optional[dict]) -> None:
    """Show migration events for current cycle with color badges."""
    st.markdown('<p class="section-header">🔀 Migration Events — Current Cycle</p>',
                unsafe_allow_html=True)

    if not placement:
        st.info("No placement data.")
        return

    mt = placement.get("migration_types", {})
    pl = placement.get("placement", {})

    any_mig = False
    for svc, mtype in mt.items():
        if mtype != "Stayed":
            any_mig = True
            info = pl.get(svc, {})
            node_tgt = info.get("node", "?")
            var_tgt  = info.get("variant", "standard")
            badge_html = _migration_badge(mtype)
            v_badge = (f'<span class="badge" style="background:'
                       f'{VARIANT_COLORS.get(var_tgt, "#607D8B")};color:#000">'
                       f'{var_tgt}</span>')
            st.markdown(
                f"<div class='card'><b>{svc}</b> → <b>{node_tgt}</b>&nbsp;"
                f"{v_badge}&nbsp;{badge_html}</div>",
                unsafe_allow_html=True,
            )

    if not any_mig:
        st.success("✅ No migrations this cycle — all services stayed.")


def render_pod_status() -> None:
    """Live pod running status table via K8s API."""
    st.markdown('<p class="section-header">🖥️ Live Pod Status</p>',
                unsafe_allow_html=True)

    snapshot = get_cluster_state_snapshot()
    pods = snapshot.get("pods", [])
    deployments = snapshot.get("deployments", [])

    if pods:
        df = pd.DataFrame(pods)
        st.dataframe(
            df[["name", "app", "phase", "ready", "restarts", "node", "ip"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("No pods returned from cluster API.")

    with st.expander("📦 Deployment Replica State"):
        if deployments:
            ddf = pd.DataFrame(deployments)
            st.dataframe(
                ddf[["name", "ready", "desired", "available", "updated"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No deployments returned from cluster API.")

    # K8s events
    with st.expander("📋 Recent K8s Events"):
        events = get_recent_k8s_warning_events()
        if not events:
            st.success("No warning events.")
            return

        def _is_verifier_noise(ev: dict[str, str]) -> bool:
            obj = ev.get("object_name", "")
            msg = ev.get("message", "")
            return (
                obj.startswith("placement-test-")
                or (
                    "DefaultBinder" in msg
                    and "not found" in msg
                    and "placement-test-" in msg
                )
            )

        workload_events = [e for e in events if not _is_verifier_noise(e)]
        verifier_events = [e for e in events if _is_verifier_noise(e)]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Warning Events", len(events))
        with c2:
            st.metric("Workload-Relevant", len(workload_events))
        with c3:
            st.metric("Verifier Noise", len(verifier_events))

        show_verifier = st.toggle(
            "Show Digital Twin verifier events",
            value=False,
            help="placement-test-* events are verifier telemetry, not real workload incidents.",
        )

        def _aggregate(rows: list[dict[str, str]]) -> pd.DataFrame:
            agg: dict[tuple[str, str, str], dict[str, Any]] = {}
            for ev in rows:
                msg = ev.get("message", "")
                msg_norm = re.sub(r'placement-test-[a-f0-9]+-\d+', "placement-test-*-*", msg)
                object_name = ev.get("object_name", "")
                if object_name.startswith("placement-test-"):
                    object_group = "placement-test-*"
                else:
                    object_group = object_name
                key = (ev.get("reason", ""), object_group, msg_norm)
                ts = ev.get("timestamp", "")
                if key not in agg:
                    agg[key] = {
                        "Last Seen (GMT+7)": _fmt_ts(ts, "%H:%M:%S"),
                        "Reason": ev.get("reason", ""),
                        "Object": object_group,
                        "Message": msg_norm,
                        "Events": 1,
                    }
                else:
                    agg[key]["Events"] += 1
                    if ts > str(agg[key].get("_ts", "")):
                        agg[key]["Last Seen (GMT+7)"] = _fmt_ts(ts, "%H:%M:%S")
                agg[key]["_ts"] = ts

            out = list(agg.values())
            out.sort(key=lambda x: str(x.get("_ts", "")), reverse=True)
            for row in out:
                row.pop("_ts", None)
            return pd.DataFrame(out)

        st.markdown("**Operational Incidents (workload-relevant)**")
        if workload_events:
            st.dataframe(_aggregate(workload_events), width="stretch", hide_index=True)
        else:
            st.info("No workload-relevant warnings in current event window.")

        if show_verifier:
            st.markdown("**Digital Twin Verifier Diagnostics**")
            if verifier_events:
                st.dataframe(_aggregate(verifier_events), width="stretch", hide_index=True)
            else:
                st.info("No verifier events in current event window.")


def render_drl_tab(placement: Optional[dict]) -> None:
    """Full DRL + Digital Twin observability tab."""

    # ── 1. Mode control ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">⚙️ System Mode Control</p>',
                unsafe_allow_html=True)

    current_mode = get_system_mode()
    mode_cols = st.columns([1, 1, 1, 2])
    mode_map  = {"MILP only":   "milp",
                 "Shadow (DRL silent)": "shadow",
                 "Hybrid (DRL gated)": "hybrid",
                 "DRL governs": "drl"}
    for i, (label, key) in enumerate(mode_map.items()):
        with mode_cols[i]:
            is_active = current_mode == key
            btn_style = "primary" if is_active else "secondary"
            if st.button(
                ("✅ " if is_active else "") + label,
                key=f"mode_btn_{key}",
                type=btn_style,
                use_container_width=True,
            ):
                rdb = _get_redis_client()
                if rdb:
                    rdb.set("system:mode", key)
                    st.cache_data.clear()
                    st.success(f"Mode set to **{key}**")
                    st.rerun()
                else:
                    st.error("Redis unavailable")
    with mode_cols[3]:
        st.info(
            "**milp** — MILP controls everything (safe default)  \n"
            "**shadow** — DRL writes silently, MILP still governs  \n"
            "**hybrid** — DRL suggests, MILP safety gate applies  \n"
            "**drl** — DRL governs, MILP is fallback"
        )

    st.divider()

    # ── 2. DRL vs MILP placement comparison ──────────────────────────────────
    st.markdown('<p class="section-header">📍 DRL vs MILP Placement Comparison</p>',
                unsafe_allow_html=True)

    drl_pl  = get_drl_placement()
    milp_pl = placement
    drl_source = (drl_pl or {}).get("_source", "none")
    drl_accepted = bool((drl_pl or {}).get("_accepted", drl_source == "live"))
    milp_ts = (milp_pl or {}).get("timestamp")
    drl_ts = (drl_pl or {}).get("timestamp")
    drl_ref_ts = (drl_pl or {}).get("milp_ref_timestamp")
    milp_age = _age_seconds(milp_ts)
    drl_age = _age_seconds(drl_ts)
    snapshot_aligned = bool(drl_ref_ts and milp_ts and str(drl_ref_ts) == str(milp_ts))
    same_placement = False
    try:
        same_placement = (milp_pl or {}).get("placement", {}) == (drl_pl or {}).get("placement", {})
    except Exception:
        same_placement = False

    if drl_source == "proposed":
        if drl_accepted:
            st.warning("DRL placement shown is `proposed` (not yet committed as live).")
        else:
            st.warning("DRL placement shown is `proposed + rejected`; MILP still governs live placement.")
    elif drl_source == "live":
        st.caption(
            f"Hybrid compare context: aligned_ref={snapshot_aligned} | "
            f"MILP age={milp_age:.1f}s | DRL age={drl_age:.1f}s"
            if isinstance(milp_age, float) and isinstance(drl_age, float)
            else f"Hybrid compare context: aligned_ref={snapshot_aligned}"
        )

    if not milp_pl:
        st.warning("No MILP placement in Redis.")
    else:
        rows = []
        milp_nodes = milp_pl.get("placement", {})
        drl_nodes  = drl_pl.get("placement", {}) if drl_pl else {}
        for svc in SERVICES:
            milp_info = milp_nodes.get(svc, {})
            drl_info  = drl_nodes.get(svc, {})
            mn = milp_info.get("node", "?")
            dn = drl_info.get("node",  "—")  if drl_pl else "—"
            mv = milp_info.get("variant", "std")
            dv = drl_info.get("variant", "—") if drl_pl else "—"
            match = "✅" if mn == dn else ("⚠️" if dn != "—" else "➖")
            rows.append({
                "Service":      svc,
                "MILP Node":   mn,
                "MILP Variant": mv,
                "DRL Node":    dn,
                "DRL Variant": dv,
                "Match":       match,
            })
        df_cmp = pd.DataFrame(rows)
        st.dataframe(df_cmp, hide_index=True, use_container_width=True)

        # Visual diff: sankey-style node allocation bar
        lcol, rcol = st.columns(2)
        for col, title, data, color in [
            (lcol, "MILP: services per node", milp_nodes, NODE_COLORS),
            (rcol, "DRL:  services per node", drl_nodes,  NODE_COLORS),
        ]:
            node_counts: dict[str, int] = {n: 0 for n in NODES}
            for svc in SERVICES:
                n = data.get(svc, {}).get("node", "")
                if n in node_counts:
                    node_counts[n] += 1
            fig = go.Figure(go.Bar(
                x=list(node_counts.keys()),
                y=list(node_counts.values()),
                marker_color=[NODE_COLORS.get(n, "#607D8B") for n in node_counts],
                text=list(node_counts.values()),
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(size=12)),
                height=200, margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                yaxis=dict(dtick=1, range=[0, len(SERVICES) + 0.5]),
            )
            with col:
                st.plotly_chart(fig, use_container_width=True,
                                key=f"placement_cmp_{title[:4].strip()}",
                                config={"displayModeBar": False})

    st.divider()

    # ── 3. Digital Twin stats ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">🔬 Digital Twin Validation (last inference)</p>',
                unsafe_allow_html=True)

    twin = get_twin_stats()
    if not twin:
        st.info("No `drl:twin_stats` in Redis — DRL agent not running or no recent inference.")
    else:
        is_safe       = twin.get("is_safe", False)
        safe_ratio    = _safe_float(twin.get("sla_safe_rate", twin.get("safe_ratio")), 0.0)
        objective_j_raw = _safe_float(twin.get("objective_j_raw", twin.get("predicted_j")), 0.0)
        objective_j_pen = _safe_float(twin.get("objective_j_penalized", twin.get("predicted_j")), 0.0)
        drl_comparable_obj = _safe_float(
            (drl_pl or {}).get("comparable_objective", (drl_pl or {}).get("objective")),
            objective_j_raw,
        )
        drl_live_obj = drl_comparable_obj
        twin_expected_obj = _safe_float(
            (drl_pl or {}).get("twin_expected_objective"),
            objective_j_raw,
        )
        comparable_source = (drl_pl or {}).get(
            "comparable_objective_source",
            (drl_pl or {}).get("objective_source", "unknown"),
        )
        milp_obj      = _safe_float((placement or {}).get("objective"), 0.0)
        n_rollouts    = _safe_int(twin.get("n_rollouts"), 0)
        reason        = twin.get("reason", "—")
        ts_twin       = twin.get("timestamp", "")
        storm_ok      = bool(twin.get("storm_ok", True))
        migration_cnt = _safe_int(twin.get("migration_count"), 0)
        v_storm_max   = _safe_int(twin.get("v_storm_max"), 2)
        model_contract = twin.get("model_contract", "unknown")
        adapter_name = twin.get("adapter_name", "none")
        inference_enabled = bool(twin.get("inference_enabled", True))

        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        with tc1:
            color = "#16A34A" if is_safe else "#EF4444"
            st.markdown(
                f'<div style="text-align:center; padding:12px; background:#1e2535; '
                f'border-radius:10px; border:2px solid {color};">'  
                f'<div style="font-size:2rem;">{"✅" if is_safe else "❌"}</div>'  
                f'<div style="color:{color}; font-weight:700; font-size:1rem;">'
                f'{"SAFE — accepted" if is_safe else "UNSAFE — rejected"}</div></div>',
                unsafe_allow_html=True,
            )
        with tc2:
            st.metric("Safe Ratio", f"{safe_ratio*100:.1f}%",
                      delta="≥80% required",
                      delta_color="normal" if safe_ratio >= 0.8 else "inverse")
        with tc3:
            if drl_source == "live":
                gap = 100 * (drl_live_obj - milp_obj) / abs(milp_obj) if milp_obj else 0
                if snapshot_aligned:
                    st.metric("DRL Comparable J", f"{drl_live_obj:.4f}",
                              delta=f"{'+' if gap>=0 else ''}{gap:.1f}% vs MILP (aligned)",
                              delta_color="off")
                else:
                    st.metric("DRL Comparable J", f"{drl_live_obj:.4f}",
                              delta=f"{'+' if gap>=0 else ''}{gap:.1f}% vs MILP (drift risk)",
                              delta_color="off")
            else:
                st.metric("DRL Proposed Twin J", f"{objective_j_raw:.4f}",
                          delta="not live", delta_color="off")
        with tc4:
            st.metric("Storm Check", "PASS" if storm_ok else "FAIL",
                      delta=f"{migration_cnt}/{v_storm_max} migrations",
                      delta_color="normal" if storm_ok else "inverse")
        with tc5:
            st.metric("Last Validated", _fmt_ts(ts_twin, "%H:%M:%S"),
                      delta=_fmt_ts(ts_twin, "%Y-%m-%d"), delta_color="off")

        st.caption(f"Monte Carlo rollouts: {n_rollouts}")
        st.caption(
            f"DRL Contract: `{model_contract}` | Adapter: `{adapter_name}` | "
            f"Inference: {'enabled' if inference_enabled else 'disabled'}"
        )
        st.caption(
            f"Comparable objective source: `{comparable_source}` | "
            f"Twin expected J: `{twin_expected_obj:.4f}`"
        )
        if abs(objective_j_pen - objective_j_raw) > 1e-9:
            st.caption(
                f"Penalized score: {objective_j_pen:.4f} "
                f"(storm/infeasibility penalties applied)"
            )
        if same_placement and abs(drl_live_obj - milp_obj) <= 1e-6 and abs(twin_expected_obj - milp_obj) > 1e-6:
            st.caption(
                "Same placement: comparable J is aligned to MILP; Twin expected J is shown "
                "separately as a Monte-Carlo robustness estimate."
            )
        if reason and reason != "—":
            st.caption(f"Rejection reason: {reason}")

        # Safe ratio gauge — invert=True: green=high (≥80% safe is good)
        fig_sr = _gauge(safe_ratio * 100, 100, "Monte Carlo Safe %", "%",
                        warn=0.80, crit=0.70, height=180, invert=True)
        st.plotly_chart(fig_sr, use_container_width=True, key="twin_safe_ratio_gauge",
                        config={"displayModeBar": False})

    st.divider()

    # ── 3b. Tier-2 Placement Verifier Status ──────────────────────────────────
    st.markdown('<p class="section-header">🔎 Tier-2 Placement Verifier</p>',
                unsafe_allow_html=True)

    tier2 = get_tier2_status()
    t2_verified = tier2.get("verified")
    t2_revoked  = tier2.get("revoked")
    t2_source = tier2.get("source", "none")
    t2_verified_stale = bool(tier2.get("verified_stale", False))
    t2_revoked_stale = bool(tier2.get("revoked_stale", False))
    t2_verified_age = tier2.get("verified_age_s")
    t2_revoked_age = tier2.get("revoked_age_s")

    if not t2_verified and not t2_revoked:
        st.info("No Tier-2 verification result in Redis — no recent verdict yet.")
    else:
        if t2_source == "latest":
            st.caption("Tier-2 source: durable latest verdict (recent TTL key expired).")
        elif t2_source == "recent":
            st.caption("Tier-2 source: recent live verdict.")
        if t2_verified_stale or t2_revoked_stale:
            st.warning(
                "Tier-2 verdict appears stale. "
                "Verifier may not have produced a fresh verdict for current proposal yet."
            )
        va, vb = st.columns(2)
        with va:
            if t2_verified:
                fp   = t2_verified.get("proposed_fingerprint", "—")
                nodes = t2_verified.get("verified_nodes")
                if isinstance(nodes, list) and nodes:
                    node = ", ".join(str(n) for n in nodes)
                else:
                    node = t2_verified.get("assigned_node", "—")
                ts   = t2_verified.get("timestamp", "")
                age_txt = f" · age={t2_verified_age:.1f}s" if isinstance(t2_verified_age, (int, float)) else ""
                st.success(
                    f"**Verified** — "
                    f"node: `{node}`  ·  fp: `{fp}`  ·  {_fmt_ts(ts, '%H:%M:%S')}{age_txt}"
                )
            else:
                st.info("No verified result yet.")
        with vb:
            if t2_revoked:
                fp     = t2_revoked.get("proposed_fingerprint", "—")
                reason = t2_revoked.get("reason", "—")
                ts     = t2_revoked.get("timestamp", "")
                age_txt = f" · age={t2_revoked_age:.1f}s" if isinstance(t2_revoked_age, (int, float)) else ""
                st.error(
                    f"**Revoked** — "
                    f"reason: `{reason}`  ·  fp: `{fp}`  ·  {_fmt_ts(ts, '%H:%M:%S')}{age_txt}"
                )
            else:
                st.info("No revoked result yet.")

    st.divider()

    # ── 4. Expert trajectory buffer ───────────────────────────────────────────
    st.markdown('<p class="section-header">📚 Expert Trajectory Buffer</p>',
                unsafe_allow_html=True)

    traj_count = get_expert_traj_count()
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.metric("milp:expert_trajectories", f"{traj_count:,}",
                  delta="LLEN in Redis", delta_color="off")
    with ec2:
        st.metric("BC training threshold", "≥ 200",
                  delta="✅ ready" if traj_count >= 200 else "⏳ collecting",
                  delta_color="normal" if traj_count >= 200 else "off")
    with ec3:
        st.metric("Minari dataset", "drl-milp-v0",
                  delta="exported ✅", delta_color="off")

    fig_buf = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(traj_count, 2000),
        number={"suffix": " traj", "font": {"size": 22}},
        title={"text": "Expert Trajectory Buffer (cap 2000 shown)"},
        gauge={
            "axis": {"range": [0, 2000], "tickfont": {"size": 10}},
            "bar":  {"color": "#16A34A" if traj_count >= 500 else "#D97706"},
            "steps": [
                {"range": [0, 200],   "color": "#2a1010"},
                {"range": [200, 500], "color": "#2a2010"},
                {"range": [500, 2000],"color": "#1e2a1e"},
            ],
            "threshold": {"line": {"color": "#16A34A", "width": 3},
                          "thickness": 0.75, "value": 500},
        },
    ))
    fig_buf.update_layout(
        height=200, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
    )
    st.plotly_chart(fig_buf, use_container_width=True,
                    config={"displayModeBar": False})


def render_log_viewer() -> None:
    """Tabbed log viewer for key deployments."""
    deploys = ["milp-agent", "variant-controller", "api-gateway",
               "detection", "gen-ai", "ingest", "preprocess", "postprocess"]
    tabs = st.tabs([f"📋 {d}" for d in deploys])
    for i, deploy in enumerate(deploys):
        with tabs[i]:
            logs = get_recent_logs(deploy, lines=30)
            st.code(logs, language="log")


def render_simulator_pipeline() -> None:
    """Summarize end-to-end pipeline behaviour from simulator JSONL outputs."""
    st.markdown('<p class="section-header">🎬 Simulator Pipeline (client_results.jsonl)</p>',
                unsafe_allow_html=True)

    sim_path = Path(SIM_RESULTS_FILE)
    rows = get_simulator_tail()
    if not rows:
        st.info(f"No simulator data found at {SIM_RESULTS_FILE}")
        return

    now_s = time.time()
    try:
        file_mtime_s = sim_path.stat().st_mtime
    except OSError:
        file_mtime_s = 0.0
    file_age_s = max(0.0, now_s - file_mtime_s) if file_mtime_s else 0.0
    if file_mtime_s and file_age_s > 120:
        st.warning(
            "Simulator JSONL is stale: "
            f"last updated {_fmt_ts(file_mtime_s, '%Y-%m-%d %H:%M:%S')} "
            f"({file_age_s / 60:.1f} min ago). "
            "Reasoning rows below may be from an old simulator run."
        )

    latest = rows[-1]
    frame_count = len(rows)

    e2e_vals = [float(r.get("e2e_latency_ms", 0.0)) for r in rows if r.get("e2e_latency_ms") is not None]
    rtt_vals = [float(r.get("client_rtt_ms", 0.0)) for r in rows if r.get("client_rtt_ms") is not None]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Frames (tail)", str(frame_count), delta=f"file: {sim_path.name}")
    with c2:
        st.metric("Latest E2E", f"{latest.get('e2e_latency_ms', 0):.1f} ms")
    with c3:
        st.metric("Latest RTT", f"{latest.get('client_rtt_ms', 0):.1f} ms")
    with c4:
        det_var = latest.get("variant", "unknown")
        st.metric("Detection Variant", str(det_var))

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.metric("GenAI Variant", str(latest.get("gen_ai_variant", "N/A")))
    with pcol2:
        st.metric("GenAI Model", str(latest.get("gen_ai_model", "N/A")))
    with pcol3:
        st.metric("GenAI Mode", str(latest.get("gen_ai_mode", "N/A")))
    with pcol4:
        report = str(latest.get("incident_report", "N/A"))
        st.metric("Report", report[:40] + ("..." if len(report) > 40 else ""))

    if e2e_vals:
        e2e_series = pd.Series(e2e_vals)
        st.caption(
            f"E2E tail stats: avg={e2e_series.mean():.1f}ms, p50={e2e_series.quantile(0.50):.1f}ms, "
            f"p95={e2e_series.quantile(0.95):.1f}ms"
        )

    # Per-stage average latency from simulator rows.
    stage_names = ["ingest_ms", "preprocess_ms", "detection_ms", "gen_ai_ms", "postprocess_ms"]
    stage_acc: dict[str, list[float]] = {k: [] for k in stage_names}
    for row in rows:
        sl = row.get("stage_latencies", {}) or {}
        stage_acc["ingest_ms"].append(float(sl.get("ingest_ms", row.get("ingest_latency_ms", 0.0) or 0.0)))
        stage_acc["preprocess_ms"].append(float(sl.get("preprocess_ms", row.get("preprocess_latency_ms", 0.0) or 0.0)))
        stage_acc["detection_ms"].append(float(sl.get("detection_ms", row.get("detection_latency_ms", 0.0) or 0.0)))
        stage_acc["gen_ai_ms"].append(float(sl.get("gen_ai_ms", row.get("gen_ai_latency_ms", 0.0) or 0.0)))
        stage_acc["postprocess_ms"].append(float(sl.get("postprocess_ms", row.get("postprocess_latency_ms", 0.0) or 0.0)))

    avg_stage = {k: (sum(v) / len(v) if v else 0.0) for k, v in stage_acc.items()}
    stage_order = ["ingest_ms", "preprocess_ms", "detection_ms", "gen_ai_ms", "postprocess_ms"]
    labels = [s.replace("_ms", "") for s in stage_order]
    values = [avg_stage[s] for s in stage_order]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=["#1f77b4", "#ff7f0e", "#d62728", "#17a2b8", "#9467bd"],
        text=[f"{v:.1f}ms" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        title=dict(text="Average Stage Latency (simulator tail)", font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis_title="ms",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Separate reasoning stream (status + delivered events) for easier debugging.
    reasoning_events: list[dict[str, Any]] = []
    for row in rows:
        row_ts = row.get("timestamp", "")
        events = row.get("reasoning_events", []) or []
        if events:
            for ev in events:
                reasoning_events.append({
                    "time": _fmt_ts(ev.get("timestamp", row_ts), "%H:%M:%S"),
                    "event": ev.get("event", "unknown"),
                    "source_frame": ev.get("source_frame_id", row.get("frame_id")),
                    "reasoning_seq": ev.get("reasoning_seq", "-"),
                    "mode": ev.get("gen_ai_mode", "unknown"),
                    "variant": ev.get("gen_ai_variant", "unknown"),
                    "model": ev.get("gen_ai_model", "unknown"),
                    "gen_ai_ms": round(float(ev.get("gen_ai_latency_ms", 0.0) or 0.0), 2),
                    "report": str(ev.get("incident_report", "")),
                })
            continue

        # Backward compatibility for old JSONL rows without reasoning_events.
        mode = row.get("gen_ai_mode", "N/A")
        report = str(row.get("incident_report", ""))
        if mode in ("queued", "sampled_out", "error") or report:
            reasoning_events.append({
                "time": _fmt_ts(row_ts, "%H:%M:%S"),
                "event": "status",
                "source_frame": row.get("frame_id"),
                "reasoning_seq": "-",
                "mode": mode,
                "variant": row.get("gen_ai_variant", "unknown"),
                "model": row.get("gen_ai_model", "unknown"),
                "gen_ai_ms": round(float(row.get("gen_ai_latency_ms", 0.0) or 0.0), 2),
                "report": report,
            })

        for delivered in row.get("reasoning_deliveries", []) or []:
            reasoning_events.append({
                "time": _fmt_ts(row_ts, "%H:%M:%S"),
                "event": "delivered",
                "source_frame": row.get("frame_id"),
                "reasoning_seq": delivered.get("reasoning_seq", "-"),
                "mode": delivered.get("gen_ai_mode", "unknown"),
                "variant": delivered.get("gen_ai_variant", "unknown"),
                "model": delivered.get("gen_ai_model", "unknown"),
                "gen_ai_ms": round(float(delivered.get("gen_ai_latency_ms", 0.0) or 0.0), 2),
                "report": str(delivered.get("incident_report", "")),
            })

    st.markdown('<p class="section-header">🧠 AI Reasoning Stream</p>', unsafe_allow_html=True)
    if reasoning_events:
        stream_df = pd.DataFrame(reasoning_events[-80:]).iloc[::-1]
        st.dataframe(stream_df, width="stretch", hide_index=True)
    else:
        st.info("No reasoning events found in simulator tail yet.")

    # Recent rows table for quick troubleshooting.
    preview = []
    for row in rows[-30:]:
        sl = row.get("stage_latencies", {}) or {}
        report = str(row.get("incident_report", ""))
        preview.append({
            "time": _fmt_ts(row.get("timestamp", ""), "%H:%M:%S"),
            "frame_id": row.get("frame_id"),
            "variant": row.get("variant"),
            "gen_ai_variant": row.get("gen_ai_variant", "N/A"),
            "gen_ai_model": row.get("gen_ai_model", "N/A"),
            "gen_ai_mode": row.get("gen_ai_mode", "N/A"),
            "reasoning_delivered": len(row.get("reasoning_deliveries", []) or []),
            "report": report[:60] + ("..." if len(report) > 60 else ""),
            "e2e_latency_ms": round(float(row.get("e2e_latency_ms", 0.0) or 0.0), 2),
            "client_rtt_ms": round(float(row.get("client_rtt_ms", 0.0) or 0.0), 2),
            "ingest_ms": round(float(sl.get("ingest_ms", row.get("ingest_latency_ms", 0.0) or 0.0)), 2),
            "preprocess_ms": round(float(sl.get("preprocess_ms", row.get("preprocess_latency_ms", 0.0) or 0.0)), 2),
            "detection_ms": round(float(sl.get("detection_ms", row.get("detection_latency_ms", 0.0) or 0.0)), 2),
            "gen_ai_ms": round(float(sl.get("gen_ai_ms", row.get("gen_ai_latency_ms", 0.0) or 0.0)), 2),
            "postprocess_ms": round(float(sl.get("postprocess_ms", row.get("postprocess_latency_ms", 0.0) or 0.0)), 2),
        })
    st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
def _render_weight_editor() -> None:
    """Sidebar expander: sliders that write objective weights to Redis milp:weights."""
    current = get_milp_weights()
    with st.expander("⚖️ Objective Weights", expanded=False):
        st.caption(
            f"Source: `{current.get('source', 'unknown')}` · "
            "Changes take effect on the next solve cycle (~30 s)."
        )
        with st.form("weight_editor_form", clear_on_submit=False):
            w_c = st.slider(
                "w_c — Energy efficiency",
                min_value=0.0, max_value=1.0,
                value=float(current.get("w_c", 0.15)),
                step=0.01,
                help="Weight for normalised energy cost C_norm.",
            )
            w_d = st.slider(
                "w_d — Migration disruption",
                min_value=0.0, max_value=1.0,
                value=float(current.get("w_d", 0.10)),
                step=0.01,
                help="Weight for normalised disruption cost D_norm.",
            )
            w_a = st.slider(
                "w_a — Detection accuracy",
                min_value=0.0, max_value=1.0,
                value=float(current.get("w_a", 0.75)),
                step=0.01,
                help="Weight for normalised accuracy gain A_norm (maximised).",
            )
            total = round(w_c + w_d + w_a, 4)
            st.caption(f"Sum = **{total}** {'✅' if abs(total - 1.0) < 0.02 else '⚠️ should be ≈ 1.0'}")
            submitted = st.form_submit_button("Apply Weights", type="primary")

        if submitted:
            if abs((w_c + w_d + w_a) - 1.0) > 0.02:
                st.error(f"Weights sum to {w_c+w_d+w_a:.3f}. Adjust so they sum to ≈ 1.0.")
            else:
                rdb = _get_redis_client()
                if rdb:
                    try:
                        rdb.set("milp:weights", json.dumps({"w_c": w_c, "w_d": w_d, "w_a": w_a}))
                        st.success(f"✅ Weights written to Redis — w_c={w_c}, w_d={w_d}, w_a={w_a}")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Redis write failed: {exc}")
                else:
                    st.error("Redis unavailable — cannot persist weights.")


# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[bool, int]:
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/microchip.svg",
                 width=40)
        st.title("MILP Control")
        st.caption("Edge Computing Dashboard")
        st.divider()

        auto = st.checkbox("🔄 Auto-refresh", value=True)
        interval = st.slider("Refresh interval (s)", 5, 60, 15)

        if st.button("🔄 Force Refresh Now"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.subheader("🔧 Quick Actions")

        st.subheader("🤖 DRL Mode")
        current_mode = get_system_mode()
        mode_css = {"milp": "mode-milp", "shadow": "mode-shadow",
                    "hybrid": "mode-hybrid", "drl": "mode-drl", "unknown": "mode-unknown"}
        st.markdown(
            f'<span class="{mode_css.get(current_mode, "mode-unknown")}">{current_mode.upper()}</span>',
            unsafe_allow_html=True,
        )
        for btn_label, btn_mode in [("Set MILP", "milp"),
                                     ("Set Shadow", "shadow"),
                                     ("Set Hybrid", "hybrid"),
                                     ("Set DRL", "drl")]:
            if st.button(btn_label, key=f"sb_mode_{btn_mode}",
                         type="primary" if current_mode == btn_mode else "secondary"):
                rdb = _get_redis_client()
                if rdb:
                    rdb.set("system:mode", btn_mode)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Redis unavailable")

        st.divider()
        if st.button("📊 Redis → Placement JSON"):
            raw = _kubectl(f"exec deploy/redis -n {NAMESPACE} -- redis-cli GET milp:placement")
            if raw:
                try:
                    st.json(json.loads(raw))
                except Exception:
                    st.code(raw)
            else:
                st.error("No data")

        if st.button("🔄 Clear Metric Cache"):
            st.cache_data.clear()
            st.success("Cache cleared, refreshing…")
            time.sleep(0.5)
            st.rerun()

        st.divider()
        st.subheader("⚙️ Control Parameters")
        st.markdown(f"**Control Interval:** `{CONTROL_INTERVAL} s`")
        st.markdown(f"**SLA Latency:** `{SLA_LATENCY_MS} ms`")
        st.markdown(f"**Solver Timeout:** `{SOLVER_TIMEOUT_S} s`")
        st.markdown(f"**Max Fallback Ratio:** `{MAX_FALLBACK_RATIO}`")

        st.divider()
        _render_weight_editor()

        st.divider()
        st.subheader("📡 Prometheus Links")
        base = PROM_URL.replace("/api/v1/query", "")
        st.markdown(f"[Open Prometheus UI]({base})")
        st.markdown(f"[Node CPU query]({base}/graph?g0.expr=1+-+avg(rate(node_cpu_seconds_total[60s])))")

        st.divider()
        st.caption(f"MILP Control v2.0 · {_fmt_ts(time.time(), '%H:%M:%S')} (GMT+7, Hanoi)")

    return auto, interval


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    auto, interval = render_sidebar()

    if auto and st_autorefresh is not None:
        st_autorefresh(interval=interval * 1000, key="milp-dash-ar")
    elif auto and st_autorefresh is None:
        st.sidebar.warning("streamlit-autorefresh not installed.\n"
                           "`pip install streamlit-autorefresh`")

    # ── Global data fetches ──────────────────────────────────────────────────
    placement   = get_milp_placement()
    confirmed   = get_confirmed_placement()
    controller_sync = get_controller_sync_status()
    pod_node_map = get_pod_node_map()

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🤖 MILP + DRL Edge Control — Cluster Dashboard")
    render_header(placement)
    st.divider()

    # ── Health bar ────────────────────────────────────────────────────────────
    render_health_bar()
    st.divider()

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_overview, tab_placement, tab_solver, tab_drl, tab_pods, tab_logs = st.tabs([
        "🌐 System Overview",
        "📍 Workloads & Placement",
        "📐 MILP Solver & QoS",
        "🤖 DRL & Twin",
        "🖥 Pods",
        "📋 Logs",
    ])

    with tab_overview:
        render_cluster_state_summary()
        render_node_resources()
        render_energy_panel()
        render_oversubscription(placement, pod_node_map)

    with tab_placement:
        if controller_sync:
            st.caption(
                "Controller sync: "
                f"in_sync={controller_sync.get('in_sync')} "
                f"drift_count={controller_sync.get('drift_count', 0)} "
                f"at {_fmt_ts(controller_sync.get('timestamp', ''), '%H:%M:%S')}"
            )
        else:
            st.caption("Controller sync: unavailable (milp:controller_sync not published yet)")
        render_placement_table(placement, confirmed, pod_node_map)
        render_migration_feed(placement)
        render_container_resources()

    with tab_solver:
        render_objective_panel(placement)
        render_detection_qos()
        render_simulator_pipeline()

    with tab_drl:
        render_drl_tab(placement)

    with tab_pods:
        render_pod_status()

    with tab_logs:
        render_log_viewer()


if __name__ == "__main__":
    main()
