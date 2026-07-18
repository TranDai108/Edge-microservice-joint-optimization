import os
import time
import logging
import asyncio
import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Histogram, Counter, make_asgi_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("api-gateway")

app = FastAPI(title="API Gateway")

# ── Metrics ──
LATENCY_BUCKETS_MS = [5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2500, 5000, 10000, 30000, 60000]

LATENCY = Histogram(
    "service_latency_ms",
    "Processing latency in ms",
    ["service"],
    buckets=LATENCY_BUCKETS_MS,
)
REQUESTS = Counter(
    "service_requests_total",
    "Total requests processed",
    ["service", "status"]
)
E2E_LATENCY = Histogram(
    "e2e_latency_ms",
    "End-to-end pipeline latency in ms",
    buckets=LATENCY_BUCKETS_MS,
)

app.mount("/metrics", make_asgi_app())

SERVICE_NAME  = os.getenv("SERVICE_NAME", "api-gateway")
MY_NODE_NAME  = os.getenv("MY_NODE_NAME", "unknown")
REDIS_HOST    = os.getenv("REDIS_HOST", "redis")
_redis_port_raw = os.getenv("REDIS_PORT", "6379")
# K8s injects SERVICE_PORT as "tcp://host:port" — extract the numeric port safely.
REDIS_PORT    = int(_redis_port_raw.rsplit(":", 1)[-1]) if ":" in _redis_port_raw else int(_redis_port_raw)
_E2E_REDIS_KEY = "kltn:e2e_latency"
_E2E_REDIS_TTL = 30  # seconds — refresh each request; stale after 30 s of silence

_rdb: aioredis.Redis | None = None
GEN_AI_FRAME_INTERVAL = max(1, int(os.getenv("GEN_AI_FRAME_INTERVAL", "1")))
GEN_AI_TIMEOUT_S = float(os.getenv("GEN_AI_TIMEOUT_S", "25"))
GEN_AI_CONNECT_TIMEOUT_S = float(os.getenv("GEN_AI_CONNECT_TIMEOUT_S", "3"))
GEN_AI_WRITE_TIMEOUT_S = float(os.getenv("GEN_AI_WRITE_TIMEOUT_S", "10"))
GEN_AI_RETRY_COUNT = max(0, int(os.getenv("GEN_AI_RETRY_COUNT", "1")))
GEN_AI_RETRY_BACKOFF_S = float(os.getenv("GEN_AI_RETRY_BACKOFF_S", "0.35"))
GEN_AI_MAX_INFLIGHT = max(1, int(os.getenv("GEN_AI_MAX_INFLIGHT", "24")))

# Ordered pipeline — each stage URL set via env vars or K8s ConfigMap
PIPELINE = [
    os.getenv("INGEST_URL",      "http://ingest:8000"),
    os.getenv("PREPROCESS_URL",  "http://preprocess:8000"),
    os.getenv("DETECTION_URL",   "http://detection:8000"),
    os.getenv("GEN_AI_URL",      "http://gen-ai:8000"),
    os.getenv("POSTPROCESS_URL", "http://postprocess:8000"),
]

PIPELINE_NAMES = ["ingest", "preprocess", "detection", "gen_ai", "postprocess"]

INGEST_URL, PREPROCESS_URL, DETECTION_URL, GEN_AI_URL, POSTPROCESS_URL = PIPELINE

_state_lock = asyncio.Lock()
_frame_seq = 0
_reasoning_seq = 0
_next_emit_reasoning_seq = 0
_reasoning_tasks: dict[int, asyncio.Task] = {}
_reasoning_ready: dict[int, dict] = {}
_gen_ai_client: httpx.AsyncClient | None = None


def _gen_ai_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=GEN_AI_CONNECT_TIMEOUT_S,
        read=GEN_AI_TIMEOUT_S,
        write=GEN_AI_WRITE_TIMEOUT_S,
        pool=GEN_AI_CONNECT_TIMEOUT_S,
    )


def _is_sampled_frame(frame_seq: int) -> bool:
    return (frame_seq % GEN_AI_FRAME_INTERVAL) == 0


def _extract_reasoning_payload(reasoned_data: dict, reasoning_seq: int) -> dict:
    return {
        "reasoning_seq": reasoning_seq,
        "frame_id": reasoned_data.get("frame_id"),
        "incident_report": reasoned_data.get("incident_report", ""),
        "gen_ai_latency_ms": reasoned_data.get("gen_ai_latency_ms", 0.0),
        "gen_ai_mode": reasoned_data.get("gen_ai_mode", "unknown"),
        "gen_ai_model": reasoned_data.get("gen_ai_model", "unknown"),
        "gen_ai_variant": reasoned_data.get("gen_ai_variant", "unknown"),
        "gen_ai_sampled": reasoned_data.get("gen_ai_sampled", True),
        "gen_ai_node": reasoned_data.get("gen_ai_node", "unknown"),
    }


def _build_async_error_payload(reasoning_seq: int, incident_report: str) -> dict:
    return {
        "reasoning_seq": reasoning_seq,
        "frame_id": None,
        "incident_report": incident_report,
        "gen_ai_latency_ms": 0.0,
        "gen_ai_mode": "error",
        "gen_ai_model": "n/a",
        "gen_ai_variant": "n/a",
        "gen_ai_sampled": True,
        "gen_ai_node": "unknown",
    }


def _format_async_exception(exc: BaseException) -> str:
    exc_type = type(exc).__name__

    if isinstance(exc, httpx.TimeoutException):
        req = getattr(exc, "request", None)
        target = str(req.url) if req and getattr(req, "url", None) else f"{GEN_AI_URL}/process"
        return (
            f"{exc_type}: timeout calling {target} "
            f"(read={GEN_AI_TIMEOUT_S:.1f}s connect={GEN_AI_CONNECT_TIMEOUT_S:.1f}s)"
        )

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else "unknown"
        target = str(exc.request.url) if exc.request is not None else f"{GEN_AI_URL}/process"
        body = ""
        if exc.response is not None:
            body = (exc.response.text or "").strip().replace("\n", " ")
        body = body[:200]
        if body:
            return f"{exc_type}: status={status} url={target} body={body}"
        return f"{exc_type}: status={status} url={target}"

    detail = str(exc).strip() or repr(exc)
    return f"{exc_type}: {detail}"


@app.on_event("startup")
async def startup_event():
    global _gen_ai_client, _rdb
    _gen_ai_client = httpx.AsyncClient(timeout=_gen_ai_timeout())
    try:
        _rdb = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True, socket_timeout=1.0,
        )
    except Exception as exc:
        log.warning("Redis unavailable at startup — e2e monitoring key disabled: %s", exc)
        _rdb = None


@app.on_event("shutdown")
async def shutdown_event():
    global _gen_ai_client, _rdb
    if _gen_ai_client is not None:
        await _gen_ai_client.aclose()
        _gen_ai_client = None
    if _rdb is not None:
        await _rdb.aclose()
        _rdb = None


async def _run_gen_ai_in_background(frame_payload: dict, reasoning_seq: int) -> dict:
    for attempt in range(GEN_AI_RETRY_COUNT + 1):
        try:
            client = _gen_ai_client
            if client is None:
                async with httpx.AsyncClient(timeout=_gen_ai_timeout()) as temp_client:
                    resp = await temp_client.post(f"{GEN_AI_URL}/process", json=frame_payload)
            else:
                resp = await client.post(f"{GEN_AI_URL}/process", json=frame_payload)

            resp.raise_for_status()
            reasoned = resp.json()
            return _extract_reasoning_payload(reasoned, reasoning_seq)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            should_retry = status >= 500 and attempt < GEN_AI_RETRY_COUNT
            if not should_retry:
                raise
            backoff_s = GEN_AI_RETRY_BACKOFF_S * (attempt + 1)
            log.warning(
                "reasoning_seq=%s gen-ai HTTP %s, retrying in %.2fs (%s/%s)",
                reasoning_seq,
                status,
                backoff_s,
                attempt + 1,
                GEN_AI_RETRY_COUNT,
            )
            await asyncio.sleep(backoff_s)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt >= GEN_AI_RETRY_COUNT:
                raise
            backoff_s = GEN_AI_RETRY_BACKOFF_S * (attempt + 1)
            log.warning(
                "reasoning_seq=%s gen-ai transient error (%s), retrying in %.2fs (%s/%s)",
                reasoning_seq,
                _format_async_exception(exc),
                backoff_s,
                attempt + 1,
                GEN_AI_RETRY_COUNT,
            )
            await asyncio.sleep(backoff_s)

    raise RuntimeError("GenAI async retry loop exhausted unexpectedly")


async def _dispatch_reasoning_if_needed(frame_payload: dict, frame_seq: int) -> bool:
    global _reasoning_seq

    if not _is_sampled_frame(frame_seq):
        return False

    async with _state_lock:
        reasoning_seq = _reasoning_seq
        _reasoning_seq += 1
        inflight = len(_reasoning_tasks)

        if inflight >= GEN_AI_MAX_INFLIGHT:
            _reasoning_ready[reasoning_seq] = _build_async_error_payload(
                reasoning_seq,
                (
                    "GenAI async skipped: queue saturated "
                    f"(inflight={inflight}, max={GEN_AI_MAX_INFLIGHT})"
                ),
            )
            log.warning(
                "reasoning_seq=%s skipped due to queue saturation (inflight=%s max=%s)",
                reasoning_seq,
                inflight,
                GEN_AI_MAX_INFLIGHT,
            )
            return False

        _reasoning_tasks[reasoning_seq] = asyncio.create_task(
            _run_gen_ai_in_background(frame_payload, reasoning_seq)
        )

    return True


async def _collect_ordered_reasoning() -> list[dict]:
    global _next_emit_reasoning_seq

    async with _state_lock:
        finished = [
            seq for seq, task in _reasoning_tasks.items()
            if task.done()
        ]
        for seq in sorted(finished):
            task = _reasoning_tasks.pop(seq)
            try:
                _reasoning_ready[seq] = task.result()
            except asyncio.CancelledError as exc:
                _reasoning_ready[seq] = _build_async_error_payload(
                    seq,
                    f"GenAI async cancelled: {_format_async_exception(exc)}",
                )
            except Exception as exc:
                _reasoning_ready[seq] = _build_async_error_payload(
                    seq,
                    f"GenAI async failed: {_format_async_exception(exc)}",
                )
                log.error(
                    "reasoning_seq=%s async failure: %s",
                    seq,
                    _format_async_exception(exc),
                )

        emitted: list[dict] = []
        while _next_emit_reasoning_seq in _reasoning_ready:
            emitted.append(_reasoning_ready.pop(_next_emit_reasoning_seq))
            _next_emit_reasoning_seq += 1

        return emitted


@app.get("/health")
async def health():
    async with _state_lock:
        queued = len(_reasoning_tasks)
        buffered = len(_reasoning_ready)
        next_seq = _next_emit_reasoning_seq

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "node": MY_NODE_NAME,
        "gen_ai_frame_interval": GEN_AI_FRAME_INTERVAL,
        "gen_ai_timeout_s": GEN_AI_TIMEOUT_S,
        "gen_ai_connect_timeout_s": GEN_AI_CONNECT_TIMEOUT_S,
        "gen_ai_retry_count": GEN_AI_RETRY_COUNT,
        "gen_ai_max_inflight": GEN_AI_MAX_INFLIGHT,
        "reasoning_queue_inflight": queued,
        "reasoning_queue_buffered": buffered,
        "reasoning_next_emit_seq": next_seq,
    }


@app.get("/pipeline")
async def pipeline_info():
    """Show current pipeline configuration."""
    return {
        "pipeline": [
            {"name": name, "url": url}
            for name, url in zip(PIPELINE_NAMES, PIPELINE)
        ]
    }


@app.post("/process")
async def process(req: Request):
    global _frame_seq

    data = await req.json()
    t0 = time.perf_counter()
    data["pipeline_start_ms"] = t0 * 1000
    data["gateway_node"] = MY_NODE_NAME

    async with _state_lock:
        frame_seq = _frame_seq
        _frame_seq += 1

    data["frame_seq"] = frame_seq

    log.info(f"frame={data.get('frame_id')} — starting pipeline")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for svc_url, svc_name in [
            (INGEST_URL, "ingest"),
            (PREPROCESS_URL, "preprocess"),
            (DETECTION_URL, "detection"),
        ]:
            try:
                resp = await client.post(f"{svc_url}/process", json=data)
                resp.raise_for_status()
                data = resp.json()
                log.info(f"frame={data.get('frame_id')} — {svc_name} OK")
            except httpx.TimeoutException:
                log.error(f"Timeout calling {svc_name} at {svc_url}")
                REQUESTS.labels(service=SERVICE_NAME, status="timeout").inc()
                raise HTTPException(
                    status_code=504,
                    detail=f"Timeout at stage: {svc_name}"
                )
            except Exception as e:
                log.error(f"Error calling {svc_name} at {svc_url}: {e}")
                REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
                raise HTTPException(
                    status_code=502,
                    detail=f"Pipeline failed at {svc_name}: {str(e)}"
                )

        dispatched_reasoning = await _dispatch_reasoning_if_needed(dict(data), frame_seq)

        try:
            resp = await client.post(f"{POSTPROCESS_URL}/process", json=data)
            resp.raise_for_status()
            data = resp.json()
            log.info(f"frame={data.get('frame_id')} — postprocess OK")
        except httpx.TimeoutException:
            log.error(f"Timeout calling postprocess at {POSTPROCESS_URL}")
            REQUESTS.labels(service=SERVICE_NAME, status="timeout").inc()
            raise HTTPException(
                status_code=504,
                detail="Timeout at stage: postprocess"
            )
        except Exception as e:
            log.error(f"Error calling postprocess at {POSTPROCESS_URL}: {e}")
            REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
            raise HTTPException(
                status_code=502,
                detail=f"Pipeline failed at postprocess: {str(e)}"
            )

    delivered_reasoning = await _collect_ordered_reasoning()
    data["gen_ai_parallel_enabled"] = True
    data["gen_ai_frame_interval"] = GEN_AI_FRAME_INTERVAL
    data["gen_ai_dispatched"] = dispatched_reasoning
    data["reasoning_deliveries"] = delivered_reasoning

    async with _state_lock:
        data["reasoning_queue_inflight"] = len(_reasoning_tasks)
        data["reasoning_queue_buffered"] = len(_reasoning_ready)

    if delivered_reasoning:
        latest = delivered_reasoning[-1]
        data["incident_report"] = latest.get("incident_report", "")
        data["gen_ai_latency_ms"] = latest.get("gen_ai_latency_ms", 0.0)
        data["gen_ai_mode"] = latest.get("gen_ai_mode", "unknown")
        data["gen_ai_model"] = latest.get("gen_ai_model", "unknown")
        data["gen_ai_variant"] = latest.get("gen_ai_variant", "unknown")
        data["gen_ai_sampled"] = latest.get("gen_ai_sampled", True)
        data["gen_ai_node"] = latest.get("gen_ai_node", "unknown")
        data["reasoning_seq"] = latest.get("reasoning_seq")
    elif dispatched_reasoning:
        data["incident_report"] = "Reasoning queued; result will be emitted in-order in a subsequent frame."
        data["gen_ai_mode"] = "queued"

    e2e_ms = (time.perf_counter() - t0) * 1000
    data["e2e_latency_ms"] = e2e_ms

    LATENCY.labels(service=SERVICE_NAME).observe(e2e_ms)
    E2E_LATENCY.observe(e2e_ms)
    REQUESTS.labels(service=SERVICE_NAME, status="success").inc()

    # Fire-and-forget: write latest e2e latency to Redis for monitoring fallback.
    # Dashboard reads kltn:e2e_latency when Prometheus is unavailable.
    if _rdb is not None:
        async def _push_e2e():
            try:
                await _rdb.set(_E2E_REDIS_KEY, str(round(e2e_ms, 2)), ex=_E2E_REDIS_TTL)
            except Exception:
                pass
        asyncio.create_task(_push_e2e())

    log.info(
        f"frame={data.get('frame_id')} — pipeline complete "
        f"e2e={e2e_ms:.1f}ms detections={data.get('detection_count', 0)} "
        f"reasoning_delivered={len(delivered_reasoning)}"
    )
    return JSONResponse(data)
