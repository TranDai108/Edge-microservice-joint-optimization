import os
import time
import logging
import random
import asyncio
import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from prometheus_client import Histogram, Counter, make_asgi_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("gen-ai")

app = FastAPI(title="GenAI Service")

# Metrics
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
REPORT_GEN_LATENCY = Histogram(
    "gen_ai_report_latency_ms",
    "Incident report generation latency in ms",
    ["variant", "mode"],
    buckets=LATENCY_BUCKETS_MS,
)

app.mount("/metrics", make_asgi_app())

SERVICE_NAME = os.getenv("SERVICE_NAME", "gen-ai")
MY_NODE_NAME = os.getenv("MY_NODE_NAME", "unknown")
VARIANT_ID = os.getenv("VARIANT_ID", "qwen-1.5b-nano")
GEN_AI_MODE = os.getenv("GEN_AI_MODE", "mock").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "25"))
GEN_AI_SAMPLING_RATE = max(1, int(os.getenv("GEN_AI_SAMPLING_RATE", "1")))
GEN_AI_FALLBACK_ON_ERROR = os.getenv("GEN_AI_FALLBACK_ON_ERROR", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Default model map used when running in ollama mode.
OLLAMA_MODEL_BY_VARIANT = {
    "qwen-1.5b-nano": "qwen2.5:1.5b",
    "llama-3b-small": "llama3.2:3b",
    "gemma2-2b-medium": "gemma2:2b",
}

# Variant-based mocked SLM latency (ms) for safe baseline MILP testing.
MOCK_VARIANT_LAT_MS = {
    "qwen-1.5b-nano": 150,
    "llama-3b-small": 420,
    "gemma2-2b-medium": 760,
}


def _detected_labels(data: dict) -> list[str]:
    detections = data.get("detections", [])
    labels = []
    for det in detections:
        label = str(det.get("label", "unknown")).strip()
        if label:
            labels.append(label)
    return labels


def _build_prompt(labels: list[str]) -> str:
    detected = ", ".join(labels) if labels else "none"
    return (
        "You are an automated security guard. "
        "Based on the following objects detected in the camera feed, "
        "write a concise 1-sentence security log. "
        f"Detected: {detected}"
    )


def _mock_report(labels: list[str]) -> str:
    if not labels:
        return "No suspicious object detected in this frame."
    top = labels[:4]
    joined = ", ".join(top)
    return f"Camera observed {joined}; continue routine monitoring for unusual behavior."


def _resolve_ollama_model() -> str:
    """Resolve model from VARIANT_ID first; fallback to explicit OLLAMA_MODEL."""
    return OLLAMA_MODEL_BY_VARIANT.get(VARIANT_ID, OLLAMA_MODEL)


def _format_exception(exc: BaseException) -> str:
    """Return useful error detail even for exceptions whose str() is empty."""
    if isinstance(exc, httpx.TimeoutException):
        req = getattr(exc, "request", None)
        target = str(req.url) if req is not None else f"{OLLAMA_URL}/api/generate"
        return (
            f"{type(exc).__name__}: timeout calling {target} "
            f"(timeout={OLLAMA_TIMEOUT_S:.1f}s)"
        )
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        body = response.text if response is not None else ""
        body = " ".join(body.split())[:500]
        return f"{type(exc).__name__}: status={status} body={body}"

    detail = str(exc).strip() or repr(exc)
    return f"{type(exc).__name__}: {detail}"


async def _generate_with_ollama(prompt: str) -> str:
    model_name = _resolve_ollama_model()
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 64,
        },
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_S) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        body = resp.json()
    text = str(body.get("response", "")).strip()
    if not text:
        raise RuntimeError("Empty Ollama response")
    return text


def _should_sample_frame(frame_id: str) -> bool:
    if GEN_AI_SAMPLING_RATE <= 1:
        return True
    # Deterministic sampling by frame id to keep behavior stable across retries.
    return (sum(ord(ch) for ch in frame_id) % GEN_AI_SAMPLING_RATE) == 0


@app.on_event("startup")
async def startup_event():
    log.info(
        "GenAI service started: variant=%s mode=%s model=%s fallback_on_error=%s timeout=%.1fs node=%s",
        VARIANT_ID,
        GEN_AI_MODE,
        _resolve_ollama_model(),
        GEN_AI_FALLBACK_ON_ERROR,
        OLLAMA_TIMEOUT_S,
        MY_NODE_NAME,
    )


@app.on_event("shutdown")
async def shutdown_event():
    log.info(
        "GenAI service shutdown: draining in-flight requests (variant=%s node=%s)",
        VARIANT_ID, MY_NODE_NAME,
    )


@app.get("/health")
async def health(response: Response):
    # Semantic health: if ollama mode is enabled, dependency must be reachable.
    if GEN_AI_MODE == "ollama":
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                tags = await client.get(f"{OLLAMA_URL}/api/tags")
                tags.raise_for_status()
        except Exception as exc:
            response.status_code = 503
            return {
                "status": "dependency_unavailable",
                "service": SERVICE_NAME,
                "mode": GEN_AI_MODE,
                "variant": VARIANT_ID,
                "node": MY_NODE_NAME,
                "detail": _format_exception(exc),
            }

    response.status_code = 200
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "mode": GEN_AI_MODE,
        "variant": VARIANT_ID,
        "node": MY_NODE_NAME,
        "ollama_model": _resolve_ollama_model(),
        "fallback_on_error": GEN_AI_FALLBACK_ON_ERROR,
        "timeout_s": OLLAMA_TIMEOUT_S,
    }


@app.post("/process")
async def process(req: Request):
    data = await req.json()
    t0 = time.perf_counter()

    try:
        frame_id = str(data.get("frame_id", ""))
        sampled = _should_sample_frame(frame_id)
        labels = _detected_labels(data)
        prompt = _build_prompt(labels)
        gen_ai_error = None

        if not sampled:
            report = "GenAI skipped by sampling policy."
            effective_mode = "sampled_out"
            model_name = "skipped"
        elif GEN_AI_MODE == "ollama":
            model_name = _resolve_ollama_model()
            try:
                report = await _generate_with_ollama(prompt)
                effective_mode = GEN_AI_MODE
            except Exception as exc:
                gen_ai_error = _format_exception(exc)
                if not GEN_AI_FALLBACK_ON_ERROR:
                    raise
                log.warning(
                    "Ollama generation failed; returning fallback report: frame=%s variant=%s model=%s error=%s",
                    frame_id,
                    VARIANT_ID,
                    model_name,
                    gen_ai_error,
                )
                report = _mock_report(labels)
                effective_mode = "ollama_fallback"
        else:
            base_ms = MOCK_VARIANT_LAT_MS.get(VARIANT_ID, 250)
            jitter_ms = random.randint(0, 80)
            await asyncio.sleep((base_ms + jitter_ms) / 1000.0)
            report = _mock_report(labels)
            effective_mode = GEN_AI_MODE
            model_name = "mock"

        lat_ms = (time.perf_counter() - t0) * 1000.0

        data["gen_ai_node"] = MY_NODE_NAME
        data["gen_ai_variant"] = VARIANT_ID
        data["gen_ai_mode"] = effective_mode
        data["gen_ai_model"] = model_name
        data["gen_ai_sampled"] = sampled
        data["gen_ai_sampling_rate"] = GEN_AI_SAMPLING_RATE
        data["gen_ai_prompt"] = prompt
        data["incident_report"] = report
        data["gen_ai_latency_ms"] = lat_ms
        if gen_ai_error:
            data["gen_ai_error"] = gen_ai_error

        LATENCY.labels(service=SERVICE_NAME).observe(lat_ms)
        REPORT_GEN_LATENCY.labels(variant=VARIANT_ID, mode=effective_mode).observe(lat_ms)
        if not sampled:
            status = "skipped"
        elif gen_ai_error:
            status = "fallback"
        else:
            status = "success"
        REQUESTS.labels(service=SERVICE_NAME, status=status).inc()

        log.info(
            "frame=%s variant=%s mode=%s sampled=%s labels=%d lat=%.1fms",
            frame_id,
            VARIANT_ID,
            effective_mode,
            sampled,
            len(labels),
            lat_ms,
        )
        return data

    except HTTPException:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        raise
    except Exception as exc:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        detail = _format_exception(exc)
        log.exception("Unexpected error in gen-ai: %s", detail)
        raise HTTPException(status_code=500, detail=detail)
