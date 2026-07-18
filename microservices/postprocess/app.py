import os
import time
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Histogram, Counter, make_asgi_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("postprocess")

app = FastAPI(title="Postprocess Service")

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
PIPELINE_LATENCY = Histogram(
    "pipeline_total_latency_ms",
    "Total latency across all pipeline stages",
    buckets=LATENCY_BUCKETS_MS,
)
OBJECTS_PER_CLASS = Counter(
    "detected_objects_total",
    "Total detected objects by class label",
    ["label"]
)

app.mount("/metrics", make_asgi_app())

SERVICE_NAME = os.getenv("SERVICE_NAME", "postprocess")
MY_NODE_NAME = os.getenv("MY_NODE_NAME", "unknown")


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "node": MY_NODE_NAME}


@app.post("/process")
async def process(req: Request):
    data = await req.json()
    t0 = time.perf_counter()

    try:
        detections = data.get("detections", [])

        # ── Aggregate detection results ──
        class_counts = {}
        for det in detections:
            label = det.get("label", "unknown")
            class_counts[label] = class_counts.get(label, 0) + 1
            OBJECTS_PER_CLASS.labels(label=label).inc()

        # ── Compute per-stage latency summary ──
        stage_latencies = {
            "ingest_ms":      data.get("ingest_latency_ms", 0),
            "preprocess_ms":  data.get("preprocess_latency_ms", 0),
            "detection_ms":   data.get("detection_latency_ms", 0),
            "gen_ai_ms":      data.get("gen_ai_latency_ms", 0),
            "postprocess_ms": 0,   # filled in below
        }

        # ── Build clean response — strip raw image from payload ──
        data.pop("image_b64", None)

        data["postprocess_node"]  = MY_NODE_NAME
        data["object_classes"]    = list(class_counts.keys())
        data["class_counts"]      = class_counts
        data["postprocess_latency_ms"] = (time.perf_counter() - t0) * 1000
        stage_latencies["postprocess_ms"] = data["postprocess_latency_ms"]
        data["stage_latencies"]   = stage_latencies

        # ── Pipeline total (sum of stages, not wall clock) ──
        pipeline_total = sum(stage_latencies.values())
        data["pipeline_stage_total_ms"] = pipeline_total

        PIPELINE_LATENCY.observe(pipeline_total)
        LATENCY.labels(service=SERVICE_NAME).observe(data["postprocess_latency_ms"])
        REQUESTS.labels(service=SERVICE_NAME, status="success").inc()

        log.info(
            f"frame={data.get('frame_id')} "
            f"classes={data['object_classes']} "
            f"acc={data.get('accuracy_proxy', 0):.3f} "
            f"pipeline_total={pipeline_total:.1f}ms "
            f"variant={data.get('variant', 'unknown')}"
        )
        return JSONResponse(data)

    except HTTPException:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        raise
    except Exception as e:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        log.error(f"Unexpected error in postprocess: {e}")
        raise HTTPException(status_code=500, detail=str(e))
