import os
import time
import logging
import base64
import io
from fastapi import FastAPI, Request, HTTPException
from prometheus_client import Histogram, Counter, make_asgi_app
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("ingest")

app = FastAPI(title="Ingest Service")

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
FRAME_SIZE = Histogram(
    "frame_size_bytes",
    "Size of incoming frame in bytes",
    buckets=[1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

app.mount("/metrics", make_asgi_app())

SERVICE_NAME = os.getenv("SERVICE_NAME", "ingest")
MY_NODE_NAME = os.getenv("MY_NODE_NAME", "unknown")


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "node": MY_NODE_NAME}


@app.post("/process")
async def process(req: Request):
    data = await req.json()
    t0 = time.perf_counter()

    try:
        # ── Validate required fields ──
        if "frame_id" not in data:
            raise HTTPException(status_code=400, detail="Missing frame_id")
        if "image_b64" not in data:
            raise HTTPException(status_code=400, detail="Missing image_b64")

        # ── Decode and validate image ──
        try:
            img_bytes = base64.b64decode(data["image_b64"])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        try:
            img = Image.open(io.BytesIO(img_bytes))
            width, height = img.size
            img_format = img.format or "JPEG"
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot decode image")

        # ── Attach ingest metadata ──
        data["ingest_ok"]       = True
        data["source_node"]     = MY_NODE_NAME
        data["image_width"]     = width
        data["image_height"]    = height
        data["image_format"]    = img_format
        data["image_size_bytes"] = len(img_bytes)
        data["ingest_latency_ms"] = (time.perf_counter() - t0) * 1000

        FRAME_SIZE.observe(len(img_bytes))
        LATENCY.labels(service=SERVICE_NAME).observe(data["ingest_latency_ms"])
        REQUESTS.labels(service=SERVICE_NAME, status="success").inc()

        log.info(
            f"frame={data['frame_id']} "
            f"size={len(img_bytes)}B "
            f"dim={width}x{height} "
            f"lat={data['ingest_latency_ms']:.1f}ms"
        )
        return data

    except HTTPException:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        raise
    except Exception as e:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        log.error(f"Unexpected error in ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))
