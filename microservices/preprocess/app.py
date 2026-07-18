import os
import time
import logging
import base64
import io
from fastapi import FastAPI, Request, HTTPException
from prometheus_client import Histogram, Counter, make_asgi_app
from PIL import Image, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("preprocess")

app = FastAPI(title="Preprocess Service")

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
RESIZE_RATIO = Histogram(
    "preprocess_resize_ratio",
    "Ratio of original to resized image area",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0]
)

app.mount("/metrics", make_asgi_app())

SERVICE_NAME  = os.getenv("SERVICE_NAME",  "preprocess")
MY_NODE_NAME  = os.getenv("MY_NODE_NAME",  "unknown")
TARGET_WIDTH  = int(os.getenv("TARGET_WIDTH",  "640"))
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT", "640"))
JPEG_QUALITY  = int(os.getenv("JPEG_QUALITY",  "85"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "node": MY_NODE_NAME,
        "target_size": f"{TARGET_WIDTH}x{TARGET_HEIGHT}"
    }


@app.post("/process")
async def process(req: Request):
    data = await req.json()
    t0 = time.perf_counter()

    try:
        if "image_b64" not in data:
            raise HTTPException(status_code=400, detail="Missing image_b64")

        # ── Decode ──
        img_bytes = base64.b64decode(data["image_b64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        # ── Resize to YOLO input size (letterbox to preserve aspect ratio) ──
        img = ImageOps.fit(img, (TARGET_WIDTH, TARGET_HEIGHT), method=Image.LANCZOS)

        # ── Re-encode ──
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        new_bytes = buf.getvalue()
        data["image_b64"] = base64.b64encode(new_bytes).decode()

        # ── Attach preprocessing metadata ──
        orig_area = orig_w * orig_h
        new_area  = TARGET_WIDTH * TARGET_HEIGHT
        ratio     = new_area / orig_area if orig_area > 0 else 1.0

        data["preprocessed"]        = True
        data["preprocess_node"]     = MY_NODE_NAME
        data["original_size"]       = [orig_w, orig_h]
        data["input_size"]          = [TARGET_WIDTH, TARGET_HEIGHT]
        data["preprocess_latency_ms"] = (time.perf_counter() - t0) * 1000

        RESIZE_RATIO.observe(ratio)
        LATENCY.labels(service=SERVICE_NAME).observe(data["preprocess_latency_ms"])
        REQUESTS.labels(service=SERVICE_NAME, status="success").inc()

        log.info(
            f"frame={data.get('frame_id')} "
            f"orig={orig_w}x{orig_h} → {TARGET_WIDTH}x{TARGET_HEIGHT} "
            f"lat={data['preprocess_latency_ms']:.1f}ms"
        )
        return data

    except HTTPException:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        raise
    except Exception as e:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        log.error(f"Unexpected error in preprocess: {e}")
        raise HTTPException(status_code=500, detail=str(e))
