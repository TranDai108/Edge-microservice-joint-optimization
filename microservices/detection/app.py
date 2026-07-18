import os
import time
import logging
import base64
import io
from fastapi import FastAPI, Request, HTTPException, Response
from prometheus_client import Histogram, Counter, Gauge, make_asgi_app
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("detection")

app = FastAPI(title="Detection Service")

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
ACCURACY = Histogram(
    "detection_accuracy",
    "Mean confidence score per frame",
    ["variant"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
DETECTION_COUNT = Histogram(
    "detection_object_count",
    "Number of objects detected per frame",
    ["variant"],
    buckets=[0, 1, 2, 5, 10, 20, 50]
)
MODEL_LOADED = Gauge(
    "detection_model_loaded",
    "Whether the YOLO model is loaded successfully"
)

app.mount("/metrics", make_asgi_app())

SERVICE_NAME = os.getenv("SERVICE_NAME", "detection")
MY_NODE_NAME = os.getenv("MY_NODE_NAME", "unknown")
MODEL_PATH_CFG = os.getenv("MODEL_PATH", "/app/model.pt")
ALLOW_MODEL_DISCOVERY = os.getenv("ALLOW_MODEL_DISCOVERY", "false").lower() == "true"
VARIANT_ID   = os.getenv("VARIANT_ID",   "yolo26-nano")
CONF_THRESH  = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESH   = float(os.getenv("IOU_THRESHOLD",  "0.45"))


def _resolve_model_path(configured_path: str) -> str:
    if configured_path and os.path.exists(configured_path):
        return configured_path

    if not ALLOW_MODEL_DISCOVERY:
        log.error(
            "Configured MODEL_PATH does not exist and model discovery is disabled. "
            "Set ALLOW_MODEL_DISCOVERY=true only for explicit debugging use."
        )
        return configured_path

    # Optional debugging escape hatch: explicit opt-in fallback discovery.
    for path in sorted([p for p in os.listdir("/app") if p.endswith(".pt")]):
        candidate = f"/app/{path}"
        if os.path.exists(candidate):
            log.warning(
                f"Configured MODEL_PATH not found: {configured_path}. "
                f"Using discovered model file: {candidate}"
            )
            return candidate

    return configured_path


MODEL_PATH = _resolve_model_path(MODEL_PATH_CFG)

# ── Load model at startup ──
model = None
try:
    from ultralytics import YOLO
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file does not exist: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    MODEL_LOADED.set(1)
    log.info(f"Model loaded: {MODEL_PATH} (variant={VARIANT_ID})")
except Exception as e:
    MODEL_LOADED.set(0)
    log.error(
        f"Failed to load model (configured={MODEL_PATH_CFG}, resolved={MODEL_PATH}): {e}"
    )


@app.on_event("startup")
async def startup_event():
    log.info(
        "Detection service started: variant=%s node=%s model_loaded=%s",
        VARIANT_ID, MY_NODE_NAME, model is not None,
    )


@app.on_event("shutdown")
async def shutdown_event():
    log.info(
        "Detection service shutdown: draining in-flight requests (variant=%s node=%s)",
        VARIANT_ID, MY_NODE_NAME,
    )


@app.get("/health")
async def health(response: Response):
    model_ready = model is not None
    response.status_code = 200 if model_ready else 503
    return {
        "status": "ok" if model_ready else "model_not_loaded",
        "service": SERVICE_NAME,
        "variant": VARIANT_ID,
        "node": MY_NODE_NAME,
        "configured_model_path": MODEL_PATH_CFG,
        "resolved_model_path": MODEL_PATH,
        "conf_threshold": CONF_THRESH,
    }


@app.post("/process")
async def detect(req: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = await req.json()
    t0 = time.perf_counter()

    try:
        if "image_b64" not in data:
            raise HTTPException(status_code=400, detail="Missing image_b64")

        # ── Decode image ──
        img_bytes = base64.b64decode(data["image_b64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ── Run inference ──
        results = model(
            img,
            verbose=False,
            conf=CONF_THRESH,
            iou=IOU_THRESH
        )[0]

        # ── Extract detections ──
        boxes = []
        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                boxes.append({
                    "xyxy":  box.xyxy[0].tolist(),
                    "conf":  float(box.conf[0]),
                    "cls":   int(box.cls[0]),
                    "label": model.names[int(box.cls[0])]
                })

        conf_values = [b["conf"] for b in boxes]
        mean_conf   = sum(conf_values) / len(conf_values) if conf_values else 0.0

        # ── Attach detection metadata ──
        data["detections"]           = boxes
        data["detection_count"]      = len(boxes)
        data["accuracy_proxy"]       = mean_conf
        data["variant"]              = VARIANT_ID
        data["detection_node"]       = MY_NODE_NAME
        data["detection_latency_ms"] = (time.perf_counter() - t0) * 1000

        ACCURACY.labels(variant=VARIANT_ID).observe(mean_conf)
        DETECTION_COUNT.labels(variant=VARIANT_ID).observe(len(boxes))
        LATENCY.labels(service=SERVICE_NAME).observe(data["detection_latency_ms"])
        REQUESTS.labels(service=SERVICE_NAME, status="success").inc()

        log.info(
            f"frame={data.get('frame_id')} "
            f"variant={VARIANT_ID} "
            f"detections={len(boxes)} "
            f"acc={mean_conf:.3f} "
            f"lat={data['detection_latency_ms']:.1f}ms"
        )
        return data

    except HTTPException:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        raise
    except Exception as e:
        REQUESTS.labels(service=SERVICE_NAME, status="error").inc()
        log.error(f"Unexpected error in detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
