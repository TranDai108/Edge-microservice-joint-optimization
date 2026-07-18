import os
import asyncio
from fastapi import FastAPI, Request

app = FastAPI(title="Mock Stage")
STAGE = os.getenv("STAGE", "ingest")
LAT_MS = float(os.getenv("LAT_MS", "5"))


@app.get("/health")
async def health():
    return {"status": "ok", "stage": STAGE, "lat_ms": LAT_MS}


@app.post("/process")
async def process(req: Request):
    data = await req.json()
    await asyncio.sleep(LAT_MS / 1000.0)

    if STAGE == "ingest":
        data["ingest_ok"] = True
        data["ingest_latency_ms"] = LAT_MS
    elif STAGE == "preprocess":
        data["preprocessed"] = True
        data["preprocess_latency_ms"] = LAT_MS
    elif STAGE == "detection":
        data["detections"] = [
            {"label": "car", "conf": 0.9},
            {"label": "person", "conf": 0.8},
        ]
        data["detection_count"] = 2
        data["accuracy_proxy"] = 0.85
        data["variant"] = "yolo26-nano"
        data["detection_latency_ms"] = LAT_MS
    return data
