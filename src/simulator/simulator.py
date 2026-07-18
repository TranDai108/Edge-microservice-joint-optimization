"""
client/simulator.py

Sends video frames to the api-gateway at a controlled FPS rate and logs
per-frame results to a JSONL file for thesis analysis.

Config loaded from client/config.env (or environment variables):
  GATEWAY_URL   — e.g. http://localhost:30080
  VIDEO_SOURCE  — path to video file, or "0" for webcam
  TARGET_FPS    — target send rate (default 15)
  RESULTS_FILE  — output JSONL path
  MAX_FRAMES    — stop after N frames (0 = unlimited)
  TIMEOUT_SEC   — per-request timeout in seconds (default 15)
  DISPLAY_VIDEO — show live window (True/False)
  SLA_MS        — SLA violation threshold in ms
  ENCODE_WIDTH  — width to resize frames before sending (default 640)
  SHOW_DEBUG    — show resolution overlay on screen (default True)

ROOT CAUSE OF BOUNDING BOX DRIFT (fixed here):
  The YOLO server resizes every incoming image to 640xN before inference,
  then returns xyxy coordinates in that resized space. If the client sends
  a 1280x720 frame but draws the returned coords on the original 1280x720
  display frame, everything appears at half-scale — shifted to the top-left.

SOLUTION:
  1. Client resizes frame to ENCODE_WIDTH (640) before encoding/sending.
  2. Server receives 640xN, runs inference, returns coords in 640xN space. ✓
  3. Client rescales coords back up to display resolution before drawing.
  This gives pixel-perfect alignment regardless of server model input size.
"""

import cv2
import base64
import requests
import time
import json
import logging
import os
import sys
import signal
import threading
import queue
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import dotenv

# ── Load config ───────────────────────────────────────────────────────────────
config_path = Path(__file__).parent / "config.env"
if config_path.exists():
    dotenv.load_dotenv(config_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("simulator")
pipeline_log = logging.getLogger("simulator.pipeline")
reasoning_log = logging.getLogger("simulator.reasoning")

# ── Config ────────────────────────────────────────────────────────────────────
GATEWAY_URL   = os.getenv("GATEWAY_URL",  "http://localhost:30080")
VIDEO_SOURCE  = os.getenv("VIDEO_SOURCE", "0")
TARGET_FPS    = float(os.getenv("TARGET_FPS",   "15"))
RESULTS_FILE  = os.getenv(
    "RESULTS_FILE",
    str(Path(__file__).parent / "client_results.jsonl")
)
MAX_FRAMES    = int(os.getenv("MAX_FRAMES",  "0"))
TIMEOUT_SEC   = float(os.getenv("TIMEOUT_SEC", "15"))
SLA_MS        = float(os.getenv("SLA_MS", "1500"))
STATS_EVERY   = int(os.getenv("STATS_EVERY", "50"))
DISPLAY_VIDEO = os.getenv("DISPLAY_VIDEO", "False").lower() in ("true", "1", "yes")
SHOW_DEBUG    = os.getenv("SHOW_DEBUG", "True").lower() in ("true", "1", "yes")

# THE KEY FIX: must match your server's model input width.
# Standard YOLO (v5/v8/v11) resizes input to 640px wide before inference.
# Set to 0 to send at native resolution (only if your server returns coords
# already in native resolution space, which most servers do NOT).
ENCODE_WIDTH  = int(os.getenv("ENCODE_WIDTH", "640"))


# ── Shared state ──────────────────────────────────────────────────────────────
_shutdown = False
_frame_lock = threading.Lock()


@dataclass
class DetectionSnapshot:
    """Bundles detection results with the exact resolution they were computed for."""
    detections: list = field(default_factory=list)
    sent_w: int = 640
    sent_h: int = 360


_latest_snapshot: Optional[DetectionSnapshot] = None


def _handle_sigint(sig, frame):
    global _shutdown
    log.info("Interrupt — shutting down gracefully...")
    _shutdown = True

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# ── Frame encode helpers ──────────────────────────────────────────────────────

def prepare_frame_for_send(frame):
    """
    Resize frame to ENCODE_WIDTH before sending to server.
    Returns (resized_frame, sent_w, sent_h).

    Sending at the server model's input resolution means the returned xyxy
    coordinates are directly in that resolution's coordinate space, making
    rescaling to the display frame trivial and accurate.
    """
    orig_h, orig_w = frame.shape[:2]
    if ENCODE_WIDTH > 0 and orig_w != ENCODE_WIDTH:
        scale   = ENCODE_WIDTH / orig_w
        sent_w  = ENCODE_WIDTH
        sent_h  = int(orig_h * scale)
        resized = cv2.resize(frame, (sent_w, sent_h), interpolation=cv2.INTER_AREA)
        return resized, sent_w, sent_h
    return frame, orig_w, orig_h


def encode_frame(frame) -> str:
    """Encode a cv2 BGR frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


# ── Coordinate helpers ────────────────────────────────────────────────────────

def parse_xyxy(raw) -> Optional[tuple[int, int, int, int]]:
    """
    Safely parse xyxy from multiple server response formats:
      flat list/tuple : [x1, y1, x2, y2]
      nested list     : [[x1, y1, x2, y2]]
      dict            : {"x1":…, "y1":…, "x2":…, "y2":…}
    """
    try:
        if isinstance(raw, (list, tuple)):
            inner = raw[0] if (len(raw) == 1 and isinstance(raw[0], (list, tuple))) else raw
            return (int(float(inner[0])), int(float(inner[1])),
                    int(float(inner[2])), int(float(inner[3])))
        if isinstance(raw, dict):
            return (int(float(raw["x1"])), int(float(raw["y1"])),
                    int(float(raw["x2"])), int(float(raw["y2"])))
    except Exception as exc:
        log.warning(f"parse_xyxy failed for {raw!r}: {exc}")
    return None


def rescale_coords(
    x1: int, y1: int, x2: int, y2: int,
    sent_w: int, sent_h: int,
    disp_w: int, disp_h: int,
) -> tuple[int, int, int, int]:
    """
    Scale box coordinates from sent-frame space → display-frame space.

    Example:
      Sent frame = 640x360, display frame = 1280x720, scale = 2.0x
      Server returns corner (100, 50) → display corner (200, 100) ✓
    """
    if sent_w == disp_w and sent_h == disp_h:
        return x1, y1, x2, y2
    sx = disp_w / sent_w
    sy = disp_h / sent_h
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_detections(frame, snapshot: DetectionSnapshot) -> None:
    """Draw bounding boxes onto the display frame, rescaling from sent resolution."""
    disp_h, disp_w = frame.shape[:2]

    for box in snapshot.detections:
        coords = parse_xyxy(box.get("xyxy"))
        if coords is None:
            continue

        x1, y1, x2, y2 = rescale_coords(
            *coords,
            sent_w=snapshot.sent_w, sent_h=snapshot.sent_h,
            disp_w=disp_w,         disp_h=disp_h,
        )

        label = box.get("label", "?")
        conf  = float(box.get("conf", 0.0))
        text  = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label with filled background for readability
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame,
                      (x1, label_y - th - 4),
                      (x1 + tw + 4, label_y + baseline),
                      (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_debug_overlay(frame, snapshot: Optional[DetectionSnapshot]) -> None:
    """Show resolution info on-screen to verify coordinate alignment."""
    h, w = frame.shape[:2]
    scale_str = f"{w / snapshot.sent_w:.2f}x" if snapshot else "--"
    lines = [
        f"Display : {w}x{h}",
        f"Sent    : {snapshot.sent_w}x{snapshot.sent_h}" if snapshot else "Sent: --",
        f"Scale   : {scale_str}",
        f"EncodeW : {ENCODE_WIDTH}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)


# ── Networking helpers ────────────────────────────────────────────────────────

def wait_for_gateway(max_retries: int = 10, delay: float = 3.0) -> bool:
    log.info(f"Waiting for gateway at {GATEWAY_URL}...")
    for attempt in range(1, max_retries + 1):
        for path in ("/health", "/"):
            try:
                resp = requests.get(f"{GATEWAY_URL}{path}", timeout=3)
                if resp.status_code < 500:
                    log.info(f"Gateway reachable via {path} (HTTP {resp.status_code})")
                    return True
            except Exception:
                pass
        log.info(f"Attempt {attempt}/{max_retries} failed, retrying in {delay}s...")
        time.sleep(delay)
    log.error("Gateway not reachable. Exiting.")
    return False


def open_video_source() -> cv2.VideoCapture:
    source = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open video source: {VIDEO_SOURCE!r}")
        sys.exit(1)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"Video opened: {VIDEO_SOURCE!r}  native={cam_w}x{cam_h}  FPS={fps:.1f}  frames={total}")
    if ENCODE_WIDTH > 0:
        log.info(f"Encoding at: {ENCODE_WIDTH}px wide  (scale factor = {cam_w / ENCODE_WIDTH:.2f}x will be applied to returned coords)")
    else:
        log.info("Encoding at native resolution (ENCODE_WIDTH=0)")
    return cap


# ── Stats helpers ─────────────────────────────────────────────────────────────

def log_rolling_stats(window: deque, sla_violations: int, total_frames: int):
    if not window:
        return
    rtts = [r["rtt"]  for r in window if "rtt"  in r]
    e2es = [r["e2e"]  for r in window if "e2e"  in r]
    accs = [r["acc"]  for r in window if "acc"  in r]
    errs = sum(1 for r in window if r.get("error"))
    if rtts and e2es and accs:
        log.info(
            f"[stats/{total_frames}]  "
            f"RTT={sum(rtts)/len(rtts):.0f}ms  "
            f"e2e={sum(e2es)/len(e2es):.0f}ms  "
            f"acc={sum(accs)/len(accs):.3f}  "
            f"err={errs}  SLA_viol={sla_violations}"
        )
    else:
        log.info(f"[stats/{total_frames}] no successful frames in window  err={errs}")


def print_summary(all_stats: list, sla_violations: int):
    ok    = [r for r in all_stats if not r.get("error")]
    rtts  = [r["rtt"] for r in ok if "rtt" in r]
    e2es  = [r["e2e"] for r in ok if "e2e" in r]
    accs  = [r["acc"] for r in ok if "acc" in r]
    vars_ = sorted({r["variant"] for r in ok if "variant" in r})
    log.info("=" * 55)
    log.info(f"SUMMARY  total={len(all_stats)}  ok={len(ok)}  err={len(all_stats)-len(ok)}")
    log.info(f"  SLA violations : {sla_violations}  (e2e > {SLA_MS:.0f}ms)")
    log.info(f"  Variants seen  : {vars_}")
    if rtts:
        log.info(f"  RTT  avg/min/max : {sum(rtts)/len(rtts):.1f}/{min(rtts):.1f}/{max(rtts):.1f} ms")
    if e2es:
        log.info(f"  E2E  avg/min/max : {sum(e2es)/len(e2es):.1f}/{min(e2es):.1f}/{max(e2es):.1f} ms")
    if accs:
        log.info(f"  Acc  avg         : {sum(accs)/len(accs):.4f}")
    log.info("=" * 55)


# ── Network worker thread ─────────────────────────────────────────────────────

def network_worker(frame_queue: queue.Queue, results_file: str):
    """
    Consumer thread:
      1. Pull latest frame from queue
      2. Resize to ENCODE_WIDTH (server model input resolution)
      3. POST to gateway, get detections back
      4. Store DetectionSnapshot(detections, sent_w, sent_h) for UI thread
    """
    global _latest_snapshot, _shutdown

    sla_violations = 0
    frame_id       = 0
    last_variant   = None
    rolling        = deque(maxlen=STATS_EVERY)
    all_stats: list = []

    dir_name = os.path.dirname(results_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(results_file, "a") as out:
        while not _shutdown:
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Resize before sending — coords returned by server will be in this space
            encoded_frame, sent_w, sent_h = prepare_frame_for_send(frame)

            payload = {"frame_id": frame_id, "image_b64": encode_frame(encoded_frame)}
            record  = {"frame_id": frame_id, "timestamp": time.time(),
                       "sent_w": sent_w, "sent_h": sent_h}
            stat    = {"frame_id": frame_id}

            try:
                t0   = time.perf_counter()
                resp = requests.post(f"{GATEWAY_URL}/process", json=payload, timeout=TIMEOUT_SEC)
                resp.raise_for_status()
                result = resp.json()
                rtt_ms = (time.perf_counter() - t0) * 1000

                result["client_rtt_ms"] = rtt_ms
                record.update(result)

                e2e     = result.get("e2e_latency_ms", 0.0)
                variant = result.get("variant", "?")
                acc     = result.get("accuracy_proxy", 0.0)
                det     = result.get("detection_count", 0)
                gen_mode = result.get("gen_ai_mode", "N/A")
                gen_sampled = result.get("gen_ai_sampled", None)
                gen_ms = float(result.get("gen_ai_latency_ms", 0.0) or 0.0)
                gen_report = str(result.get("incident_report", ""))
                gen_report_short = gen_report[:80] + ("..." if len(gen_report) > 80 else "")
                gen_dispatched = bool(result.get("gen_ai_dispatched", False))
                reasoning_deliveries = result.get("reasoning_deliveries", []) or []
                stat.update({"rtt": rtt_ms, "e2e": e2e, "acc": acc, "variant": variant})

                reasoning_events = []
                for delivered in reasoning_deliveries:
                    reasoning_events.append({
                        "event": "delivered",
                        "timestamp": time.time(),
                        "source_frame_id": frame_id,
                        "reasoning_seq": delivered.get("reasoning_seq"),
                        "frame_id": delivered.get("frame_id"),
                        "gen_ai_mode": delivered.get("gen_ai_mode", "unknown"),
                        "gen_ai_variant": delivered.get("gen_ai_variant", "unknown"),
                        "gen_ai_model": delivered.get("gen_ai_model", "unknown"),
                        "gen_ai_latency_ms": float(delivered.get("gen_ai_latency_ms", 0.0) or 0.0),
                        "incident_report": str(delivered.get("incident_report", "")),
                    })

                if gen_dispatched or gen_mode in ("queued", "sampled_out", "error"):
                    reasoning_events.append({
                        "event": "status",
                        "timestamp": time.time(),
                        "source_frame_id": frame_id,
                        "reasoning_seq": None,
                        "frame_id": frame_id,
                        "gen_ai_mode": gen_mode,
                        "gen_ai_variant": result.get("gen_ai_variant", "unknown"),
                        "gen_ai_model": result.get("gen_ai_model", "unknown"),
                        "gen_ai_latency_ms": gen_ms,
                        "incident_report": gen_report,
                        "gen_ai_dispatched": gen_dispatched,
                    })

                if reasoning_events:
                    record["reasoning_events"] = reasoning_events

                # Store snapshot tied to the resolution we actually sent
                with _frame_lock:
                    _latest_snapshot = DetectionSnapshot(
                        detections=result.get("detections", []),
                        sent_w=sent_w,
                        sent_h=sent_h,
                    )

                sla_tag = ""
                if e2e > SLA_MS:
                    sla_violations += 1
                    sla_tag = f"  ⚠ SLA ({e2e:.0f}ms > {SLA_MS:.0f}ms)"

                if variant != last_variant and last_variant is not None:
                    log.info("=" * 55)
                    log.info(f"  🔄 VARIANT CHANGED: {last_variant} → {variant}  (frame {frame_id})")
                    log.info("=" * 55)
                last_variant = variant

                pipeline_log.info(
                    f"frame={frame_id:05d}  rtt={rtt_ms:.1f}ms  e2e={e2e:.1f}ms  "
                    f"det={det}  variant={variant}  acc={acc:.3f}  "
                    f"gen_mode={gen_mode}  gen_sampled={gen_sampled}  gen_ms={gen_ms:.1f}" + sla_tag
                )

                if gen_report_short and gen_mode in ("queued", "sampled_out", "error"):
                    reasoning_log.info(
                        f"frame={frame_id:05d}  event=status  mode={gen_mode}  "
                        f"dispatched={gen_dispatched}  report=\"{gen_report_short}\""
                    )

                for delivered in reasoning_deliveries:
                    delivered_report = str(delivered.get("incident_report", ""))
                    delivered_report_short = delivered_report[:80] + ("..." if len(delivered_report) > 80 else "")
                    reasoning_log.info(
                        f"frame={frame_id:05d}  event=delivered  reasoning_seq={delivered.get('reasoning_seq')}  "
                        f"mode={delivered.get('gen_ai_mode', 'unknown')}  latency={float(delivered.get('gen_ai_latency_ms', 0.0) or 0.0):.1f}ms  "
                        f"report=\"{delivered_report_short}\""
                    )

            except requests.Timeout:
                log.warning(f"frame={frame_id:05d}  TIMEOUT after {TIMEOUT_SEC}s")
                record["error"] = stat["error"] = "timeout"
            except requests.HTTPError as exc:
                log.warning(f"frame={frame_id:05d}  HTTP {exc}")
                record["error"] = stat["error"] = str(exc)
            except Exception as exc:
                log.warning(f"frame={frame_id:05d}  error: {exc}")
                record["error"] = stat["error"] = str(exc)

            out.write(json.dumps(record) + "\n")
            out.flush()
            rolling.append(stat)
            all_stats.append(stat)
            frame_id += 1

            if frame_id % STATS_EVERY == 0:
                log_rolling_stats(rolling, sla_violations, frame_id)
            if MAX_FRAMES > 0 and frame_id >= MAX_FRAMES:
                log.info(f"Reached MAX_FRAMES={MAX_FRAMES}. Stopping.")
                _shutdown = True
                break

    print_summary(all_stats, sla_violations)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global _shutdown

    if not wait_for_gateway():
        sys.exit(1)

    cap            = open_video_source()
    frame_interval = 1.0 / TARGET_FPS
    frame_queue    = queue.Queue(maxsize=1)

    worker_thread = threading.Thread(
        target=network_worker,
        args=(frame_queue, RESULTS_FILE),
        daemon=True,
    )
    worker_thread.start()

    log.info(
        f"Running — fps={TARGET_FPS}  max={MAX_FRAMES or 'unlimited'}  "
        f"SLA={SLA_MS:.0f}ms  encode_width={ENCODE_WIDTH or 'native'}  "
        f"debug_overlay={SHOW_DEBUG}"
    )

    last_sent_time = 0.0

    display_enabled = DISPLAY_VIDEO
    display_probe_done = False

    while cap.isOpened() and not _shutdown:
        # Drain camera hardware buffer continuously
        ret, frame = cap.read()
        if not ret:
            if VIDEO_SOURCE.isdigit():
                log.warning("Camera read failed. Retrying...")
                time.sleep(0.5)
                continue
            else:
                log.info("Video ended — looping.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        now = time.perf_counter()

        # Feed freshest frame to network worker at TARGET_FPS rate
        if now - last_sent_time >= frame_interval:
            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(frame.copy())
                last_sent_time = now
            except queue.Full:
                pass

        # UI runs at full camera FPS — draw latest boxes rescaled to display res
        if display_enabled:
            if not display_probe_done:
                try:
                    cv2.namedWindow("Edge AI Live Camera", cv2.WINDOW_NORMAL)
                except cv2.error as exc:
                    log.warning(
                        "DISPLAY_VIDEO=True but OpenCV GUI is unavailable (%s). Falling back to headless mode.",
                        exc,
                    )
                    display_enabled = False
                finally:
                    display_probe_done = True

        if display_enabled:
            with _frame_lock:
                snapshot = _latest_snapshot  # immutable dataclass, thread-safe read

            if snapshot is not None:
                draw_detections(frame, snapshot)
            if SHOW_DEBUG:
                draw_debug_overlay(frame, snapshot)

            cv2.imshow("Edge AI Live Camera", frame)
            key = cv2.waitKey(1)
            if key in (27, ord('q')):
                _shutdown = True
                break
        else:
            if not VIDEO_SOURCE.isdigit():
                time.sleep(1.0 / 30.0)

    _shutdown = True
    cap.release()
    if display_enabled:
        cv2.destroyAllWindows()
    worker_thread.join(timeout=3.0)
    log.info("Simulator exited.")


if __name__ == "__main__":
    main()
