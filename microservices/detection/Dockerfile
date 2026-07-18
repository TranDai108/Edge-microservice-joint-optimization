FROM python:3.11-slim
WORKDIR /app

# Install system dependency required by opencv (pulled in by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install ultralytics with headless opencv to avoid the libGL issue
RUN pip install --no-cache-dir \
    "ultralytics" \
    "opencv-python-headless" \
    pillow fastapi uvicorn requests prometheus-client

ARG MODEL_FILE=yolo26n.pt
ARG VARIANT_ID=yolo26-nano
ENV MODEL_PATH=/app/${MODEL_FILE}
ENV VARIANT_ID=${VARIANT_ID}
ENV ALLOW_MODEL_DISCOVERY=false

# Download model natively via Python
RUN python -c "from ultralytics import YOLO; YOLO('${MODEL_FILE}')"

COPY app.py .

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
