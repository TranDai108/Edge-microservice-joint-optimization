FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY src/milp_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Core app files
COPY src/milp_agent/milp_agent.py /app/milp_agent.py
COPY src/milp_agent/redis_state.py /app/redis_state.py

# Shared project modules
COPY src/solver /app/solver
COPY src/config.py /app/config.py
COPY src/k8s_client.py /app/k8s_client.py
COPY src/variant_catalog.py /app/variant_catalog.py
COPY src/variant_controller /app/variant_controller
# HA package (leader election + sentinel-aware redis client)
COPY src/ha /app/ha

ENV PYTHONPATH=/app:/app/solver

EXPOSE 8080

CMD ["python", "milp_agent.py"]
