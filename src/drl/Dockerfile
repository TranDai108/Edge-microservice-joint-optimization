FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (avoids pulling the large CUDA wheel)
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/drl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Entry-point script
COPY src/drl/drl_agent.py /app/drl_agent.py

# DRL package (digital_twin, edge_env, reward, scenario_generator, models/)
COPY src/drl/__init__.py         /app/drl/__init__.py
COPY src/drl/digital_twin.py     /app/drl/digital_twin.py
COPY src/drl/edge_env.py         /app/drl/edge_env.py
COPY src/drl/legacy44_stormsafe_decoder.py /app/drl/legacy44_stormsafe_decoder.py
COPY src/drl/reward.py           /app/drl/reward.py
COPY src/drl/scenario_generator.py /app/drl/scenario_generator.py
COPY src/drl/twin_sync.py        /app/drl/twin_sync.py
COPY src/drl/placement_verifier.py /app/drl/placement_verifier.py
COPY src/drl/models/             /app/models/

# Shared project modules
COPY src/config.py               /app/config.py
COPY src/variant_catalog.py      /app/variant_catalog.py
COPY src/solver/                 /app/solver/
# HA package (leader election + sentinel-aware redis client)
COPY src/ha                      /app/ha
COPY src/k8s_client.py           /app/k8s_client.py

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["python", "drl_agent.py"]
