#!/usr/bin/env bash
set -euo pipefail

# One-command redeploy for MILP control-plane (milp-agent + variant-controller).
# Usage:
#   ./scripts/redeploy_control_plane.sh
# Optional env overrides:
#   REGISTRY=192.168.100.3:5000
#   NAMESPACE=default
#   KUBECTL_BIN="sudo -n kubectl"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REGISTRY:-192.168.100.3:5000}"
NAMESPACE="${NAMESPACE:-default}"
IMAGE="${REGISTRY}/milp-agent:latest"
KUBECTL_BIN="${KUBECTL_BIN:-sudo -n kubectl}"

read -r -a KUBECTL <<<"${KUBECTL_BIN}"

echo "[1/5] Preflight checks"
command -v docker >/dev/null 2>&1
"${KUBECTL[@]}" get ns >/dev/null

echo "[2/5] Build + push ${IMAGE}"
docker build -f "${ROOT_DIR}/src/milp_agent/Dockerfile" -t "${IMAGE}" "${ROOT_DIR}"
docker push "${IMAGE}"

echo "[3/5] Apply manifests + restart deployments"
"${KUBECTL[@]}" apply -f "${ROOT_DIR}/k8s/milp-agent-deployment.yaml"
"${KUBECTL[@]}" rollout restart deployment/milp-agent -n "${NAMESPACE}"
"${KUBECTL[@]}" rollout restart deployment/variant-controller -n "${NAMESPACE}"

echo "[4/5] Wait for rollout"
"${KUBECTL[@]}" rollout status deployment/milp-agent -n "${NAMESPACE}" --timeout=240s
"${KUBECTL[@]}" rollout status deployment/variant-controller -n "${NAMESPACE}" --timeout=240s

echo "[5/5] Verify payload schema + pod health"
payload="$("${KUBECTL[@]}" exec deploy/redis -n "${NAMESPACE}" -- redis-cli GET milp:placement)"
python3 - <<'PY' "$payload"
import json
import sys

raw = sys.argv[1]
if not raw:
    raise SystemExit("milp:placement is empty")

data = json.loads(raw)
required = [
    "timestamp",
    "norm_cost_energy",
    "norm_cost_disruption",
    "norm_gain_accuracy",
    "cost_energy",
    "cost_disruption",
    "gain_accuracy",
]
missing = [k for k in required if k not in data]
if missing:
    raise SystemExit(f"Missing fields in milp:placement: {missing}")

print("placement schema: OK")
print("objective:", data.get("objective"))
print("solve_time:", data.get("solve_time"))
print("timestamp:", data.get("timestamp"))
PY

"${KUBECTL[@]}" get pods -n "${NAMESPACE}" -o wide | grep -E "milp-agent|variant-controller|redis"

echo "Done. Control-plane redeploy and validation completed."
