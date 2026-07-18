#!/usr/bin/env bash
set -euo pipefail

# Verify MILP desired placement against Redis confirmed placement and live pod state.
# Usage:
#   ./scripts/check_milp_sync.sh
# Optional env overrides:
#   NAMESPACE=default
#   KUBECTL_BIN="kubectl"

NAMESPACE="${NAMESPACE:-default}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"

read -r -a KUBECTL <<<"${KUBECTL_BIN}"

echo "[1/3] Read desired MILP placement from Redis"
placement_raw="$(${KUBECTL[@]} exec deploy/redis -n "${NAMESPACE}" -- redis-cli GET milp:placement)"
confirmed_raw="$(${KUBECTL[@]} exec deploy/redis -n "${NAMESPACE}" -- redis-cli GET milp:confirmed_placement || true)"

if [[ -z "${placement_raw}" ]]; then
  echo "ERROR: milp:placement is empty"
  exit 2
fi

echo "[2/3] Read live pod state from Kubernetes"
pods_json="$(${KUBECTL[@]} get pods -n "${NAMESPACE}" -o json)"

echo "[3/3] Compare desired vs confirmed vs live"
python3 - <<'PY' "$placement_raw" "$confirmed_raw" "$pods_json"
import json
import sys

placement_raw = sys.argv[1]
confirmed_raw = sys.argv[2]
pods_raw = sys.argv[3]

payload = json.loads(placement_raw)
placement = payload.get("placement", {})
confirmed = json.loads(confirmed_raw) if confirmed_raw else {}
pods = json.loads(pods_raw).get("items", [])

services = ["api-gateway", "ingest", "preprocess", "detection", "gen-ai", "postprocess"]

def default_variant(service: str) -> str:
    if service == "detection":
        return "yolo26-nano"
    if service == "gen-ai":
        return "qwen-1.5b-nano"
    return "standard"

def get_live_state(service: str) -> dict:
    running = [
        p for p in pods
        if p.get("metadata", {}).get("labels", {}).get("app") == service
        and p.get("status", {}).get("phase") == "Running"
    ]
    if not running:
        return {"node": "", "variant": default_variant(service), "pod": ""}

    pod = sorted(running, key=lambda p: p.get("metadata", {}).get("creationTimestamp", ""))[-1]
    node = pod.get("spec", {}).get("nodeName", "")
    pod_name = pod.get("metadata", {}).get("name", "")
    variant = default_variant(service)
    for c in pod.get("spec", {}).get("containers", []):
        for e in c.get("env", []) or []:
            if e.get("name") == "VARIANT_ID":
                variant = e.get("value", variant)
                break
    return {"node": node, "variant": variant, "pod": pod_name}

mismatches = []
header = (
    f"{'SERVICE':<12} {'DESIRED_NODE':<16} {'LIVE_NODE':<16} "
    f"{'DESIRED_VAR':<18} {'LIVE_VAR':<18} {'CONF_NODE':<16} {'CONF_VAR':<18}"
)
print(header)
print("-" * len(header))

for service in services:
    desired = placement.get(service, {})
    desired_node = desired.get("node", "")
    desired_variant = desired.get("variant", default_variant(service))

    live = get_live_state(service)
    live_node = live.get("node", "")
    live_variant = live.get("variant", default_variant(service))

    conf = confirmed.get(service, {})
    conf_node = conf.get("node", "")
    conf_variant = conf.get("variant", default_variant(service))

    print(
        f"{service:<12} {desired_node:<16} {live_node:<16} "
        f"{desired_variant:<18} {live_variant:<18} {conf_node:<16} {conf_variant:<18}"
    )

    node_ok = (not desired_node) or (desired_node == live_node)
    variant_ok = desired_variant == live_variant
    conf_node_ok = (not conf_node) or (conf_node == live_node)
    conf_variant_ok = conf_variant == live_variant

    if not (node_ok and variant_ok and conf_node_ok and conf_variant_ok):
        mismatches.append(
            {
                "service": service,
                "desired": {"node": desired_node, "variant": desired_variant},
                "confirmed": {"node": conf_node, "variant": conf_variant},
                "live": {"node": live_node, "variant": live_variant},
            }
        )

print()
if mismatches:
    print(f"SYNC_STATUS: DRIFT ({len(mismatches)} service(s))")
    print(json.dumps(mismatches, indent=2))
    raise SystemExit(1)

print("SYNC_STATUS: OK")
PY
