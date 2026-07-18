#!/usr/bin/env bash
# patch_kwok_capacity.sh — Patch KWOK twin-node capacity/allocatable subresource.
#
# WHY: kubectl apply does NOT update Node status on existing resources.
#      Running this script (or the equivalent K8s Job) after every apply of
#      kwok-twin-nodes.yaml ensures the scheduler sees correct CPU/memory for
#      the virtual nodes so placement-verifier test-pods can be scheduled.
#
# Memory derivation (mirrors edge_env.py _NODE_CAP_MEM):
#   n0, n1: 3.82 GiB = 3911 MiB    |    n2, n3: 7.75 GiB = 7937 MiB
#
# CPU derivation (mirrors edge_env.py _NODE_CAP_CPU):
#   n0, n2, n3: 4 cores    |    n1: 2 cores
#
# Usage:
#   bash k8s/scripts/patch_kwok_capacity.sh
#   bash k8s/scripts/patch_kwok_capacity.sh --dry-run
set -euo pipefail

KUBECTL="${KUBECTL:-kubectl}"
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run=client"
  echo "[dry-run] Showing patch commands without applying"
fi

patch_node() {
  local name="$1" cpu="$2" mem="$3"
  echo "  Patching $name  (cpu=$cpu, mem=$mem)..."
  $KUBECTL patch node "$name" --subresource=status --type=merge $DRY_RUN \
    -p "{\"status\":{\"capacity\":{\"cpu\":\"$cpu\",\"memory\":\"$mem\",\"pods\":\"110\"},\"allocatable\":{\"cpu\":\"$cpu\",\"memory\":\"$mem\",\"pods\":\"110\"}}}"
}

echo "=== KWOK twin-node capacity patch ==="
patch_node twin-edge-nodes-1 "4" "3911Mi"
patch_node twin-edge-nodes-2 "2" "3911Mi"
patch_node twin-edge-nodes-3 "4" "7937Mi"
patch_node twin-edge-nodes-4 "4" "7937Mi"
echo "=== Done ==="
