#!/usr/bin/env bash
# k8s/scripts/apply-ha.sh
#
# One-shot script to apply all HA manifests in the correct order.
# Run this from the project root:
#   bash k8s/scripts/apply-ha.sh
#
# Steps:
#   1. Label edge nodes so HA deployments' nodeSelector matches.
#   2. Apply RBAC for leader election Leases.
#   3. Migrate Redis from single-pod to Sentinel HA.
#   4. Apply updated MILP + DRL + Variant controller deployments.
#
# Idempotent: safe to re-run.

set -euo pipefail

EDGE_NODES=("edge-nodes-1" "edge-nodes-2" "edge-nodes-3" "edge-nodes-4")

echo "=== Step 1: Label edge nodes ==="
for NODE in "${EDGE_NODES[@]}"; do
  if kubectl get node "$NODE" &>/dev/null; then
    kubectl label node "$NODE" node-role.kubernetes.io/edge=true --overwrite
    echo "  Labelled: $NODE"
  else
    echo "  WARNING: node $NODE not found — skipping"
  fi
done

echo ""
echo "=== Step 2: Apply leader election RBAC ==="
kubectl apply -f k8s/leader-election-rbac.yaml

echo ""
echo "=== Step 3: Migrate Redis to Sentinel HA ==="
# Remove old single-pod Redis (Service + Deployment)
kubectl delete -f k8s/redis-deployment.yaml --ignore-not-found || true
# Wait for old pods to terminate
echo "  Waiting for old redis pods to terminate..."
kubectl wait --for=delete pod -l app=redis -n default --timeout=30s 2>/dev/null || true
# Apply new HA stack
kubectl apply -f k8s/redis-ha.yaml
echo "  Waiting for redis-primary-0 to be ready..."
kubectl rollout status statefulset/redis-primary -n default --timeout=120s
echo "  Waiting for redis-replica-0 to be ready..."
kubectl rollout status statefulset/redis-replica -n default --timeout=120s
echo "  Waiting for redis-sentinel to be ready..."
kubectl rollout status deployment/redis-sentinel -n default --timeout=120s

echo ""
echo "=== Step 4: Apply updated controller deployments ==="
kubectl apply -f k8s/milp-agent-deployment.yaml
kubectl apply -f k8s/drl-agent-deployment.yaml

echo "  Waiting for milp-agent rollout..."
kubectl rollout status deployment/milp-agent -n default --timeout=180s
echo "  Waiting for drl-agent rollout..."
kubectl rollout status deployment/drl-agent -n default --timeout=180s
echo "  Waiting for variant-controller rollout..."
kubectl rollout status deployment/variant-controller -n default --timeout=120s

echo ""
echo "=== Step 5: Verify leader election ==="
sleep 5  # give Leases time to be acquired
echo "  MILP leader:"
kubectl get lease milp-leader -n default -o jsonpath='  holder={.spec.holderIdentity} epoch={.spec.leaseTransitions}' 2>/dev/null || echo "  (lease not yet created)"
echo ""
echo "  DRL leader:"
kubectl get lease drl-leader -n default -o jsonpath='  holder={.spec.holderIdentity} epoch={.spec.leaseTransitions}' 2>/dev/null || echo "  (lease not yet created)"
echo ""
echo "  Variant leader:"
kubectl get lease variant-controller-leader -n default -o jsonpath='  holder={.spec.holderIdentity}' 2>/dev/null || echo "  (lease not yet created)"
echo ""

echo ""
echo "=== Step 6: Verify Redis Sentinel ==="
kubectl exec deploy/redis-sentinel -n default -- \
  redis-cli -p 26379 sentinel get-master-addr-by-name mymaster 2>/dev/null \
  && echo "  Sentinel: primary discovered OK" \
  || echo "  WARNING: Sentinel not yet ready — retry in 30 s"

echo ""
echo "=== HA apply complete ==="
echo ""
echo "Next: run the T1–T5 chaos tests from docs/HA_Runbook.md §3 to validate."
