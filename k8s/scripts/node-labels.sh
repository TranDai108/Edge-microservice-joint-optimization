#!/bin/bash
kubectl label node edge-nodes-1 node-id=n0 --overwrite
kubectl label node edge-nodes-2 node-id=n1 --overwrite
kubectl label node edge-nodes-3 node-id=n2 --overwrite
kubectl label node edge-nodes-4 node-id=n3 --overwrite
