#!/bin/bash
set -e
REGISTRY="192.168.100.3:5000"

# Non-detection services
for SVC in api-gateway ingest preprocess postprocess; do
  echo "==> Building $SVC"
  docker build \
    -t $REGISTRY/$SVC:latest \
    ../microservices/$SVC        # Docker will find the Dockerfile inside this folder
  docker push $REGISTRY/$SVC:latest
done

# GenAI service
echo "==> Building gen-ai"
docker build \
  -t $REGISTRY/gen-ai:latest \
  ../microservices/gen_ai
docker push $REGISTRY/gen-ai:latest

# Detection variants
declare -A VARIANTS=( [yolo26-nano]="yolo26n.pt" [yolo26-small]="yolo26s.pt" [yolo26-medium]="yolo26m.pt" )
for VARIANT in yolo26-nano yolo26-small yolo26-medium; do
  MODEL=${VARIANTS[$VARIANT]}
  echo "==> Building detection:$VARIANT ($MODEL)"
  docker build \
    --build-arg MODEL_FILE=$MODEL \
    --build-arg VARIANT_ID=$VARIANT \
    -t $REGISTRY/detection:$VARIANT \
    ../microservices/detection
  docker push $REGISTRY/detection:$VARIANT
done

# MILP agent image (also contains variant_controller runtime)
echo "==> Building milp-agent"
docker build \
  -f ../src/milp_agent/Dockerfile \
  -t $REGISTRY/milp-agent:latest \
  ..
docker push $REGISTRY/milp-agent:latest

# DRL agent image
echo "==> Building drl-agent"
docker build \
  -f ../src/drl/Dockerfile \
  -t $REGISTRY/drl-agent:latest \
  ..
docker push $REGISTRY/drl-agent:latest

echo "All images built and pushed."
