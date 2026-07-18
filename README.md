# KLTN Heterogeneous Edge Orchestrator

KLTN_project is an edge-computing orchestration prototype for a heterogeneous
Kubernetes cluster. The system combines a FastAPI microservice pipeline, a
MILP-based scheduler extender, a DRL placement agent, Redis-backed coordination,
and monitoring/evaluation tooling for thesis experiments.

The current implementation focuses on joint service placement and model-variant
selection for edge AI workloads. The main pipeline is:

```text
api-gateway -> ingest -> preprocess -> detection -> gen_ai -> postprocess
```

## Main Capabilities

- Kubernetes scheduler-extender service in `src/milp_agent/` for MILP-driven
  placement decisions.
- Pyomo/HiGHS MILP model in `src/solver/`, using normalized energy,
  disruption, and accuracy objectives.
- DRL inference and training utilities in `src/drl/`, including Digital Twin
  validation, PPO policies, reward alignment tests, and model artifacts.
- Redis coordination for placement state, control mode, objective weights,
  variant events, heartbeat, and online DRL buffers.
- Variant-aware services for detection (`yolo26-nano`, `yolo26-small`,
  `yolo26-medium`) and GenAI (`qwen-1.5b-nano`, `llama-3b-small`,
  `gemma2-2b-medium`).
- Kubernetes manifests for the application pipeline, Redis, MILP agent, DRL
  agent, placement verifier, twin sync, KWOK twin nodes, HA Redis, and
  observability.
- Experiment scripts and JSONL/CSV outputs for offline comparison, scenario
  stress tests, weight sensitivity, scalability, and thesis figures.

Default objective weights are defined in `src/config.py`:

```text
MILP_W_C = 0.20  # energy
MILP_W_D = 0.10  # disruption
MILP_W_A = 0.70  # accuracy / quality
```

## Repository Structure

```text
KLTN_project/
├── README.md
├── k8s/
│   ├── manifests/                # Pipeline deployments
│   │   ├── api-gateway.yaml
│   │   ├── ingest.yaml
│   │   ├── preprocess.yaml
│   │   ├── detection.yaml
│   │   ├── gen_ai.yaml
│   │   └── postprocess.yaml
│   ├── deployments/              # Additional deployment manifests
│   ├── monitoring/               # Prometheus/Grafana/Kepler configs
│   ├── scripts/                  # Cluster helper scripts
│   ├── drl-agent-deployment.yaml
│   ├── milp-agent-deployment.yaml
│   ├── placement-verifier-deployment.yaml
│   ├── redis-deployment.yaml
│   ├── redis-ha.yaml
│   ├── twin-sync-deployment.yaml
│   ├── kwok-twin-nodes.yaml
│   ├── kwok-capacity-patch-job.yaml
│   ├── extender-config.yaml
│   ├── leader-election-rbac.yaml
│   └── verifier-rbac.yaml
├── microservices/
│   ├── api-gateway/
│   ├── ingest/
│   ├── preprocess/
│   ├── detection/
│   ├── gen_ai/
│   ├── postprocess/
│   ├── mock_stage/
│   └── common/
├── monitoring/
│   ├── dashboard_ctl.sh
│   ├── dashboard_requirements.txt
│   └── milp_dashboard.py
├── results/
│   ├── thesis_main_offline/
│   ├── thesis_model_ablation/
│   ├── thesis_stress_*/
│   ├── weight_analysis/
│   └── *.jsonl / *.json / *.csv # Generated evaluation outputs
├── scripts/
│   ├── build_all.sh
│   ├── redeploy_control_plane.sh
│   ├── run_scenario_eval.py
│   ├── eval_live_state.py
│   ├── generate_drl_training_figures.py
│   ├── node_load_scenario.py
│   ├── check_milp_sync.sh
│   ├── convert_expert_data.py
│   └── setup_codegraph.sh
├── src/
│   ├── config.py
│   ├── k8s_client.py
│   ├── variant_catalog.py
│   ├── controller/               # Legacy edge controller
│   ├── drl/                      # DRL agent, env, twin, training, tests, models
│   ├── experiments/              # Scenario and evaluation runners
│   ├── ha/                       # Redis client and leader election helpers
│   ├── milp_agent/               # Scheduler extender FastAPI service
│   ├── simulator/                # Client traffic generator
│   ├── solver/                   # MILP dataset, collector, model, runner
│   └── variant_controller/       # Applies model variant changes
├── tmp/
│   ├── .codegraph/               # Local CodeGraph index/cache
│   ├── docs/
│   │   ├── figures/              # Exported thesis/evaluation charts
│   │   └── thesis/               # Thesis image assets moved out of root docs
│   ├── archive/                  # Archived docs, skills, papers, old notes
│   └── old files/
└── plot_results.ipynb
```

## Core Components

### Microservices

Each service under `microservices/` is a small FastAPI app with its own
`Dockerfile` and `requirements.txt` where needed.

- `api-gateway`: entrypoint for client requests and pipeline routing.
- `ingest`: accepts raw input payloads.
- `preprocess`: prepares input for downstream inference.
- `detection`: object-detection service with YOLO26 image variants.
- `gen_ai`: GenAI stage with multiple quality/resource variants.
- `postprocess`: formats final pipeline output.
- `mock_stage`: lightweight mock stage for local/testing workflows.
- `common`: shared helpers.

### Control Plane

- `src/milp_agent/milp_agent.py`: FastAPI scheduler-extender service. It exposes
  Kubernetes extender endpoints, writes `milp:placement`, emits Prometheus
  metrics, and uses leader election when available.
- `src/solver/`: dataset generation, Prometheus/Kubernetes metrics collection,
  and the MILP model.
- `src/variant_controller/variant_controller.py`: subscribes to Redis events and
  applies variant changes to Kubernetes deployments.
- `src/drl/drl_agent.py`: FastAPI DRL inference service on port `8001`.
- `src/drl/digital_twin.py`, `twin_sync.py`, `placement_verifier.py`: safety and
  validation helpers around DRL placement proposals.
- `src/ha/`: Redis and Kubernetes Lease based HA helpers.

### Redis Keys

Common keys used by the control plane:

```text
milp:placement             latest MILP placement payload
milp:confirmed_placement   live pod-to-node state
milp:weights               active objective weights
milp:events                variant-change pub/sub channel
milp:heartbeat             leader/agent heartbeat
drl:placement              committed DRL placement
drl:placement:proposed     shadow/proposed DRL placement
drl:twin_stats             latest Digital Twin validation stats
drl:training_buffer        online fine-tuning buffer
system:mode                milp | drl | hybrid | shadow
```

## Requirements

- Linux host with Docker.
- Kubernetes/K3s cluster with edge nodes labeled consistently with the manifests.
- `kubectl` access to the target cluster.
- Local registry reachable by the cluster. Scripts default to
  `192.168.100.3:5000`.
- Prometheus stack for live metrics. Kepler is optional for energy telemetry.
- Python environment for local scripts and notebooks.

## Build Images

`scripts/build_all.sh` currently uses paths relative to the `scripts/` directory,
so run it from there:

```bash
cd scripts
./build_all.sh
cd ..
```

The script builds and pushes:

- pipeline services: `api-gateway`, `ingest`, `preprocess`, `postprocess`
- `gen-ai`
- detection variants: `detection:yolo26-nano`, `detection:yolo26-small`,
  `detection:yolo26-medium`
- `milp-agent`
- `drl-agent`

Override the registry by editing `REGISTRY` in the script, or use
`scripts/redeploy_control_plane.sh` for the MILP control plane where `REGISTRY`
is environment-configurable.

## Deploy

Typical deployment order:

```bash
# Label edge nodes.
./k8s/scripts/node-labels.sh

# Deploy Redis and RBAC.
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/leader-election-rbac.yaml
kubectl apply -f k8s/verifier-rbac.yaml

# Deploy the application pipeline.
kubectl apply -f k8s/manifests/

# Deploy control-plane services.
kubectl apply -f k8s/milp-agent-deployment.yaml
kubectl apply -f k8s/drl-agent-deployment.yaml
kubectl apply -f k8s/placement-verifier-deployment.yaml
kubectl apply -f k8s/twin-sync-deployment.yaml
```

Optional KWOK/Digital Twin resources:

```bash
kubectl apply -f k8s/kwok-twin-nodes.yaml
kubectl apply -f k8s/kwok-capacity-patch-job.yaml
```

Optional monitoring resources:

```bash
kubectl apply -f k8s/monitoring/kepler-servicemonitor.yaml
kubectl apply -f k8s/monitoring/grafana-node-resource-energy-dashboard.yaml
kubectl apply -f k8s/monitoring/grafana-pipeline-milp-dashboard.yaml
```

For a focused MILP control-plane rebuild/redeploy:

```bash
REGISTRY=192.168.100.3:5000 NAMESPACE=default ./scripts/redeploy_control_plane.sh
```

## Health Checks

```bash
# MILP agent
kubectl port-forward deploy/milp-agent 8080:8080
curl -s http://127.0.0.1:8080/health

# DRL agent
kubectl port-forward deploy/drl-agent 8001:8001
curl -s http://127.0.0.1:8001/health

# Logs
kubectl logs -f deployment/milp-agent --tail=100
kubectl logs -f deployment/drl-agent --tail=100
kubectl logs -f deployment/variant-controller --tail=100
```

Inspect Redis state:

```bash
kubectl exec deploy/redis -- redis-cli GET milp:placement
kubectl exec deploy/redis -- redis-cli GET drl:placement
kubectl exec deploy/redis -- redis-cli GET drl:twin_stats
kubectl exec deploy/redis -- redis-cli GET system:mode
```

Switch control mode:

```bash
kubectl exec deploy/redis -- redis-cli SET system:mode milp
kubectl exec deploy/redis -- redis-cli SET system:mode shadow
kubectl exec deploy/redis -- redis-cli SET system:mode drl
kubectl exec deploy/redis -- redis-cli SET system:mode hybrid
```

## Run Experiments

Scenario runner:

```bash
python3 src/experiments/run_scenario.py --scenario energy_saving --dry-run
python3 src/experiments/run_scenario.py --scenario quality_maximise --dry-run
python3 src/experiments/run_scenario.py --scenario balanced --dry-run
python3 src/experiments/run_scenario.py --scenario ram_pressure --dry-run
python3 src/experiments/run_scenario.py --scenario storm_test --dry-run
```

Offline comparison and analysis scripts:

```bash
python3 src/experiments/run_offline_comparison.py
python3 src/experiments/run_offline_comparsion_v2.py
python3 src/experiments/weight_sensitivity_analysis.py
python3 src/experiments/scalability_benchmark.py
python3 scripts/generate_drl_training_figures.py
```

Outputs are stored in `results/`, with thesis-oriented subdirectories such as
`thesis_main_offline/`, `thesis_model_ablation/`, `thesis_stress_*`, and
`weight_analysis/`.

Exported thesis charts and thesis image copies are currently staged under
`tmp/docs/figures/` and `tmp/docs/thesis/.../img/`.

## Traffic Simulation

The traffic generator lives in `src/simulator/`.

```bash
cd src/simulator
python3 simulator.py
```

Configuration files:

- `src/simulator/config.env`
- `src/simulator/config.mac.env`

Typical output is appended to JSONL files under `results/`.

## Notes

- `tmp/docs/` contains moved thesis figures and generated document assets.
- `tmp/archive/` contains archived documents, older plans, reference papers, and
  historical notes. It is useful for thesis traceability, but it is not part of
  the active runtime path.
- `results/` and `tmp/` are local/generated-output areas in `.gitignore`.
- `src/controller/edge_controller.py` is kept as a legacy controller. The active
  path is the scheduler-extender based `milp_agent` plus optional DRL/hybrid
  controllers.
- Generated artifacts such as models, logs, notebooks, and result files are
  already present in this repository and may be large.
