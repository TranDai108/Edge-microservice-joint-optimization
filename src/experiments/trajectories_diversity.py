"""DEPRECATED — use inject_diverse_trajectories.py instead.

This script has been superseded by the canonical trajectory injector which
provides a systematic parameter grid (weight anchors + random weights ×
placement templates × storm levels), dry-run mode, CLI flags, and accurate
unique_action_ratio reporting.

Equivalent command:
    python3 src/experiments/inject_diverse_trajectories.py --redis-host localhost

To flush existing trajectories first:
    python3 src/experiments/inject_diverse_trajectories.py --redis-host localhost --flush
"""
import sys
from pathlib import Path
import sys
import numpy as np

root=Path('/home/ubuntu/KLTN_project')
sys.path.insert(0,str(root/'src'))
sys.path.insert(0,str(root/'src'/'solver'))

from solver.metrics_collector import build_dataset_from_cluster
from solver.milp_model import solve_placement
from milp_agent.milp_agent import _build_state_vector, _build_action_vector
from milp_agent.redis_state import write_expert_trajectory
import k8s_client

r=redis.Redis(host='localhost',port=6379,decode_responses=True)

hosts=sorted(k8s_client.get_all_worker_nodes())
if len(hosts)<4:
    hosts=['edge-nodes-1','edge-nodes-2','edge-nodes-3','edge-nodes-4']

services=['api-gateway','ingest','preprocess','detection','gen-ai','postprocess']
det_vars=['yolo26-nano','yolo26-small','yolo26-medium']
gen_vars=['qwen-1.5b-nano','llama-3b-small','gemma2-2b-medium']
weight_sets=[
 (0.80,0.15,0.05),
 (0.60,0.30,0.10),
 (0.40,0.20,0.40),
 (0.25,0.15,0.60),
 (0.20,0.10,0.70),
 (0.10,0.05,0.85),
]

rng=random.Random(123)
inserted=0
failed=0
seen_actions=set()

for _ in range(260):
    lp={}
    for svc in services:
        node=rng.choice(hosts)
        if svc=='detection':
            var=rng.choice(det_vars)
        elif svc=='gen-ai':
            var=rng.choice(gen_vars)
        else:
            var='standard'
        lp[svc]={'node':node,'variant':var}

    w_c,w_d,w_a=rng.choice(weight_sets)
    theta_max=rng.choice([1.0,1.1,1.2,1.3,1.4])
    v_storm=rng.choice([2,3,4,5,6])

    try:
        ds=build_dataset_from_cluster(
            last_placement=lp,
            w_c=w_c,
            w_d=w_d,
            w_a=w_a,
            theta_max=theta_max,
            v_storm_max=v_storm,
        )
        result=solve_placement(ds,verbose=False)
        if not result:
            failed += 1
            continue
        state=_build_state_vector(ds,result)
        action=_build_action_vector(ds,result)
        reward=-float(result.objective_value)
        if len(state)!=44 or len(action)!=6:
            failed += 1
            continue
        arr=np.asarray(state,dtype=float)
        if not np.isfinite(arr).all():
            failed += 1
            continue
        write_expert_trajectory(r,state,action,reward,state)
        inserted += 1
        seen_actions.add(tuple(action))
    except Exception:
        failed += 1

print('inserted',inserted)
print('failed',failed)
print('unique_actions_inserted',len(seen_actions))
print('total_len',r.llen('milp:expert_trajectories'))
