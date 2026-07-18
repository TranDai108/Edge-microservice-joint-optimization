#!/usr/bin/env python3
"""Evaluate a DRL model against MILP on the current live Redis state.

Usage:
    python scripts/eval_live_state.py --model src/drl/models/ppo_new.zip
    python scripts/eval_live_state.py --model src/drl/models/ppo_new.zip \
        --compare src/drl/models/ppo_old.zip
"""
import argparse
import json
import sys

sys.path.insert(0, "src")

import numpy as np
import redis
from sb3_contrib import MaskablePPO

from drl.edge_env import NODE_IDS, EdgeEnv, build_mock_dataset, compute_action_masks
from drl.reward import evaluate_objective_for_placement

DET_VARS = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
GEN_VARS = ["qwen-1.5b-nano", "llama-3b-small", "gemma2-2b-medium"]
NN = {0: "n0(19.9W)", 1: "n1(41W)", 2: "n2(19.8W)", 3: "n3(10W)"}

H2ID = {
    "edge-nodes-1": "n0",
    "edge-nodes-2": "n1",
    "edge-nodes-3": "n2",
    "edge-nodes-4": "n3",
}
S2ID = {
    "api-gateway": "m0",
    "ingest": "m1",
    "preprocess": "m2",
    "detection": "m3",
    "gen-ai": "m4",
    "postprocess": "m5",
}


def decode_action(a: list[int]) -> dict[str, tuple[str, str]]:
    return {
        "m0": ("standard", NODE_IDS[a[0]]),
        "m1": ("standard", NODE_IDS[a[1]]),
        "m2": ("standard", NODE_IDS[a[2]]),
        "m3": (DET_VARS[min(a[3] // 4, 2)], NODE_IDS[a[3] % 4]),
        "m4": (GEN_VARS[min(a[4] // 4, 2)], NODE_IDS[a[4] % 4]),
        "m5": ("standard", NODE_IDS[a[5]]),
    }


def build_ds_from_live(state: np.ndarray, live_pl: dict) -> object:
    ds = build_mock_dataset()
    for k in list(ds.x_prev):
        ds.x_prev[k] = 0.0
    for svc, info in live_pl.items():
        svc_id = S2ID.get(svc, svc)
        node_id = H2ID.get(info["node"], info["node"])
        key = (svc_id, info["variant"], node_id)
        if key in ds.x_prev:
            ds.x_prev[key] = 1.0
    ds.w_c = float(state[41])
    ds.w_d = float(state[42])
    ds.w_a = float(state[43])
    return ds


def eval_model(path: str, state: np.ndarray, ds) -> tuple[float, list[int]]:
    model = MaskablePPO.load(path)
    masks = compute_action_masks(state)
    action, _ = model.predict(state, deterministic=True, action_masks=masks)
    a = action.tolist()
    pl = decode_action(a)
    obj = evaluate_objective_for_placement(ds, pl)
    return obj.objective_j, a


def print_placement(a: list[int], label: str) -> None:
    pl = decode_action(a)
    svc_names = ["m0(api-gw)", "m1(ingest)", "m2(preproc)",
                 "m3(detect)", "m4(gen-ai)", "m5(postproc)"]
    print(f"\n  [{label}]")
    for i, (svc, (var, node)) in enumerate(pl.items()):
        node_idx = NODE_IDS.index(node)
        print(f"    {svc_names[i]:<14} → {NN[node_idx]:<14}  variant={var}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval DRL model on live state")
    parser.add_argument("--model", required=True, help="Path to DRL model zip")
    parser.add_argument("--compare", default=None, help="Optional second model to compare")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    rdb = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    env = EdgeEnv(mock=False)
    env.rdb = rdb
    env.reset()
    state = env._build_live_state()

    live_milp_raw = rdb.get("milp:placement")
    if not live_milp_raw:
        print("ERROR: milp:placement not in Redis — is the MILP agent running?")
        sys.exit(1)

    live_milp = json.loads(live_milp_raw)
    live_pl = live_milp["placement"]
    ds = build_ds_from_live(state, live_pl)

    # MILP reference
    pl_milp = {S2ID[k]: (v["variant"], H2ID[v["node"]]) for k, v in live_pl.items()}
    milp_obj = evaluate_objective_for_placement(ds, pl_milp)
    ref_j = milp_obj.objective_j

    print("=" * 60)
    print(f"  Live State Evaluation  (w_a={ds.w_a:.2f} w_c={ds.w_c:.2f} w_d={ds.w_d:.2f})")
    print("=" * 60)
    print(f"\n  MILP J (reference) = {ref_j:.4f}")

    # Evaluate new model
    drl_j, drl_a = eval_model(args.model, state, ds)
    gap = (drl_j - ref_j) / abs(ref_j) * 100
    verdict = "✅ GOOD" if gap < 5 else ("⚠️  OK" if gap < 15 else "❌ POOR")
    print(f"\n  NEW MODEL ({args.model})")
    print(f"    J = {drl_j:.4f}    gap = {gap:+.2f}%   {verdict}")
    print_placement(drl_a, "NEW placement")

    # Evaluate compare model if provided
    if args.compare:
        old_j, old_a = eval_model(args.compare, state, ds)
        old_gap = (old_j - ref_j) / abs(ref_j) * 100
        print(f"\n  COMPARE MODEL ({args.compare})")
        print(f"    J = {old_j:.4f}    gap = {old_gap:+.2f}%")
        print_placement(old_a, "OLD placement")
        improvement = old_gap - gap
        print(f"\n  Improvement new vs old: {improvement:+.2f}pp gap reduction")

    print("\n" + "=" * 60)
    print("  Acceptance criteria:")
    print(f"    gap < 5%   → {gap < 5}")
    print(f"    gap < 15%  → {gap < 15}")
    n1_used = any(NODE_IDS[a] == "n1" for a in drl_a[:4] + [drl_a[5]])
    det_good = drl_a[3] // 4 >= 1  # small or medium
    print(f"    no n1 node → {not n1_used}")
    print(f"    det ≥small → {det_good}")
    print("=" * 60)


if __name__ == "__main__":
    main()
