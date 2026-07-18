#!/usr/bin/env python3
"""Convert drl_regret_cases.jsonl (milp_placement dicts) to BC training format.

Input:  results/drl_regret_cases.jsonl  (from run_offline_comparison --dump-state-action)
Output: results/idle_bc_expert.jsonl    ({state: [...], action: [...]})

Usage:
    python scripts/convert_expert_data.py
    python scripts/convert_expert_data.py --input results/my_cases.jsonl \
        --output results/my_expert.jsonl
"""
import argparse
import json
import sys

sys.path.insert(0, "src")

from collections import Counter

from drl.edge_env import NODE_IDS
from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS

DET_VARS = list(DETECTION_VARIANTS)
GEN_VARS = list(GEN_AI_VARIANTS)


def placement_to_action(pl: dict) -> list[int]:
    """Convert {svc_id: [variant, node_id]} → MultiDiscrete action [6]."""
    def simple_node(svc_id: str) -> int:
        _, node = pl[svc_id]
        return NODE_IDS.index(node)

    def det_action() -> int:
        var, node = pl["m3"]
        vi = DET_VARS.index(var) if var in DET_VARS else 0
        ni = NODE_IDS.index(node)
        return vi * len(NODE_IDS) + ni

    def gen_action() -> int:
        var, node = pl["m4"]
        vi = GEN_VARS.index(var) if var in GEN_VARS else 0
        ni = NODE_IDS.index(node)
        return vi * len(NODE_IDS) + ni

    return [
        simple_node("m0"), simple_node("m1"), simple_node("m2"),
        det_action(), gen_action(), simple_node("m5"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/drl_regret_cases.jsonl")
    parser.add_argument("--output", default="results/idle_bc_expert.jsonl")
    args = parser.parse_args()

    out: list[dict] = []
    skipped = 0
    node_counter: Counter = Counter()
    var_counter: Counter = Counter()

    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = rec.get("state")
            milp_pl = rec.get("milp_placement")
            if not state or not milp_pl:
                skipped += 1
                continue
            try:
                action = placement_to_action(milp_pl)
            except (ValueError, KeyError) as e:
                skipped += 1
                continue
            out.append({"state": state, "action": action})
            node_counter[milp_pl["m0"][1]] += 1
            var_counter[milp_pl["m3"][0]] += 1

    with open(args.output, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    n = len(out)
    print(f"Written {n} expert pairs  (skipped {skipped})")
    print(f"\nNode distribution (m0):")
    for node, cnt in sorted(node_counter.items()):
        bar = "█" * (cnt * 30 // n)
        print(f"  {node:<14} {cnt:>4}  ({cnt/n*100:5.1f}%)  {bar}")
    print(f"\nDetection variant distribution:")
    for var, cnt in sorted(var_counter.items()):
        bar = "█" * (cnt * 30 // n)
        print(f"  {var:<20} {cnt:>4}  ({cnt/n*100:5.1f}%)  {bar}")

    if node_counter.get("n3", 0) / n < 0.20:
        print("\n⚠️  WARNING: < 20% MILP placements on n3 — expert data may be biased to n0")
        print("   → Run more cycles or check _build_milp_dataset_from_state x_prev decoding")
    if var_counter.get("yolo26-medium", 0) / n < 0.15:
        print("\n⚠️  WARNING: < 15% medium variant — check w_a range in scenario profile")
    else:
        print("\n✅ Expert data looks good — proceed to BC training")


if __name__ == "__main__":
    main()
