#!/home/ubuntu/KLTN_project/.venv/bin/python
"""Offline scenario-driven comparison: MILP vs DRL vs Hybrid (+ baselines).

No Redis. No Kubernetes. Runs entirely from synthetic cluster states generated
by ScenarioGenerator and evaluated by the pure-Python reward function.

Pipeline (per cycle)
─────────────────────
  ScenarioGenerator.sample_state()       → 44-dim state vector
        │
        ├─ MILP:   state → MILPDataset → solve_placement()  → J, energy, migrations
        ├─ DRL:    state → PPO.predict() → _action_to_placement() → evaluate_objective()
        └─ Hybrid: DRL action → DigitalTwinValidator.validate()
                   • is_safe  → use DRL placement
                   • rejected → fall back to MILP placement

Output files (same schema as run_comparison.py, notebook-ready):
  results/off_comparison_milp.jsonl
  results/off_comparison_drl.jsonl
  results/off_comparison_hybrid.jsonl
  results/off_comparison_random.jsonl
  results/off_comparison_roundrobin.jsonl

Typical run (200 cycles, all modes, ~30 s):
  python3 src/experiments/run_offline_comparison.py

Fast targeted run:
  python3 src/experiments/run_offline_comparison.py --cycles 200 --modes milp drl hybrid
  python3 src/experiments/run_offline_comparison.py --scenario cascade_failure --cycles 50
"""
from __future__ import annotations

import argparse
import json
import logging
import random as _random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]

# ── Venv self-re-exec guard ───────────────────────────────────────────────────
# If stable_baselines3 is not importable (e.g. run with system python3),
# transparently re-launch under the project venv python and exit.
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

def _ensure_venv() -> None:
    import importlib.util, os
    if importlib.util.find_spec("stable_baselines3") is None and _VENV_PYTHON.exists():
        import subprocess
        result = subprocess.run([str(_VENV_PYTHON)] + sys.argv)
        sys.exit(result.returncode)

_ensure_venv()
for _p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local imports (no Redis, no FastAPI) ──────────────────────────────────────
from drl.scenario_generator import ScenarioGenerator, ScenarioType
from drl.edge_env import (
    NODE_IDS,
    SERVICE_IDS,
    build_mock_dataset,
    compute_action_masks,
)
from drl.legacy44_stormsafe_decoder import (
    DEFAULT_ACTION_DIMS as LEGACY_ACTION_DIMS,
    DEFAULT_TOPK_PER_HEAD as LEGACY_TOPK_PER_HEAD,
    apply_flat_mask_to_head_logits,
    decode_legacy44_action,
    select_stormsafe_action,
    split_head_logits,
)
from drl.reward import evaluate_objective_for_placement
from drl.digital_twin import DigitalTwinValidator
from solver.milp_model import solve_placement
from solver.dataset_generator import MILPDataset

# ── Constants ─────────────────────────────────────────────────────────────────
SLA_LATENCY_MS = 1500.0

MODES = ["milp", "drl", "hybrid"]
ALL_MODES = ["milp", "drl", "hybrid", "random", "roundrobin", "k8s_default_light", "k8s_default_quality"]

_NODE_HOSTNAMES = ["edge-nodes-1", "edge-nodes-2", "edge-nodes-3", "edge-nodes-4"]

# Mock e2e latency model: base + per-node load contribution
# Kept simple and deterministic given a state vector so MILP and DRL see
# identical latency for the same scenario.
_BASE_E2E_MS = 80.0
_E2E_LOAD_SCALE = 300.0   # max additional ms at full CPU load

log = logging.getLogger("offline-comparison")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _round_ms(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def _e2e_from_state(state: np.ndarray) -> float:
    """Estimate E2E latency (ms) from CPU load in state vector.

    Uses the state's existing e2e field [39] if it looks realistic,
    otherwise synthesises from CPU utilisation — ensuring both agents
    see the same latency for a given scenario.
    """
    e2e_raw = float(state[39])
    if 0.0 < e2e_raw < 5000.0:
        return e2e_raw
    # Fallback: proportional to max node CPU utilisation
    max_cpu = float(np.max(state[0:4]))
    return _BASE_E2E_MS + _E2E_LOAD_SCALE * max_cpu


def _build_milp_dataset_from_state(state: np.ndarray) -> MILPDataset:
    """Convert a 44-dim scenario state vector into a solvable MILPDataset.

    Extracts node CPU util [0:4], mem_gb [4:8], e_cpu_unit [8:12] and the
    MILP weights [41:44] from the state.  Resource requirements and
    variant catalogs are taken from build_mock_dataset() so they match
    the DRL agent's training distribution exactly.
    """
    base = build_mock_dataset()

    # Override node energy costs with scenario values (thermal throttle, etc.)
    e_cpu_unit = state[8:12].tolist()
    for i, node in enumerate(base.nodes):
        node.energy_cost = max(0.1, float(e_cpu_unit[i]))

    # Override objective weights from state [41:44]
    w_c = float(state[41])
    w_d = float(state[42])
    w_a = float(state[43])
    # Normalise so weights sum to 1 (guard against scenario jitter)
    total = w_c + w_d + w_a
    if total > 0:
        w_c, w_d, w_a = w_c / total, w_d / total, w_a / total
    base.w_c = w_c
    base.w_d = w_d
    base.w_a = w_a

    # Adjust available CPU headroom by marking nodes as partially loaded.
    # The MILP's capacity constraint uses node.cap_cpu directly; we reduce
    # effective capacity proportional to current utilisation from the state.
    cpu_util = state[0:4].tolist()
    for i, node in enumerate(base.nodes):
        occupied = float(cpu_util[i]) * node.cap_cpu
        # Reduce cap_cpu so the MILP treats already-used capacity as unavailable.
        node.cap_cpu = max(0.5, node.cap_cpu - occupied)

    # Same for memory
    mem_used_gb = state[4:8].tolist()
    for i, node in enumerate(base.nodes):
        node.cap_mem_gb = max(0.1, node.cap_mem_gb - float(mem_used_gb[i]))

    # ── Decode x_prev from state[15:39] (placement one-hot) + state[12:15] ──
    # build_mock_dataset() always initialises x_prev to all-services-on-n0.
    # The scenario generator randomises state[15:39], so we must decode x_prev
    # from the state to compute the correct migration cost for MILP.
    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415
    det_vars  = list(DETECTION_VARIANTS)
    gen_vars  = list(GEN_AI_VARIANTS)
    det_var_idx = int(np.argmax(state[12:15])) if state[12:15].sum() > 0 else 0
    # Clear existing x_prev
    for k in list(base.x_prev):
        base.x_prev[k] = 0.0
    for svc_idx, svc_id in enumerate(["m0", "m1", "m2", "m3", "m4", "m5"]):
        slot = state[15 + svc_idx * 4: 15 + svc_idx * 4 + 4]
        node_idx = int(np.argmax(slot)) if slot.sum() > 0 else 0
        node_id  = NODE_IDS[node_idx]
        if svc_id == "m3":
            var = det_vars[det_var_idx]
        elif svc_id == "m4":
            var = gen_vars[0]   # gen-ai variant not encoded in state; use lightest
        else:
            var = "standard"
        if (svc_id, var, node_id) in base.x_prev:
            base.x_prev[(svc_id, var, node_id)] = 1.0

    return base


def _decode_drl_action(action: np.ndarray) -> dict[str, tuple[str, str]]:
    """Decode PPO MultiDiscrete action → {svc_id: (variant, node_id)}."""
    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415

    a = np.asarray(action, dtype=np.int64).tolist()

    def _vn(value: int, variants) -> tuple[str, str]:
        n = len(NODE_IDS)
        vi = min(max(int(value) // n, 0), len(variants) - 1)
        ni = min(max(int(value) % n, 0), n - 1)
        return list(variants)[vi], NODE_IDS[ni]

    det_var, det_node = _vn(int(a[3]), DETECTION_VARIANTS)
    gen_var, gen_node = _vn(int(a[4]), GEN_AI_VARIANTS)

    return {
        "m0": ("standard", NODE_IDS[min(int(a[0]), 3)]),
        "m1": ("standard", NODE_IDS[min(int(a[1]), 3)]),
        "m2": ("standard", NODE_IDS[min(int(a[2]), 3)]),
        "m3": (det_var, det_node),
        "m4": (gen_var, gen_node),
        "m5": ("standard", NODE_IDS[min(int(a[5]), 3)]),
    }


def _decode_legacy44_stormsafe_action(action: np.ndarray) -> dict[str, tuple[str, str]]:
    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415

    return decode_legacy44_action(
        action,
        node_ids=NODE_IDS,
        detection_variants=list(DETECTION_VARIANTS),
        gen_ai_variants=list(GEN_AI_VARIANTS),
    )


def _migration_count_from_x_prev(
    ds: MILPDataset,
    placement: dict[str, tuple[str, str]],
) -> int:
    return int(
        sum(
            1
            for svc, (var, node) in placement.items()
            if float(ds.x_prev.get((svc, var, node), 0.0)) < 0.5
        )
    )


def _head_logits_from_policy(model: Any, obs: np.ndarray, flat_mask: np.ndarray) -> list[np.ndarray]:
    """Extract per-head logits for legacy 44-dim MultiDiscrete PPO models.

    This mirrors the runtime storm-safe decoder path used by the DRL agent so
    offline thesis results evaluate the deployed artifact rather than a raw
    greedy MultiDiscrete decode.
    """
    import torch  # noqa: PLC0415

    action_dims = [int(x) for x in getattr(model.action_space, "nvec", LEGACY_ACTION_DIMS)]
    try:
        with torch.inference_mode():
            obs_t, _ = model.policy.obs_to_tensor(obs)
            features = model.policy.extract_features(obs_t)
            latent_pi, _ = model.policy.mlp_extractor(features)
            flat_logits = (
                model.policy.action_net(latent_pi)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            return split_head_logits(flat_logits, action_dims=action_dims)
    except Exception:
        action, _ = model.predict(obs, deterministic=True)
        act = np.asarray(action, dtype=np.int64).reshape(-1)
        heads: list[np.ndarray] = []
        for i, dim in enumerate(action_dims):
            logits = np.full((dim,), -1e9, dtype=np.float64)
            idx = int(act[i]) if i < len(act) and 0 <= int(act[i]) < dim else 0
            logits[idx] = 0.0
            heads.append(logits)
        return heads


def _predict_stormsafe(
    model: Any,
    ds: MILPDataset,
    state: np.ndarray,
) -> tuple[np.ndarray, dict[str, tuple[str, str]], dict[str, Any]]:
    """Return storm-safe legacy action, placement and decision metadata."""
    obs = np.asarray(state, dtype=np.float32).reshape(-1)
    action_dims = [int(x) for x in getattr(model.action_space, "nvec", LEGACY_ACTION_DIMS)]
    flat_mask = compute_action_masks(obs)
    head_logits = _head_logits_from_policy(model, obs, flat_mask)
    masked_logits = apply_flat_mask_to_head_logits(
        head_logits=head_logits,
        flat_mask=flat_mask,
        action_dims=action_dims,
    )
    outcome = select_stormsafe_action(
        dataset=ds,
        head_logits=masked_logits,
        decode_action_fn=_decode_legacy44_stormsafe_action,
        objective_fn=evaluate_objective_for_placement,
        action_dims=action_dims,
        topk_per_head=LEGACY_TOPK_PER_HEAD,
    )
    meta = {
        "decoder_mode": outcome.decoder_mode,
        "storm_safe": bool(outcome.migration_count <= outcome.storm_max),
        "migration_count": int(outcome.migration_count),
        "storm_max": int(outcome.storm_max),
        "fallback_full_enumeration": bool(outcome.fallback_full_enumeration),
        "candidates_evaluated": int(outcome.candidates_evaluated),
        "decoder_feasible": bool(outcome.feasible),
        "objective_j_selected": float(outcome.objective_j),
    }
    return np.asarray(outcome.action, dtype=np.int64), outcome.placement, meta


def _milp_result_to_row(
    result,
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    mode: str,
    scenario_type: str,
) -> dict:
    """Convert a MILP PlacementResult into a notebook-compatible JSONL row."""
    e2e = _e2e_from_state(state)
    if result is None:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle,
            "mode": mode,
            "scenario": scenario_type,
            "objective_J": None,
            "e2e_latency_ms": e2e,
            "energy_w": None,
            "migrations": None,
            "sla_violated": e2e > SLA_LATENCY_MS,
            "infeasible": True,
        }

    migrations = sum(1 for v in result.migration_types.values() if v != "Stayed")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "mode": mode,
        "scenario": scenario_type,
        "objective_J": round(float(result.objective_value), 6),
        "e2e_latency_ms": round(e2e, 2),
        "energy_w": round(float(result.cost_energy), 4) if result.cost_energy is not None else None,
        "migrations": migrations,
        "sla_violated": e2e > SLA_LATENCY_MS,
        "infeasible": False,
        # Extra fields for richer analysis (ignored by notebook, useful for research)
        "norm_cost_energy": round(float(result.norm_cost_energy), 6),
        "norm_cost_disruption": round(float(result.norm_cost_disruption), 6),
        "norm_gain_accuracy": round(float(result.norm_gain_accuracy), 6),
        "solve_time_s": round(float(result.solve_time), 4),
    }


def _drl_placement_to_row(
    placement: dict[str, tuple[str, str]],
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    mode: str,
    scenario_type: str,
    twin_is_safe: Optional[bool] = None,
    twin_predicted_j: Optional[float] = None,
    fallback_to_milp: bool = False,
    decision_time_ms: Optional[float] = None,
    inference_time_ms: Optional[float] = None,
    decoder_mode: Optional[str] = None,
    storm_safe: Optional[bool] = None,
    migration_count: Optional[int] = None,
    storm_max: Optional[int] = None,
) -> dict:
    """Evaluate a DRL/Hybrid placement and produce a notebook-compatible row."""
    obj = evaluate_objective_for_placement(ds, placement)
    e2e = _e2e_from_state(state)
    migrations = _migration_count_from_x_prev(ds, placement)
    effective_storm_max = int(getattr(ds, "v_storm_max", 0)) if storm_max is None else int(storm_max)
    effective_migration_count = migrations if migration_count is None else int(migration_count)
    effective_storm_safe = (
        effective_migration_count <= effective_storm_max
        if storm_safe is None and effective_storm_max >= 0
        else storm_safe
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "mode": mode,
        "scenario": scenario_type,
        "objective_J": round(obj.objective_j, 6),
        "e2e_latency_ms": round(e2e, 2),
        "energy_w": round(obj.cost_energy, 4),
        "migrations": migrations,
        "sla_violated": e2e > SLA_LATENCY_MS,
        "infeasible": False,
        "norm_cost_energy": round(obj.norm_cost_energy, 6),
        "norm_cost_disruption": round(obj.norm_cost_disruption, 6),
        "norm_gain_accuracy": round(obj.norm_gain_accuracy, 6),
        # DRL / Hybrid specific
        "twin_is_safe": twin_is_safe,
        "twin_predicted_j": round(twin_predicted_j, 6) if twin_predicted_j is not None else None,
        "hybrid_fell_back_to_milp": fallback_to_milp,
        "decision_time_ms": _round_ms(decision_time_ms),
        "inference_time_ms": _round_ms(inference_time_ms),
        "decoder_mode": decoder_mode,
        "storm_safe": effective_storm_safe,
        "migration_count": effective_migration_count,
        "storm_max": effective_storm_max,
    }


def _random_placement(rng: _random.Random) -> dict[str, tuple[str, str]]:
    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415

    return {
        "m0": ("standard", rng.choice(NODE_IDS)),
        "m1": ("standard", rng.choice(NODE_IDS)),
        "m2": ("standard", rng.choice(NODE_IDS)),
        "m3": (rng.choice(list(DETECTION_VARIANTS)), rng.choice(NODE_IDS)),
        "m4": (rng.choice(list(GEN_AI_VARIANTS)), rng.choice(NODE_IDS)),
        "m5": ("standard", rng.choice(NODE_IDS)),
    }


def _roundrobin_placement(cycle: int) -> dict[str, tuple[str, str]]:
    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415

    shift = (cycle - 1) % len(NODE_IDS)
    nodes = [NODE_IDS[(i + shift) % len(NODE_IDS)] for i in range(len(SERVICE_IDS))]
    det_var = list(DETECTION_VARIANTS)[0]   # lightest — deterministic
    gen_var = list(GEN_AI_VARIANTS)[0]
    return {
        "m0": ("standard", nodes[0]),
        "m1": ("standard", nodes[1]),
        "m2": ("standard", nodes[2]),
        "m3": (det_var, nodes[3]),
        "m4": (gen_var, nodes[4]),
        "m5": ("standard", nodes[5]),
    }


# ── Heuristic action generators (for candidate reranking) ────────────────────────────

def _cheapest_energy_action(state: np.ndarray) -> np.ndarray:
    """Place all services on the node with the lowest energy cost (state[8:12])."""
    cheapest = int(np.argmin(state[8:12]))
    cheapest = min(max(cheapest, 0), 3)
    return np.array([
        cheapest,            # m0
        cheapest,            # m1
        cheapest,            # m2
        0 * 4 + cheapest,   # m3: yolo26-nano (variant 0)
        0 * 4 + cheapest,   # m4: qwen-1.5b-nano (variant 0)
        cheapest,            # m5
    ], dtype=np.int64)


def _highest_accuracy_action(state: np.ndarray) -> np.ndarray:
    """Use highest-accuracy variants; place on lowest-energy node."""
    best_node = int(np.argmin(state[8:12]))
    best_node = min(max(best_node, 0), 3)
    return np.array([
        best_node,            # m0
        best_node,            # m1
        best_node,            # m2
        2 * 4 + best_node,   # m3: yolo26-medium (variant 2, highest accuracy)
        2 * 4 + best_node,   # m4: gemma2-2b-medium (variant 2)
        best_node,            # m5
    ], dtype=np.int64)


def _current_placement_action(state: np.ndarray) -> np.ndarray:
    """Return an action that keeps all services at their current placement."""
    action = np.zeros(6, dtype=np.int64)
    det_slice = state[12:15]
    det_vi = int(np.argmax(det_slice)) if float(det_slice.max()) > 0.0 else 0
    for svc_idx in range(6):
        slot = 15 + svc_idx * 4
        ps = state[slot:slot + 4]
        node_idx = int(np.argmax(ps)) if float(ps.max()) > 0.0 else 0
        if svc_idx == 3:    # m3 detection
            action[svc_idx] = det_vi * 4 + node_idx
        elif svc_idx == 4:  # m4 gen-ai — variant not in state, default to 0
            action[svc_idx] = 0 * 4 + node_idx
        else:
            action[svc_idx] = node_idx
    return action


# ── DRL capture helper ────────────────────────────────────────────────────────────

def _run_drl_and_capture(
    model,
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
) -> tuple[dict, list[int]]:
    """Run deterministic DRL inference; return (row, action_list) without re-inference."""
    try:
        start = time.perf_counter()
        action_arr, placement, meta = _predict_stormsafe(model, ds, state)
        decision_ms = _elapsed_ms(start)
        row = _drl_placement_to_row(
            placement,
            ds,
            state,
            cycle,
            "drl",
            scenario_type,
            decision_time_ms=decision_ms,
            inference_time_ms=decision_ms,
            decoder_mode=meta.get("decoder_mode"),
            storm_safe=meta.get("storm_safe"),
            migration_count=meta.get("migration_count"),
            storm_max=meta.get("storm_max"),
        )
        return row, action_arr.tolist()
    except Exception as exc:
        log.warning("DRL inference+capture failed cycle=%d: %s", cycle, exc)
        e2e = _e2e_from_state(state)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle, "mode": "drl", "scenario": scenario_type,
            "objective_J": None, "e2e_latency_ms": e2e,
            "energy_w": None, "migrations": None,
            "sla_violated": e2e > SLA_LATENCY_MS, "infeasible": True,
        }
        return row, []


# ── Regret report writer ────────────────────────────────────────────────────────────

def _write_regret_report(
    cycle_data: list[dict],
    results_dir: Path,
    top_n: int,
) -> None:
    """Write drl_regret_report.json (per-scenario summary) and drl_regret_cases.jsonl (top-N worst)."""
    from collections import defaultdict as _dd  # noqa: PLC0415

    valid = [r for r in cycle_data if r.get("gap_pct") is not None]
    if not valid:
        log.warning("No valid regret cases (need both milp + drl modes in the same run)")
        return

    by_scenario: dict[str, list[float]] = _dd(list)
    for r in valid:
        by_scenario[r["scenario"]].append(r["gap_pct"])

    scenario_stats: dict[str, dict] = {}
    for scen, gaps in sorted(by_scenario.items()):
        arr = np.array(gaps)
        scenario_stats[scen] = {
            "count": int(len(gaps)),
            "mean_gap_pct": round(float(np.mean(arr)), 4),
            "p50_gap_pct":  round(float(np.percentile(arr, 50)), 4),
            "p95_gap_pct":  round(float(np.percentile(arr, 95)), 4),
            "positive_gap_rate": round(float(np.mean(arr > 0)), 4),
        }

    all_gaps = np.array([r["gap_pct"] for r in valid])
    report = {
        "global": {
            "n_valid_cycles": int(len(valid)),
            "mean_gap_pct":  round(float(np.mean(all_gaps)), 4),
            "p50_gap_pct":   round(float(np.percentile(all_gaps, 50)), 4),
            "p95_gap_pct":   round(float(np.percentile(all_gaps, 95)), 4),
            "positive_gap_rate": round(float(np.mean(all_gaps > 0)), 4),
        },
        "per_scenario": scenario_stats,
    }
    report_path = results_dir / "drl_regret_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Regret report -> %s", report_path)

    worst = sorted(valid, key=lambda r: r.get("gap_pct", 0.0), reverse=True)[:top_n]
    cases_path = results_dir / "drl_regret_cases.jsonl"
    with open(cases_path, "w", encoding="utf-8") as fh:
        for case in worst:
            fh.write(json.dumps(case, ensure_ascii=True) + "\n")
    log.info("Top-%d regret cases -> %s", top_n, cases_path)


# ── Per-mode runners ────────────────────────────────────────────────────────────

def run_milp_cycle(
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
) -> dict:
    try:
        result = solve_placement(ds)
    except Exception as exc:
        log.warning("MILP solve failed cycle=%d: %s", cycle, exc)
        result = None
    return _milp_result_to_row(result, ds, state, cycle, "milp", scenario_type)


def run_drl_cycle(
    model,
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
) -> dict:
    try:
        start = time.perf_counter()
        _, placement, meta = _predict_stormsafe(model, ds, state)
        decision_ms = _elapsed_ms(start)
        return _drl_placement_to_row(
            placement,
            ds,
            state,
            cycle,
            "drl",
            scenario_type,
            decision_time_ms=decision_ms,
            inference_time_ms=decision_ms,
            decoder_mode=meta.get("decoder_mode"),
            storm_safe=meta.get("storm_safe"),
            migration_count=meta.get("migration_count"),
            storm_max=meta.get("storm_max"),
        )
    except Exception as exc:
        log.warning("DRL inference failed cycle=%d: %s", cycle, exc)
        e2e = _e2e_from_state(state)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle, "mode": "drl", "scenario": scenario_type,
            "objective_J": None, "e2e_latency_ms": e2e,
            "energy_w": None, "migrations": None,
            "sla_violated": e2e > SLA_LATENCY_MS, "infeasible": True,
        }


def run_hybrid_cycle(
    model,
    twin: DigitalTwinValidator,
    milp_result,
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
    use_reranking: bool = False,
    n_stochastic: int = 5,
    hybrid_epsilon: float = 0.0,
    rolling_milp_j: Optional[float] = None,
) -> dict:
    """Pick min(J_DRL, J_MILP) after Digital Twin safety validation.

    When use_reranking=True, builds a candidate set (PPO deterministic +
    stochastic + heuristics) and lets find_best_action() pick the best.

    Selection logic:
      1. Twin safety gate: if DRL placement is unsafe (low feasible_rate) or
         high uncertainty (std_reward > 0.25) → MILP wins by default.
      2. If DRL passes safety gate: compute actual J of DRL placement via
         evaluate_objective_for_placement() and compare with MILP J.
         Pick whichever has the lower (better) J.
    """
    try:
        start = time.perf_counter()
        det_action, det_placement, decoder_meta = _predict_stormsafe(model, ds, state)
        decoder_ms = _elapsed_ms(start)

        if use_reranking:
            candidates: list[np.ndarray] = [det_action]
            for _ in range(n_stochastic):
                stoch_a, _ = model.predict(state, deterministic=False)
                candidates.append(np.asarray(stoch_a, dtype=np.int64))
            candidates.append(_cheapest_energy_action(state))
            candidates.append(_highest_accuracy_action(state))
            candidates.append(_current_placement_action(state))
            best_action, twin_result = twin.find_best_action(candidates, state)
        else:
            best_action = det_action
            twin_result = twin.validate(best_action, state)

        drl_placement = (
            det_placement
            if np.array_equal(best_action, det_action)
            else _decode_drl_action(best_action)
        )

        # ── Step 1: Twin safety gate ──────────────────────────────────────────
        twin_unsafe = not twin_result.is_safe
        twin_high_variance = twin_result.std_reward > 0.25

        if twin_unsafe or twin_high_variance:
            reason = "twin_unsafe" if twin_unsafe else "twin_high_variance"
            log.debug(
                "Hybrid cycle=%d: %s (feasible=%.0f%% std=%.3f) → MILP",
                cycle, reason, twin_result.feasible_rate * 100, twin_result.std_reward,
            )
            use_milp = True
        else:
            # ── Step 2: Direct J comparison — pick lower (better) J ──────────
            drl_j = float(evaluate_objective_for_placement(ds, drl_placement).objective_j)
            milp_j = float(milp_result.objective_value) if milp_result is not None else None

            if milp_j is not None and milp_j < drl_j:
                log.debug(
                    "Hybrid cycle=%d: MILP wins J comparison (milp=%.4f < drl=%.4f)",
                    cycle, milp_j, drl_j,
                )
                use_milp = True
            else:
                log.debug(
                    "Hybrid cycle=%d: DRL wins J comparison (drl=%.4f)",
                    cycle, drl_j,
                )
                use_milp = False

        # ── Step 3: Commit chosen placement ──────────────────────────────────
        if not use_milp:
            return _drl_placement_to_row(
                drl_placement, ds, state, cycle, "hybrid", scenario_type,
                twin_is_safe=twin_result.is_safe,
                twin_predicted_j=twin_result.predicted_j,
                fallback_to_milp=False,
                decision_time_ms=decoder_ms,
                inference_time_ms=decoder_ms,
                decoder_mode=decoder_meta.get("decoder_mode"),
                storm_safe=decoder_meta.get("storm_safe"),
                migration_count=decoder_meta.get("migration_count"),
                storm_max=decoder_meta.get("storm_max"),
            )
        else:
            log.debug(
                "Hybrid cycle=%d: MILP fallback (twin_safe=%s)",
                cycle, twin_result.is_safe,
            )
            if milp_result is not None:
                milp_placement = {
                    svc_id: (var, node)
                    for svc_id, (var, node) in milp_result.placement.items()
                }
                return _drl_placement_to_row(
                    milp_placement, ds, state, cycle, "hybrid", scenario_type,
                    twin_is_safe=twin_result.is_safe,
                    twin_predicted_j=twin_result.predicted_j,
                    fallback_to_milp=True,
                    decision_time_ms=decoder_ms,
                    inference_time_ms=decoder_ms,
                    decoder_mode=(
                        "milp_wins_j_compare"
                        if not (twin_unsafe or twin_high_variance)
                        else "milp_fallback_twin_rejected"
                    ),
                    storm_safe=_migration_count_from_x_prev(ds, milp_placement) <= int(getattr(ds, "v_storm_max", 0)),
                    migration_count=_migration_count_from_x_prev(ds, milp_placement),
                    storm_max=int(getattr(ds, "v_storm_max", 0)),
                )
            else:
                e2e = _e2e_from_state(state)
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cycle": cycle, "mode": "hybrid", "scenario": scenario_type,
                    "objective_J": None, "e2e_latency_ms": e2e,
                    "energy_w": None, "migrations": None,
                    "sla_violated": e2e > SLA_LATENCY_MS, "infeasible": True,
                    "twin_is_safe": False,
                    "twin_predicted_j": twin_result.predicted_j,
                    "hybrid_fell_back_to_milp": True,
                }
    except Exception as exc:
        log.warning("Hybrid cycle failed cycle=%d: %s", cycle, exc)
        return _milp_result_to_row(milp_result, ds, state, cycle, "hybrid", scenario_type)


# ── K8s default scheduler simulation ─────────────────────────────────────────
# Variant profiles for the two baseline modes:
#   light   — lightest AI models (low resource, low accuracy)
#   quality — highest-accuracy AI models that still fit in edge memory
_K8S_DEFAULT_VARIANT_PROFILES: dict[str, dict[str, str]] = {
    "k8s_default_light": {
        "m3": "yolo26-nano",    # detection: lightest
        "m4": "qwen-1.5b-nano", # gen-ai: lightest
    },
    "k8s_default_quality": {
        "m3": "yolo26-medium",  # detection: highest accuracy
        "m4": "llama-3b-small", # gen-ai: best quality
    },
}

_K8S_DECISION_LATENCY_MS: float = 50.0  # typical kube-scheduler bind latency (ms)


def _k8s_default_schedule(
    ds: MILPDataset,
    variant_profile: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """Simulate K8s LeastAllocated scheduling for all 6 services.

    Greedy: assign services in dataset order (m0..m5).  Each assignment
    updates the remaining CPU and RAM headroom so subsequent services see
    the correct available capacity — mirrors the kube-scheduler serial Bind
    loop.

    Scoring (mirrors KubeScheduler LeastAllocatedPriority plugin):
        score(n) = (cpu_free(n)/cpu_cap(n) + mem_free(n)/mem_cap(n)) / 2

    Filter: prefer nodes where cpu_free >= r_cpu AND mem_free >= r_mem;
            fall back to all nodes if none are feasible (best-effort /
            over-commit, matching K8s behaviour when requests are not set).
    """
    cpu_avail = {n.node_id: n.cap_cpu     for n in ds.nodes}
    mem_avail = {n.node_id: n.cap_mem_gb  for n in ds.nodes}
    cpu_cap   = {n.node_id: max(n.cap_cpu,    1e-6) for n in ds.nodes}
    mem_cap   = {n.node_id: max(n.cap_mem_gb, 1e-6) for n in ds.nodes}
    node_ids  = [n.node_id for n in ds.nodes]

    placement: dict[str, tuple[str, str]] = {}

    for svc in ds.services:
        svc_id  = svc.service_id
        variant = variant_profile.get(svc_id, "standard")
        if variant not in svc.valid_variants:
            variant = svc.valid_variants[0]  # safe fallback

        r_cpu = ds.r_req.get((svc_id, variant), 0.1)
        r_mem = ds.r_mem.get((svc_id, variant), 0.1)

        # Filter phase: nodes with sufficient remaining CPU and RAM
        feasible = [
            n for n in node_ids
            if cpu_avail[n] >= r_cpu and mem_avail[n] >= r_mem
        ]
        if not feasible:
            feasible = node_ids  # over-commit fallback

        # Score phase: LeastAllocated — prefer nodes with most free resources
        best_node = max(
            feasible,
            key=lambda n: (
                cpu_avail[n] / cpu_cap[n] + mem_avail[n] / mem_cap[n]
            ) / 2.0,
        )

        placement[svc_id] = (variant, best_node)
        cpu_avail[best_node] = max(0.0, cpu_avail[best_node] - r_cpu)
        mem_avail[best_node] = max(0.0, mem_avail[best_node] - r_mem)

    return placement


def run_k8s_default_cycle(
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
    mode: str,
) -> dict:
    """Simulate K8s default (LeastAllocated) scheduling and evaluate with
    the same objective function used by MILP and DRL."""
    profile   = _K8S_DEFAULT_VARIANT_PROFILES[mode]
    placement = _k8s_default_schedule(ds, profile)
    row = _drl_placement_to_row(placement, ds, state, cycle, mode, scenario_type)
    row["inference_time_ms"] = _K8S_DECISION_LATENCY_MS
    return row


def run_random_cycle(
    rng: _random.Random,
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
) -> dict:
    placement = _random_placement(rng)
    return _drl_placement_to_row(placement, ds, state, cycle, "random", scenario_type)


def run_roundrobin_cycle(
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
) -> dict:
    placement = _roundrobin_placement(cycle)
    return _drl_placement_to_row(placement, ds, state, cycle, "roundrobin", scenario_type)


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline scenario-driven comparison (no Redis/K8s required)."
    )
    parser.add_argument("--cycles", type=int, default=200,
                        help="Number of scenario cycles to evaluate per mode (default: 200)")
    parser.add_argument(
        "--modes", nargs="+", default=MODES, choices=ALL_MODES,
        help="Which modes to evaluate (default: milp drl hybrid)",
    )
    parser.add_argument(
        "--scenario", default=None,
        choices=[s.value for s in ScenarioType] + ["mixed"],
        help="Fix a single scenario type, or 'mixed' for random sampling (default: mixed)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--model", default=None,
        help="Path to PPO model .zip (default: src/drl/models/ppo_simulator.zip)",
    )
    parser.add_argument(
        "--twin-rollouts", type=int, default=10,
        help="Digital Twin MC rollouts per hybrid cycle (default: 10; reduce for speed)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for .jsonl files (default: results/)",
    )
    parser.add_argument(
        "--out-prefix", default="off_comparison",
        help="Output filename prefix (default: off_comparison → off_comparison_{mode}.jsonl; "
             "use 'comparison' to write comparison_{mode}.jsonl instead)",
    )
    parser.add_argument(
        "--no-milp", action="store_true",
        help="Skip MILP solver (fast: DRL/Hybrid use reward-fn only, no LP)",
    )
    parser.add_argument(
        "--dump-state-action", action="store_true",
        help="Record per-cycle state, DRL action and MILP placement for regret analysis. "
             "Writes results/drl_regret_cases.jsonl and results/drl_regret_report.json. "
             "Requires both 'milp' and 'drl' in --modes.",
    )
    parser.add_argument(
        "--reranking", action="store_true",
        help="Enable candidate reranking in hybrid mode: generates stochastic PPO samples "
             "plus heuristic actions, evaluated by Digital Twin to pick the best.",
    )
    parser.add_argument(
        "--n-stochastic", type=int, default=5,
        help="Number of stochastic PPO samples for candidate reranking (default: 5).",
    )
    parser.add_argument(
        "--hybrid-epsilon", type=float, default=0.03,
        help="Regret-based fallback threshold ε: if twin-predicted J > rolling_milp_J×(1+ε), "
             "fall back to MILP. Set to 0 to disable (default: 0.03).",
    )
    parser.add_argument(
        "--top-n-regret", type=int, default=100,
        help="Top-N worst DRL-vs-MILP gap cycles to include in drl_regret_cases.jsonl (default: 100).",
    )
    parser.add_argument(
        "--from-states",
        default=None,
        help="Path to JSONL file of pre-captured live states (from live_state_capture.py). "
             "When provided, replays these states instead of generating synthetic ones. "
             "--cycles is ignored; the file length determines the number of cycles.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    model_path = args.model or str(
        SRC_ROOT / "drl" / "models" / "ppo_scenarios.zip"
    )

    # ── Load PPO model (only if DRL/Hybrid modes requested) ──────────────────
    needs_drl = any(m in args.modes for m in ("drl", "hybrid"))
    model = None
    if needs_drl:
        log.info("Loading PPO model from %s", model_path)
        try:
            from stable_baselines3 import PPO  # noqa: PLC0415
            model = PPO.load(model_path)
            log.info("PPO model loaded OK (standard PPO)")
        except Exception as exc1:
            log.warning("Standard PPO load failed (%s), trying MaskablePPO ...", exc1)
            try:
                from sb3_contrib import MaskablePPO  # noqa: PLC0415
                model = MaskablePPO.load(model_path)
                log.info("PPO model loaded OK (MaskablePPO)")
            except Exception as exc2:
                log.error("Cannot load PPO model: %s", exc2)
                log.error("DRL/Hybrid modes will be skipped. Pass a valid --model path.")
                args.modes = [m for m in args.modes if m not in ("drl", "hybrid")]

    # ── Digital Twin (only for hybrid) ───────────────────────────────────────
    twin: Optional[DigitalTwinValidator] = None
    if "hybrid" in args.modes:
        twin = DigitalTwinValidator(
            n_rollouts=args.twin_rollouts,
            seed=args.seed,
        )
        log.info("DigitalTwinValidator ready (n_rollouts=%d)", args.twin_rollouts)

    # ── Pre-captured live states (Phase B replay) ────────────────────────────
    _live_states: list[dict] = []
    if args.from_states:
        from_path = Path(args.from_states)
        if not from_path.exists():
            log.error("--from-states file not found: %s", from_path)
            sys.exit(1)
        with open(from_path) as fh:
            _live_states = [json.loads(line) for line in fh if line.strip()]
        log.info("Loaded %d pre-captured live states from %s", len(_live_states), from_path)
        args.cycles = len(_live_states)  # override cycles to match file length

    # ── Scenario generator ────────────────────────────────────────────────────
    gen = ScenarioGenerator(seed=args.seed)
    fixed_scenario: Optional[ScenarioType] = None
    if args.scenario and args.scenario != "mixed":
        fixed_scenario = ScenarioType(args.scenario)
        log.info("Fixed scenario: %s", fixed_scenario.value)
    else:
        log.info("Mixed scenario sampling (weighted random)")

    rng = _random.Random(args.seed)

    # Pre-build scenario sampling weights so we can track the actual scenario
    # type that was sampled each cycle (ScenarioGenerator.sample_state() selects
    # internally and does not return the chosen type).
    _scenario_types = gen._scenario_types
    _scenario_weights = np.asarray(gen._weights, dtype=np.float64)
    _scenario_weights /= _scenario_weights.sum()
    _scenario_rng = np.random.default_rng(args.seed + 1)

    # ── Open output files ─────────────────────────────────────────────────────
    out_handles = {}
    for mode in args.modes:
        path = results_dir / f"{args.out_prefix}_{mode}.jsonl"
        out_handles[mode] = open(path, "w", encoding="utf-8")  # noqa: SIM115
        log.info("Output: %s", path)

    log.info(
        "Starting offline comparison — modes=%s cycles=%d reranking=%s epsilon=%.2f",
        args.modes, args.cycles, args.reranking, args.hybrid_epsilon,
    )
    t_start = time.perf_counter()

    # ── Regret-mining state ───────────────────────────────────────────────────
    _cycle_data: list[dict] = []           # per-cycle records (dump-state-action)
    _milp_j_history: list[float] = []     # rolling window for MILP J baseline
    _rolling_milp_j: Optional[float] = None

    try:
        for cycle in range(1, args.cycles + 1):
            # ── 1. Sample scenario state ──────────────────────────────────────
            if _live_states:
                # Replay pre-captured live state
                rec = _live_states[cycle - 1]
                state = np.asarray(rec["state"], dtype=np.float32)
                scenario_type = "idle_cluster"  # live KWOK state = idle pattern
            elif fixed_scenario:
                sampled_scenario = fixed_scenario
                state = gen.sample_state(scenario=sampled_scenario)
                scenario_type = sampled_scenario.value
            else:
                idx = int(_scenario_rng.choice(len(_scenario_types), p=_scenario_weights))
                sampled_scenario = _scenario_types[idx]
                state = gen.sample_state(scenario=sampled_scenario)
                scenario_type = sampled_scenario.value

            # ── 2. Build MILPDataset from state ──────────────────────────────
            ds = _build_milp_dataset_from_state(state)

            # ── 3. Solve MILP once per cycle (shared across milp + hybrid) ────
            milp_result = None
            if ("milp" in args.modes or "hybrid" in args.modes) and not args.no_milp:
                try:
                    milp_result = solve_placement(ds)
                except Exception as exc:
                    log.warning("MILP solve failed cycle=%d: %s", cycle, exc)

            # Update rolling MILP J baseline (50-cycle window)
            if milp_result is not None and milp_result.objective_value is not None:
                _milp_j_history.append(float(milp_result.objective_value))
                if len(_milp_j_history) > 50:
                    _milp_j_history.pop(0)
                _rolling_milp_j = float(np.mean(_milp_j_history))

            # Prepare per-cycle record for regret mining
            _cycle_rec: dict = {
                "cycle": cycle,
                "scenario": scenario_type,
                "milp_j": float(milp_result.objective_value) if milp_result else None,
            }
            if args.dump_state_action:
                _cycle_rec["state"] = state.tolist()
                if milp_result is not None:
                    _cycle_rec["milp_placement"] = {
                        k: list(v) for k, v in milp_result.placement.items()
                    }

            # ── 4. Evaluate each requested mode ──────────────────────────────
            for mode in args.modes:
                if mode == "milp":
                    row = _milp_result_to_row(milp_result, ds, state, cycle, "milp", scenario_type)

                elif mode == "drl":
                    if args.dump_state_action:
                        row, captured_action = _run_drl_and_capture(
                            model, ds, state, cycle, scenario_type
                        )
                        _cycle_rec["drl_j"] = row.get("objective_J")
                        _cycle_rec["drl_action"] = captured_action
                    else:
                        row = run_drl_cycle(model, ds, state, cycle, scenario_type)

                elif mode == "hybrid":
                    row = run_hybrid_cycle(
                        model, twin, milp_result, ds, state, cycle, scenario_type,
                        use_reranking=args.reranking,
                        n_stochastic=args.n_stochastic,
                        hybrid_epsilon=args.hybrid_epsilon,
                        rolling_milp_j=_rolling_milp_j if args.hybrid_epsilon > 0 else None,
                    )

                elif mode == "random":
                    row = run_random_cycle(rng, ds, state, cycle, scenario_type)

                elif mode == "roundrobin":
                    row = run_roundrobin_cycle(ds, state, cycle, scenario_type)

                elif mode in ("k8s_default_light", "k8s_default_quality"):
                    row = run_k8s_default_cycle(ds, state, cycle, scenario_type, mode)

                else:
                    continue

                out_handles[mode].write(json.dumps(row, ensure_ascii=True) + "\n")
                out_handles[mode].flush()

            # Finalise regret record for this cycle
            if args.dump_state_action and _cycle_rec.get("milp_j") is not None and "drl_j" in _cycle_rec:
                drl_j = _cycle_rec.get("drl_j")
                milp_j = _cycle_rec["milp_j"]
                if drl_j is not None:
                    gap = drl_j - milp_j
                    denom = max(abs(milp_j), 1e-9)
                    _cycle_rec["gap"] = round(gap, 6)
                    _cycle_rec["gap_pct"] = round(gap / denom * 100, 4)
                _cycle_data.append(_cycle_rec)

            # ── 5. Progress log every 25 cycles ──────────────────────────────
            if cycle % 25 == 0 or cycle == args.cycles:
                elapsed = time.perf_counter() - t_start
                log.info(
                    "cycle %d/%d  elapsed=%.1fs  scenario=%s  rolling_milp_J=%s",
                    cycle, args.cycles, elapsed, scenario_type,
                    f"{_rolling_milp_j:.4f}" if _rolling_milp_j is not None else "n/a",
                )

    finally:
        for handle in out_handles.values():
            handle.close()

    elapsed = time.perf_counter() - t_start
    log.info(
        "Done. %d cycles × %d modes in %.1f s (%.1f ms/cycle).",
        args.cycles, len(args.modes),
        elapsed, elapsed / max(args.cycles, 1) * 1000,
    )
    log.info("Results written to: %s", results_dir)

    # ── Write regret report (if requested) ───────────────────────────────────
    if args.dump_state_action:
        _write_regret_report(_cycle_data, results_dir, args.top_n_regret)


if __name__ == "__main__":
    main()
