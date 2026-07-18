#!/home/ubuntu/KLTN_project/.venv/bin/python
"""Offline comparison v2: MILP + multi-model DRL adapters (legacy + dynamic).

Key upgrades vs run_offline_comparison.py:
- Supports multiple models in one run (`--models ...`)
- Supports heterogeneous observation dimensions (e.g., 44-dim legacy, 96-dim dynamic)
- Supports both SB3 .zip and dynamic BC .pt checkpoints
- Keeps notebook-ready JSONL outputs per mode/model
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import random as _random
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _ensure_venv() -> None:
    import importlib.util
    if importlib.util.find_spec("stable_baselines3") is None and _VENV_PYTHON.exists():
        import subprocess
        result = subprocess.run([str(_VENV_PYTHON)] + sys.argv)
        sys.exit(result.returncode)


_ensure_venv()
for _p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _configure_runtime_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(max(1, int(os.getenv("DRL_TORCH_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "1")))))
    except Exception:
        pass
    try:
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


_configure_runtime_threads()

from drl.scenario_generator import ScenarioGenerator, ScenarioType
from drl.edge_env import NODE_IDS, build_mock_dataset, compute_action_masks
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
from solver.dataset_generator import MILPDataset
from solver.milp_model import solve_placement

SLA_LATENCY_MS = 1500.0
log = logging.getLogger("offline-comparison-v2")
ALL_MODES = [
    "milp",
    "drl",
    "hybrid",
    "random",
    "roundrobin",
    "k8s_default_light",
    "k8s_default_quality",
]

_K8S_DEFAULT_VARIANT_PROFILES: dict[str, dict[str, str]] = {
    "k8s_default_light": {
        "m3": "yolo26-nano",
        "m4": "qwen-1.5b-nano",
    },
    "k8s_default_quality": {
        "m3": "yolo26-medium",
        "m4": "llama-3b-small",
    },
}
_K8S_DECISION_LATENCY_MS: float = 50.0


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _round_ms(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def _stamp_timing(
    row: dict[str, Any],
    *,
    decision_time_ms: Optional[float] = None,
    timing_scope: Optional[str] = None,
) -> dict[str, Any]:
    if decision_time_ms is not None:
        row["decision_time_ms"] = _round_ms(decision_time_ms)
    if timing_scope is not None:
        row["timing_scope"] = timing_scope
    return row


def _stamp_decoder(
    row: dict[str, Any],
    *,
    decoder_mode: Optional[str] = None,
    storm_safe: Optional[bool] = None,
    migration_count: Optional[int] = None,
    storm_max: Optional[int] = None,
) -> dict[str, Any]:
    if decoder_mode is not None:
        row["decoder_mode"] = decoder_mode
    if storm_safe is not None:
        row["storm_safe"] = bool(storm_safe)
    if migration_count is not None:
        row["migration_count"] = int(migration_count)
    if storm_max is not None:
        row["storm_max"] = int(storm_max)
    return row


def _e2e_from_state(state: np.ndarray) -> float:
    e2e_raw = float(state[39])
    if 0.0 < e2e_raw < 5000.0:
        return e2e_raw
    max_cpu = float(np.max(state[0:4]))
    return 80.0 + 300.0 * max_cpu


def _build_milp_dataset_from_state(state: np.ndarray) -> MILPDataset:
    base = build_mock_dataset()

    e_cpu_unit = state[8:12].tolist()
    for i, node in enumerate(base.nodes):
        node.energy_cost = max(0.1, float(e_cpu_unit[i]))

    w_c = float(state[41])
    w_d = float(state[42])
    w_a = float(state[43])
    total = w_c + w_d + w_a
    if total > 0:
        w_c, w_d, w_a = w_c / total, w_d / total, w_a / total
    base.w_c, base.w_d, base.w_a = w_c, w_d, w_a

    cpu_util = state[0:4].tolist()
    for i, node in enumerate(base.nodes):
        occupied = float(cpu_util[i]) * node.cap_cpu
        node.cap_cpu = max(0.5, node.cap_cpu - occupied)

    mem_used_gb = state[4:8].tolist()
    for i, node in enumerate(base.nodes):
        node.cap_mem_gb = max(0.1, node.cap_mem_gb - float(mem_used_gb[i]))

    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415
    det_vars = list(DETECTION_VARIANTS)
    gen_vars = list(GEN_AI_VARIANTS)
    det_var_idx = int(np.argmax(state[12:15])) if state[12:15].sum() > 0 else 0

    for k in list(base.x_prev):
        base.x_prev[k] = 0.0

    for svc_idx, svc_id in enumerate(["m0", "m1", "m2", "m3", "m4", "m5"]):
        slot = state[15 + svc_idx * 4: 15 + svc_idx * 4 + 4]
        node_idx = int(np.argmax(slot)) if slot.sum() > 0 else 0
        node_id = NODE_IDS[node_idx]
        if svc_id == "m3":
            var = det_vars[det_var_idx]
        elif svc_id == "m4":
            var = gen_vars[0]
        else:
            var = "standard"
        if (svc_id, var, node_id) in base.x_prev:
            base.x_prev[(svc_id, var, node_id)] = 1.0

    return base


def _milp_result_to_row(
    result,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
    inference_time_ms: Optional[float] = None,
    decision_time_ms: Optional[float] = None,
    timing_scope: Optional[str] = None,
    decoder_mode: Optional[str] = None,
    storm_safe: Optional[bool] = None,
    migration_count: Optional[int] = None,
    storm_max: Optional[int] = None,
) -> dict[str, Any]:
    e2e = _e2e_from_state(state)
    if result is None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle,
            "mode": "milp",
            "scenario": scenario_type,
            "inference_time_ms": _round_ms(inference_time_ms),
            "objective_J": None,
            "e2e_latency_ms": e2e,
            "energy_w": None,
            "migrations": None,
            "sla_violated": e2e > SLA_LATENCY_MS,
            "infeasible": True,
        }
        _stamp_timing(row, decision_time_ms=decision_time_ms, timing_scope=timing_scope)
        return _stamp_decoder(
            row,
            decoder_mode=decoder_mode,
            storm_safe=storm_safe,
            migration_count=migration_count,
            storm_max=storm_max,
        )
    migrations = sum(1 for v in result.migration_types.values() if v != "Stayed")
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "mode": "milp",
        "scenario": scenario_type,
        "inference_time_ms": _round_ms(inference_time_ms),
        "objective_J": round(float(result.objective_value), 6),
        "e2e_latency_ms": round(e2e, 2),
        "energy_w": round(float(result.cost_energy), 4) if result.cost_energy is not None else None,
        "migrations": migrations,
        "sla_violated": e2e > SLA_LATENCY_MS,
        "infeasible": False,
        "norm_cost_energy": round(float(result.norm_cost_energy), 6),
        "norm_cost_disruption": round(float(result.norm_cost_disruption), 6),
        "norm_gain_accuracy": round(float(result.norm_gain_accuracy), 6),
        "solve_time_s": round(float(result.solve_time), 4),
    }
    _stamp_timing(row, decision_time_ms=decision_time_ms, timing_scope=timing_scope)
    return _stamp_decoder(
        row,
        decoder_mode=decoder_mode,
        storm_safe=storm_safe,
        migration_count=migration_count if migration_count is not None else migrations,
        storm_max=storm_max,
    )


def _placement_to_row(
    placement: dict[str, tuple[str, str]],
    ds: MILPDataset,
    state: np.ndarray,
    cycle: int,
    scenario_type: str,
    mode: str,
    model_name: Optional[str] = None,
    inference_time_ms: Optional[float] = None,
    decision_time_ms: Optional[float] = None,
    timing_scope: Optional[str] = None,
    decoder_mode: Optional[str] = None,
    storm_safe: Optional[bool] = None,
    migration_count: Optional[int] = None,
    storm_max: Optional[int] = None,
) -> dict[str, Any]:
    obj = evaluate_objective_for_placement(ds, placement)
    e2e = _e2e_from_state(state)
    migrations = _migration_count_from_x_prev(ds, placement)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "mode": mode,
        "model_name": model_name,
        "scenario": scenario_type,
        "inference_time_ms": _round_ms(inference_time_ms),
        "objective_J": round(obj.objective_j, 6),
        "e2e_latency_ms": round(e2e, 2),
        "energy_w": round(obj.cost_energy, 4),
        "migrations": migrations,
        "sla_violated": e2e > SLA_LATENCY_MS,
        "infeasible": False,
        "norm_cost_energy": round(obj.norm_cost_energy, 6),
        "norm_cost_disruption": round(obj.norm_cost_disruption, 6),
        "norm_gain_accuracy": round(obj.norm_gain_accuracy, 6),
    }
    _stamp_timing(row, decision_time_ms=decision_time_ms, timing_scope=timing_scope)
    return _stamp_decoder(
        row,
        decoder_mode=decoder_mode,
        storm_safe=storm_safe,
        migration_count=migration_count if migration_count is not None else migrations,
        storm_max=storm_max if storm_max is not None else int(getattr(ds, "v_storm_max", 0)),
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
    from drl.edge_env import SERVICE_IDS  # noqa: PLC0415
    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415

    shift = (cycle - 1) % len(NODE_IDS)
    nodes = [NODE_IDS[(i + shift) % len(NODE_IDS)] for i in range(len(SERVICE_IDS))]
    det_var = list(DETECTION_VARIANTS)[0]
    gen_var = list(GEN_AI_VARIANTS)[0]
    return {
        "m0": ("standard", nodes[0]),
        "m1": ("standard", nodes[1]),
        "m2": ("standard", nodes[2]),
        "m3": (det_var, nodes[3]),
        "m4": (gen_var, nodes[4]),
        "m5": ("standard", nodes[5]),
    }


def _k8s_default_schedule(
    ds: MILPDataset,
    variant_profile: dict[str, str],
) -> dict[str, tuple[str, str]]:
    cpu_avail = {n.node_id: n.cap_cpu for n in ds.nodes}
    mem_avail = {n.node_id: n.cap_mem_gb for n in ds.nodes}
    cpu_cap = {n.node_id: max(n.cap_cpu, 1e-6) for n in ds.nodes}
    mem_cap = {n.node_id: max(n.cap_mem_gb, 1e-6) for n in ds.nodes}
    node_ids = [n.node_id for n in ds.nodes]
    placement: dict[str, tuple[str, str]] = {}

    for svc in ds.services:
        svc_id = svc.service_id
        variant = variant_profile.get(svc_id, "standard")
        if variant not in svc.valid_variants:
            variant = svc.valid_variants[0]
        r_cpu = ds.r_req.get((svc_id, variant), 0.1)
        r_mem = ds.r_mem.get((svc_id, variant), 0.1)
        feasible = [n for n in node_ids if cpu_avail[n] >= r_cpu and mem_avail[n] >= r_mem]
        if not feasible:
            feasible = node_ids
        best_node = max(
            feasible,
            key=lambda n: (cpu_avail[n] / cpu_cap[n] + mem_avail[n] / mem_cap[n]) / 2.0,
        )
        placement[svc_id] = (variant, best_node)
        cpu_avail[best_node] = max(0.0, cpu_avail[best_node] - r_cpu)
        mem_avail[best_node] = max(0.0, mem_avail[best_node] - r_mem)

    return placement


def _decode_legacy_action(action: np.ndarray) -> dict[str, tuple[str, str]]:
    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415
    return decode_legacy44_action(
        action,
        node_ids=NODE_IDS,
        detection_variants=list(DETECTION_VARIANTS),
        gen_ai_variants=list(GEN_AI_VARIANTS),
    )


@dataclass
class DynamicEpisode:
    storm_max: int
    w_c: float
    w_d: float
    w_a: float
    nodes: list[dict[str, Any]]
    services: list[dict[str, Any]]


def _extract_prev_placement(ds: MILPDataset) -> dict[str, tuple[str, str]]:
    prev: dict[str, tuple[str, str]] = {}
    for svc in ds.services:
        found = None
        for var in svc.valid_variants:
            for n in ds.nodes:
                if ds.x_prev.get((svc.service_id, var, n.node_id), 0.0) > 0.5:
                    found = (var, n.node_id)
                    break
            if found is not None:
                break
        if found is None:
            found = (svc.valid_variants[0], ds.nodes[0].node_id)
        prev[svc.service_id] = found
    return prev


def _build_dynamic_episode(ds: MILPDataset) -> DynamicEpisode:
    prev = _extract_prev_placement(ds)
    nodes = [
        {
            "node_id": n.node_id,
            "cap_cpu": float(n.cap_cpu),
            "cap_mem_gb": float(n.cap_mem_gb),
            "energy_cost": float(n.energy_cost),
            "e_mem_unit": float(n.e_mem_unit),
        }
        for n in ds.nodes
    ]
    services = []
    for svc in ds.services:
        variants = [
            {
                "variant_id": v,
                "cpu": float(ds.r_req[(svc.service_id, v)]),
                "mem_gb": float(ds.r_mem[(svc.service_id, v)]),
                "acc": float(ds.acc[(svc.service_id, v)]),
            }
            for v in svc.valid_variants
        ]
        pvar, pnode = prev[svc.service_id]
        services.append(
            {
                "service_id": svc.service_id,
                "service_type": svc.service_type,
                "migration_cost": float(svc.migration_cost),
                "variants": variants,
                "prev_variant": pvar,
                "prev_node": pnode,
            }
        )
    return DynamicEpisode(
        storm_max=int(ds.v_storm_max),
        w_c=float(ds.w_c),
        w_d=float(ds.w_d),
        w_a=float(ds.w_a),
        nodes=nodes,
        services=services,
    )


def _build_index_maps(ep: DynamicEpisode) -> tuple[dict[str, int], dict[str, int]]:
    node_idx = {n["node_id"]: i for i, n in enumerate(ep.nodes)}
    svc_idx = {s["service_id"]: i for i, s in enumerate(ep.services)}
    return node_idx, svc_idx


def _action_index(variant_idx: int, node_idx: int, max_nodes: int) -> int:
    return variant_idx * max_nodes + node_idx


def _decode_action_index(action_idx: int, max_nodes: int) -> tuple[int, int]:
    return action_idx // max_nodes, action_idx % max_nodes


def _infer_dynamic_shape(obs_dim: int, action_dim: int) -> tuple[int, int]:
    # Obs formula from v2 pipeline: obs_dim = 12 + 9*N + 4*V, action_dim = N*V
    candidates: list[tuple[int, int]] = []
    for n in range(1, 65):
        if action_dim % n != 0:
            continue
        v = action_dim // n
        if 12 + 9 * n + 4 * v == obs_dim:
            candidates.append((n, v))
    if not candidates:
        raise ValueError(
            f"Cannot infer dynamic shape from obs_dim={obs_dim}, action_dim={action_dim}."
        )
    # Prefer lower max_nodes and realistic max_variants.
    candidates.sort(key=lambda x: (abs(x[0] - 8), x[1]))
    return candidates[0]


def _build_mask_for_service(
    ep: DynamicEpisode,
    service: dict[str, Any],
    residual_cpu: list[float],
    residual_mem: list[float],
    migration_used: int,
    max_nodes: int,
    max_variants: int,
) -> np.ndarray:
    action_dim = max_nodes * max_variants
    mask = np.zeros((action_dim,), dtype=np.float32)
    node_index, _ = _build_index_maps(ep)

    for vi, var in enumerate(service["variants"]):
        if vi >= max_variants:
            break
        for ni, node in enumerate(ep.nodes):
            if ni >= max_nodes:
                break
            delta_mig = 0
            if service["prev_variant"] != var["variant_id"] or service["prev_node"] != node["node_id"]:
                delta_mig = 1
            if migration_used + delta_mig > ep.storm_max:
                continue
            if residual_cpu[ni] + 1e-9 < float(var["cpu"]):
                continue
            if residual_mem[ni] + 1e-9 < float(var["mem_gb"]):
                continue
            mask[_action_index(vi, ni, max_nodes)] = 1.0

    if float(mask.sum()) <= 0.0:
        prev_node_idx = node_index.get(service["prev_node"], 0)
        prev_variant_idx = 0
        for i, v in enumerate(service["variants"]):
            if v["variant_id"] == service["prev_variant"]:
                prev_variant_idx = i
                break
        prev_variant_idx = min(prev_variant_idx, max_variants - 1)
        prev_node_idx = min(prev_node_idx, max_nodes - 1)
        mask[_action_index(prev_variant_idx, prev_node_idx, max_nodes)] = 1.0

    return mask


def _vectorize_state_for_service(
    ep: DynamicEpisode,
    service: dict[str, Any],
    step_idx: int,
    residual_cpu: list[float],
    residual_mem: list[float],
    migration_used: int,
    max_nodes: int,
    max_variants: int,
) -> np.ndarray:
    max_cap_cpu = max(n["cap_cpu"] for n in ep.nodes)
    max_cap_mem = max(n["cap_mem_gb"] for n in ep.nodes)
    max_energy = max(n["energy_cost"] for n in ep.nodes)
    max_mem_energy = max(max(n["e_mem_unit"] for n in ep.nodes), 1e-6)
    max_mig_cost = max(s["migration_cost"] for s in ep.services)

    feats: list[float] = []
    feats.extend([
        len(ep.nodes) / max(max_nodes, 1),
        len(ep.services) / max(len(ep.services), 1),
        ep.w_c,
        ep.w_d,
        ep.w_a,
        (ep.storm_max - migration_used) / max(ep.storm_max, 1),
        step_idx / max(len(ep.services) - 1, 1),
    ])

    for i in range(max_nodes):
        if i < len(ep.nodes):
            n = ep.nodes[i]
            used_cpu = max(0.0, float(n["cap_cpu"]) - residual_cpu[i])
            used_mem = max(0.0, float(n["cap_mem_gb"]) - residual_mem[i])
            feats.extend([
                float(n["cap_cpu"]) / max(max_cap_cpu, 1e-6),
                float(n["cap_mem_gb"]) / max(max_cap_mem, 1e-6),
                float(n["energy_cost"]) / max(max_energy, 1e-6),
                float(n["e_mem_unit"]) / max(max_mem_energy, 1e-6),
                residual_cpu[i] / max(float(n["cap_cpu"]), 1e-6),
                residual_mem[i] / max(float(n["cap_mem_gb"]), 1e-6),
                used_cpu / max(float(n["cap_cpu"]), 1e-6),
                used_mem / max(float(n["cap_mem_gb"]), 1e-6),
                1.0,
            ])
        else:
            feats.extend([0.0] * 9)

    prev_node_norm = 0.0
    for ni, node in enumerate(ep.nodes):
        if node["node_id"] == service["prev_node"]:
            prev_node_norm = ni / max(len(ep.nodes) - 1, 1)
            break

    for i in range(max_variants):
        if i < len(service["variants"]):
            v = service["variants"][i]
            is_prev = 1.0 if v["variant_id"] == service["prev_variant"] else 0.0
            feats.extend([
                float(v["cpu"]) / max(max_cap_cpu, 1e-6),
                float(v["mem_gb"]) / max(max_cap_mem, 1e-6),
                float(v["acc"]),
                is_prev,
            ])
        else:
            feats.extend([0.0] * 4)

    feats.extend([
        float(service["migration_cost"]) / max(max_mig_cost, 1e-6),
        1.0 if len(service["variants"]) > 1 else 0.0,
        prev_node_norm,
        1.0 if service["service_type"] in {"detection", "gen_ai"} else 0.0,
        len(service["variants"]) / max(max_variants, 1),
    ])

    return np.asarray(feats, dtype=np.float32)


class ModelAdapter:
    def __init__(self, model_name: str, model_path: Path) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.last_decision_meta: dict[str, Any] = {}
        self._last_inference_core_ms: Optional[float] = None

    def predict_placement(self, ds: MILPDataset, state_44: np.ndarray) -> dict[str, tuple[str, str]]:
        raise NotImplementedError

    def predict_action_and_placement(
        self, ds: MILPDataset, state_44: np.ndarray
    ) -> tuple[Optional[np.ndarray], dict[str, tuple[str, str]]]:
        """Return (action_array_or_None, placement).

        Legacy adapters return the raw 6-element action array needed by
        DigitalTwinValidator.  Dynamic adapters return None (no flat action
        encoding), in which case Twin validation is skipped and hybrid falls
        back to direct J comparison.
        """
        return None, self.predict_placement(ds, state_44)

    def measure_inference_time_ms(self, ds: MILPDataset, state_44: np.ndarray) -> float:
        if self._last_inference_core_ms is not None:
            return float(self._last_inference_core_ms)
        start = time.perf_counter()
        self.predict_placement(ds, state_44)
        self._last_inference_core_ms = _elapsed_ms(start)
        return float(self._last_inference_core_ms)


class LegacySB3Adapter(ModelAdapter):
    def __init__(self, model_name: str, model_path: Path, model: Any) -> None:
        super().__init__(model_name, model_path)
        self.model = model
        try:
            self.action_dims = [int(x) for x in np.asarray(model.action_space.nvec, dtype=np.int64).tolist()]
        except Exception:
            self.action_dims = [int(x) for x in LEGACY_ACTION_DIMS]

    def _head_logits(self, obs: np.ndarray, flat_mask: np.ndarray | None) -> list[np.ndarray]:
        try:
            with torch.inference_mode():
                obs_t, _ = self.model.policy.obs_to_tensor(obs)
                try:
                    features = self.model.policy.extract_features(obs_t)
                    latent_pi, _ = self.model.policy.mlp_extractor(features)
                    flat_logits = (
                        self.model.policy.action_net(latent_pi)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    return split_head_logits(flat_logits, action_dims=self.action_dims)
                except Exception:
                    pass

                dist = None
                if flat_mask is not None:
                    try:
                        dist = self.model.policy.get_distribution(
                            obs_t,
                            action_masks=flat_mask.reshape(1, -1).astype(bool),
                        )
                    except Exception:
                        dist = self.model.policy.get_distribution(obs_t)
                else:
                    dist = self.model.policy.get_distribution(obs_t)
                dists = getattr(dist, "distributions", None)
                if isinstance(dists, list) and len(dists) == len(self.action_dims):
                    out: list[np.ndarray] = []
                    for i, head in enumerate(dists):
                        if hasattr(head, "logits"):
                            logits = head.logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
                        elif hasattr(head, "probs"):
                            probs = head.probs.squeeze(0).detach().cpu().numpy().astype(np.float64)
                            logits = np.log(np.clip(probs, 1e-12, 1.0))
                        else:
                            raise ValueError("Head distribution has neither logits nor probs")
                        if logits.shape[0] != self.action_dims[i]:
                            raise ValueError("Head logit dimension mismatch")
                        out.append(logits)
                    return out
        except Exception:
            pass

        # Fallback: deterministic action as one-hot logits.
        action, _ = self.model.predict(obs, deterministic=True)
        act = np.asarray(action, dtype=np.int64).reshape(-1)
        heads: list[np.ndarray] = []
        for i, dim in enumerate(self.action_dims):
            h = np.full((dim,), -1e9, dtype=np.float64)
            idx = int(act[i]) if i < len(act) else 0
            if 0 <= idx < dim:
                h[idx] = 0.0
            else:
                h[0] = 0.0
            heads.append(h)
        return heads

    def _predict_stormsafe(
        self,
        ds: MILPDataset,
        state_44: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, tuple[str, str]]]:
        state = np.asarray(state_44, dtype=np.float32).reshape(-1)
        flat_mask = compute_action_masks(state)
        head_logits = self._head_logits(state, flat_mask)
        masked_logits = apply_flat_mask_to_head_logits(
            head_logits=head_logits,
            flat_mask=flat_mask,
            action_dims=self.action_dims,
        )
        outcome = select_stormsafe_action(
            dataset=ds,
            head_logits=masked_logits,
            decode_action_fn=_decode_legacy_action,
            objective_fn=evaluate_objective_for_placement,
            action_dims=self.action_dims,
            topk_per_head=LEGACY_TOPK_PER_HEAD,
        )
        self.last_decision_meta = {
            "decoder_mode": outcome.decoder_mode,
            "storm_safe": True,
            "migration_count": int(outcome.migration_count),
            "storm_max": int(outcome.storm_max),
            "fallback_full_enumeration": bool(outcome.fallback_full_enumeration),
            "candidates_evaluated": int(outcome.candidates_evaluated),
            "feasible": bool(outcome.feasible),
            "objective_j_selected": float(outcome.objective_j),
        }
        return outcome.action, outcome.placement

    def predict_placement(self, ds: MILPDataset, state_44: np.ndarray) -> dict[str, tuple[str, str]]:
        t0 = time.perf_counter()
        _, placement = self._predict_stormsafe(ds, state_44)
        self._last_inference_core_ms = _elapsed_ms(t0)
        return placement

    def predict_action_and_placement(
        self, ds: MILPDataset, state_44: np.ndarray
    ) -> tuple[Optional[np.ndarray], dict[str, tuple[str, str]]]:
        t0 = time.perf_counter()
        action, placement = self._predict_stormsafe(ds, state_44)
        self._last_inference_core_ms = _elapsed_ms(t0)
        return np.asarray(action, dtype=np.int64), placement


class DynamicSB3Adapter(ModelAdapter):
    def __init__(self, model_name: str, model_path: Path, model: Any) -> None:
        super().__init__(model_name, model_path)
        self.model = model
        obs_dim = int(model.observation_space.shape[0])
        action_dim = int(model.action_space.n)
        self.max_nodes, self.max_variants = _infer_dynamic_shape(obs_dim, action_dim)

    def _action_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Prefer full distribution from MaskablePPO; fallback to near-greedy.
        try:
            with torch.inference_mode():
                obs_t, _ = self.model.policy.obs_to_tensor(obs)
                dist = self.model.policy.get_distribution(
                    obs_t,
                    action_masks=mask.reshape(1, -1).astype(bool),
                )
                probs = dist.distribution.probs.squeeze(0).detach().cpu().numpy().astype(np.float64)
                return probs
        except Exception:
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask.astype(bool))
            probs = np.zeros_like(mask, dtype=np.float64)
            idx = int(action)
            if 0 <= idx < len(probs):
                probs[idx] = 1.0
            else:
                valid = np.where(mask > 0.5)[0]
                if len(valid):
                    probs[int(valid[0])] = 1.0
            return probs

    def _greedy_policy_placement(
        self,
        ds: MILPDataset,
        collect_policy_time: bool = False,
    ) -> dict[str, tuple[str, str]] | tuple[dict[str, tuple[str, str]], float]:
        ep = _build_dynamic_episode(ds)
        services = sorted(ep.services, key=lambda s: s["service_id"])
        residual_cpu = [n["cap_cpu"] for n in ep.nodes]
        residual_mem = [n["cap_mem_gb"] for n in ep.nodes]
        migration_used = 0
        placement: dict[str, tuple[str, str]] = {}
        policy_time_ms = 0.0

        for step_idx, svc in enumerate(services):
            mask = _build_mask_for_service(
                ep, svc, residual_cpu, residual_mem, migration_used,
                self.max_nodes, self.max_variants,
            )
            obs = _vectorize_state_for_service(
                ep, svc, step_idx, residual_cpu, residual_mem, migration_used,
                self.max_nodes, self.max_variants,
            )
            policy_start = time.perf_counter()
            probs = self._action_probs(obs, mask)
            if collect_policy_time:
                policy_time_ms += _elapsed_ms(policy_start)
            masked_probs = np.where(mask > 0.5, probs, -np.inf)
            action_idx = int(np.argmax(masked_probs))
            vi, ni = _decode_action_index(action_idx, self.max_nodes)

            if ni >= len(ep.nodes) or vi >= len(svc["variants"]):
                chosen_variant = svc["prev_variant"]
                chosen_node = svc["prev_node"]
            else:
                chosen_variant = svc["variants"][vi]["variant_id"]
                chosen_node = ep.nodes[ni]["node_id"]

            placement[svc["service_id"]] = (chosen_variant, chosen_node)
            var_obj = next((v for v in svc["variants"] if v["variant_id"] == chosen_variant), svc["variants"][0])
            node_idx = next((i for i, n in enumerate(ep.nodes) if n["node_id"] == chosen_node), 0)
            residual_cpu[node_idx] -= float(var_obj["cpu"])
            residual_mem[node_idx] -= float(var_obj["mem_gb"])
            if svc["prev_variant"] != chosen_variant or svc["prev_node"] != chosen_node:
                migration_used += 1

        if collect_policy_time:
            return placement, policy_time_ms
        return placement

    def measure_inference_time_ms(self, ds: MILPDataset, state_44: np.ndarray) -> float:
        if self._last_inference_core_ms is not None:
            return float(self._last_inference_core_ms)
        _, policy_time_ms = self._greedy_policy_placement(ds, collect_policy_time=True)
        self._last_inference_core_ms = float(policy_time_ms)
        return float(policy_time_ms)

    def predict_placement(self, ds: MILPDataset, state_44: np.ndarray) -> dict[str, tuple[str, str]]:
        t0 = time.perf_counter()
        ep = _build_dynamic_episode(ds)
        services = sorted(ep.services, key=lambda s: s["service_id"])

        @dataclass
        class BeamState:
            placement: dict[str, tuple[str, str]]
            residual_cpu: list[float]
            residual_mem: list[float]
            migration_used: int
            logp: float

        beam_width = 24
        topk_actions = 8
        beams = [
            BeamState(
                placement={},
                residual_cpu=[n["cap_cpu"] for n in ep.nodes],
                residual_mem=[n["cap_mem_gb"] for n in ep.nodes],
                migration_used=0,
                logp=0.0,
            )
        ]

        for step_idx, svc in enumerate(services):
            expanded: list[BeamState] = []
            for b in beams:
                mask = _build_mask_for_service(
                    ep, svc, b.residual_cpu, b.residual_mem, b.migration_used,
                    self.max_nodes, self.max_variants,
                )
                obs = _vectorize_state_for_service(
                    ep, svc, step_idx, b.residual_cpu, b.residual_mem, b.migration_used,
                    self.max_nodes, self.max_variants,
                )
                probs = self._action_probs(obs, mask)
                valid_count = int((mask > 0.5).sum())
                if valid_count <= 0:
                    continue
                k = max(1, min(topk_actions, valid_count))
                top_idx = np.argpartition(-probs, k - 1)[:k]
                top_idx = top_idx[np.argsort(-probs[top_idx])]

                for action_idx in top_idx.tolist():
                    if mask[action_idx] < 0.5:
                        continue
                    vi, ni = _decode_action_index(int(action_idx), self.max_nodes)
                    if ni >= len(ep.nodes) or vi >= len(svc["variants"]):
                        continue
                    chosen_var = svc["variants"][vi]
                    chosen_node = ep.nodes[ni]["node_id"]
                    nxt = BeamState(
                        placement=dict(b.placement),
                        residual_cpu=list(b.residual_cpu),
                        residual_mem=list(b.residual_mem),
                        migration_used=b.migration_used,
                        logp=b.logp + math.log(max(float(probs[action_idx]), 1e-12)),
                    )
                    nxt.placement[svc["service_id"]] = (chosen_var["variant_id"], chosen_node)
                    nxt.residual_cpu[ni] -= float(chosen_var["cpu"])
                    nxt.residual_mem[ni] -= float(chosen_var["mem_gb"])
                    if svc["prev_variant"] != chosen_var["variant_id"] or svc["prev_node"] != chosen_node:
                        nxt.migration_used += 1
                    expanded.append(nxt)

            if not expanded:
                break
            expanded.sort(key=lambda x: x.logp, reverse=True)
            beams = expanded[: max(1, beam_width)]

        reward_ds = ds
        for b in beams:
            for svc in services:
                if svc["service_id"] not in b.placement:
                    b.placement[svc["service_id"]] = (svc["prev_variant"], svc["prev_node"])

        best = None
        best_key = None
        for b in beams:
            try:
                obj = evaluate_objective_for_placement(reward_ds, b.placement).objective_j
            except Exception:
                obj = float("inf")
            key = (obj, -b.logp)
            if best is None or key < best_key:
                best = b
                best_key = key

        if best is None:
            placement = {svc["service_id"]: (svc["prev_variant"], svc["prev_node"]) for svc in services}
            self._last_inference_core_ms = _elapsed_ms(t0)
            self.last_decision_meta = {
                "decoder_mode": "dynamic96_beam",
                "storm_safe": True,
                "migration_count": None,
                "storm_max": int(getattr(ds, "v_storm_max", 0)),
            }
            return placement
        self._last_inference_core_ms = _elapsed_ms(t0)
        self.last_decision_meta = {
            "decoder_mode": "dynamic96_beam",
            "storm_safe": True,
            "migration_count": None,
            "storm_max": int(getattr(ds, "v_storm_max", 0)),
        }
        return best.placement


class DynamicBCAdapter(ModelAdapter):
    def __init__(self, model_name: str, model_path: Path) -> None:
        super().__init__(model_name, model_path)
        from src.drl.dynamic_v2_pipeline import DynamicSequentialPolicy  # noqa: PLC0415

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        fc = ckpt["feature_config"]
        self.max_nodes = int(fc["max_nodes"])
        self.max_variants = int(fc["max_variants"])
        input_dim = int(fc["input_dim"])
        action_dim = int(fc["action_dim"])
        self.model = DynamicSequentialPolicy(input_dim=input_dim, action_dim=action_dim)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _greedy_policy_placement(
        self,
        ds: MILPDataset,
        collect_policy_time: bool = False,
    ) -> dict[str, tuple[str, str]] | tuple[dict[str, tuple[str, str]], float]:
        ep = _build_dynamic_episode(ds)
        services = sorted(ep.services, key=lambda s: s["service_id"])
        residual_cpu = [n["cap_cpu"] for n in ep.nodes]
        residual_mem = [n["cap_mem_gb"] for n in ep.nodes]
        migration_used = 0
        placement: dict[str, tuple[str, str]] = {}
        node_map, _ = _build_index_maps(ep)
        policy_time_ms = 0.0
        for step_idx, svc in enumerate(services):
            mask = _build_mask_for_service(ep, svc, residual_cpu, residual_mem, migration_used, self.max_nodes, self.max_variants)
            obs = _vectorize_state_for_service(ep, svc, step_idx, residual_cpu, residual_mem, migration_used, self.max_nodes, self.max_variants)
            with torch.no_grad():
                policy_start = time.perf_counter()
                logits = self.model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                if collect_policy_time:
                    policy_time_ms += _elapsed_ms(policy_start)
                logits = logits.masked_fill(torch.tensor(mask, dtype=torch.float32).unsqueeze(0) <= 0.5, -1e9)
                action_idx = int(torch.argmax(logits, dim=1).item())
            vi, ni = _decode_action_index(action_idx, self.max_nodes)
            if ni >= len(ep.nodes) or vi >= len(svc["variants"]):
                chosen_variant = svc["prev_variant"]
                chosen_node = svc["prev_node"]
            else:
                chosen_variant = svc["variants"][vi]["variant_id"]
                chosen_node = ep.nodes[ni]["node_id"]
            placement[svc["service_id"]] = (chosen_variant, chosen_node)
            var_obj = next((v for v in svc["variants"] if v["variant_id"] == chosen_variant), svc["variants"][0])
            idx = node_map[chosen_node]
            residual_cpu[idx] -= float(var_obj["cpu"])
            residual_mem[idx] -= float(var_obj["mem_gb"])
            if svc["prev_variant"] != chosen_variant or svc["prev_node"] != chosen_node:
                migration_used += 1
        if collect_policy_time:
            return placement, policy_time_ms
        return placement

    def measure_inference_time_ms(self, ds: MILPDataset, state_44: np.ndarray) -> float:
        if self._last_inference_core_ms is not None:
            return float(self._last_inference_core_ms)
        _, policy_time_ms = self._greedy_policy_placement(ds, collect_policy_time=True)
        self._last_inference_core_ms = float(policy_time_ms)
        return float(policy_time_ms)

    def predict_placement(self, ds: MILPDataset, state_44: np.ndarray) -> dict[str, tuple[str, str]]:
        t0 = time.perf_counter()
        placement, policy_time_ms = self._greedy_policy_placement(ds, collect_policy_time=True)
        self._last_inference_core_ms = float(policy_time_ms)
        self.last_decision_meta = {
            "decoder_mode": "dynamic96_greedy",
            "storm_safe": True,
            "migration_count": None,
            "storm_max": int(getattr(ds, "v_storm_max", 0)),
            "policy_wall_time_ms": _elapsed_ms(t0),
        }
        return placement


class LegacyTorchZipAdapter(ModelAdapter):
    """Fallback legacy adapter for old SB3 zip artifacts that cannot be deserialized."""

    def __init__(self, model_name: str, model_path: Path) -> None:
        super().__init__(model_name, model_path)
        with zipfile.ZipFile(model_path, "r") as zf:
            policy_blob = zf.read("policy.pth")
        sd = torch.load(io.BytesIO(policy_blob), map_location="cpu")
        required = [
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.policy_net.0.bias",
            "mlp_extractor.policy_net.2.weight",
            "mlp_extractor.policy_net.2.bias",
            "action_net.weight",
            "action_net.bias",
        ]
        missing = [k for k in required if k not in sd]
        if missing:
            raise ValueError(f"Legacy torch zip missing weights: {missing}")

        self.w1 = sd["mlp_extractor.policy_net.0.weight"].detach().float()
        self.b1 = sd["mlp_extractor.policy_net.0.bias"].detach().float()
        self.w2 = sd["mlp_extractor.policy_net.2.weight"].detach().float()
        self.b2 = sd["mlp_extractor.policy_net.2.bias"].detach().float()
        self.wa = sd["action_net.weight"].detach().float()
        self.ba = sd["action_net.bias"].detach().float()
        self.input_dim = int(self.w1.shape[1])
        self.action_out_dim = int(self.wa.shape[0])

        from drl.edge_env import ACTION_DIMS  # noqa: PLC0415
        self.action_dims = [int(x) for x in np.asarray(ACTION_DIMS, dtype=np.int64).tolist()]
        if sum(self.action_dims) != self.action_out_dim:
            raise ValueError(
                f"Legacy action head mismatch: sum(ACTION_DIMS)={sum(self.action_dims)} "
                f"!= action_out_dim={self.action_out_dim}"
            )

    def _forward_logits(self, obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            x = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
            h1 = torch.relu(torch.nn.functional.linear(x, self.w1, self.b1))
            h2 = torch.relu(torch.nn.functional.linear(h1, self.w2, self.b2))
            logits = torch.nn.functional.linear(h2, self.wa, self.ba).detach().cpu().numpy().reshape(-1)
        return logits.astype(np.float64, copy=False)

    def _predict_stormsafe(
        self,
        ds: MILPDataset,
        state_44: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, tuple[str, str]]]:
        obs = np.asarray(state_44[: self.input_dim], dtype=np.float32).reshape(-1)
        flat_logits = self._forward_logits(obs)
        head_logits = split_head_logits(flat_logits, action_dims=self.action_dims)
        flat_mask = compute_action_masks(np.asarray(state_44, dtype=np.float32).reshape(-1))
        masked_logits = apply_flat_mask_to_head_logits(
            head_logits=head_logits,
            flat_mask=flat_mask,
            action_dims=self.action_dims,
        )
        outcome = select_stormsafe_action(
            dataset=ds,
            head_logits=masked_logits,
            decode_action_fn=_decode_legacy_action,
            objective_fn=evaluate_objective_for_placement,
            action_dims=self.action_dims,
            topk_per_head=LEGACY_TOPK_PER_HEAD,
        )
        self.last_decision_meta = {
            "decoder_mode": outcome.decoder_mode,
            "storm_safe": True,
            "migration_count": int(outcome.migration_count),
            "storm_max": int(outcome.storm_max),
            "fallback_full_enumeration": bool(outcome.fallback_full_enumeration),
            "candidates_evaluated": int(outcome.candidates_evaluated),
            "feasible": bool(outcome.feasible),
            "objective_j_selected": float(outcome.objective_j),
        }
        return outcome.action, outcome.placement

    def measure_inference_time_ms(self, ds: MILPDataset, state_44: np.ndarray) -> float:
        if self._last_inference_core_ms is not None:
            return float(self._last_inference_core_ms)
        start = time.perf_counter()
        self._predict_stormsafe(ds, state_44)
        self._last_inference_core_ms = _elapsed_ms(start)
        return float(self._last_inference_core_ms)

    def predict_placement(self, ds: MILPDataset, state_44: np.ndarray) -> dict[str, tuple[str, str]]:
        t0 = time.perf_counter()
        _, placement = self._predict_stormsafe(ds, state_44)
        self._last_inference_core_ms = _elapsed_ms(t0)
        return placement


def _load_model_adapter(model_path: Path, model_name: str) -> ModelAdapter:
    # Backward-compat shim for older SB3 artifacts pickled against numpy internals.
    try:
        import numpy.core as _np_core  # noqa: PLC0415
        import numpy.core.numeric as _np_numeric  # noqa: PLC0415
        sys.modules.setdefault("numpy._core", _np_core)
        sys.modules.setdefault("numpy._core.numeric", _np_numeric)
        # Handle legacy cloudpickle payloads that serialize bit generators as
        # "<class 'numpy.random._pcg64.PCG64'>".
        import numpy.random._pickle as _np_pickle  # noqa: PLC0415
        _orig_ctor = _np_pickle.__bit_generator_ctor

        def _compat_ctor(bit_generator_name="MT19937"):
            name = bit_generator_name
            if not isinstance(name, str) and hasattr(name, "__name__"):
                name = getattr(name, "__name__")
            if isinstance(name, str) and name.startswith("<class '") and name.endswith("'>"):
                name = name[len("<class '"):-2]
                name = name.split(".")[-1]
            return _orig_ctor(name)

        _np_pickle.__bit_generator_ctor = _compat_ctor
    except Exception:
        pass

    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        return DynamicBCAdapter(model_name, model_path)

    def _safe_load(loader_cls, path: Path):
        try:
            return loader_cls.load(str(path))
        except Exception:
            custom_objects = {
                "np_random": None,
                "_last_obs": None,
                "_last_episode_starts": None,
                "_last_original_obs": None,
                "_vec_normalize_env": None,
            }
            return loader_cls.load(str(path), custom_objects=custom_objects)

    # Try standard PPO first, then MaskablePPO.
    model = None
    try:
        from stable_baselines3 import PPO  # noqa: PLC0415
        model = _safe_load(PPO, model_path)
    except Exception:
        try:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415
            model = _safe_load(MaskablePPO, model_path)
        except Exception:
            if suffix == ".zip":
                return LegacyTorchZipAdapter(model_name, model_path)
            raise

    obs_dim = int(model.observation_space.shape[0])
    action_space_name = model.action_space.__class__.__name__
    if obs_dim == 44 and action_space_name == "MultiDiscrete":
        return LegacySB3Adapter(model_name, model_path, model)
    if action_space_name == "Discrete":
        return DynamicSB3Adapter(model_name, model_path, model)

    raise ValueError(
        f"Unsupported model/action-space combination for {model_name}: "
        f"obs_dim={obs_dim}, action_space={action_space_name}"
    )


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline comparison v2 with multi-model adapters (legacy + dynamic)."
    )
    parser.add_argument("--cycles", type=int, default=300)
    parser.add_argument(
        "--scenario", default=None,
        choices=[s.value for s in ScenarioType] + ["mixed"],
        help="Fixed scenario or mixed random sampling (default: mixed)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--out-prefix", default="off_comparison_v2")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        choices=ALL_MODES,
        help="Modes to evaluate. If omitted, resolved from include-* flags (or defaults to milp drl random).",
    )
    parser.add_argument(
        "--models", nargs="+", default=[],
        help="List of model paths (.zip or .pt). Supports multiple models.",
    )
    parser.add_argument(
        "--model-names", nargs="+", default=None,
        help="Optional names for --models (same length).",
    )
    parser.add_argument(
        "--strict-model-load", action="store_true",
        help="Fail fast if any model cannot be loaded. Default: skip failed models and continue.",
    )
    parser.add_argument("--include-milp", action="store_true", help="Write MILP baseline JSONL.")
    parser.add_argument("--include-random", action="store_true", help="Write random baseline JSONL.")
    parser.add_argument(
        "--twin-rollouts", type=int, default=10,
        help="Digital Twin MC rollouts per hybrid cycle (default: 10; 0 disables Twin gate).",
    )
    parser.add_argument(
        "--from-states",
        default=None,
        help="Replay pre-captured 44-dim states from JSONL (state field).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.modes is not None:
        selected_modes = list(dict.fromkeys(args.modes))
    else:
        selected_modes: list[str] = []
        if args.include_milp:
            selected_modes.append("milp")
        if args.include_random:
            selected_modes.append("random")
        if args.models:
            selected_modes.append("drl")
            selected_modes.append("hybrid")
        if not selected_modes:
            selected_modes = ["milp", "drl", "hybrid", "random"]

    results_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    model_paths = [Path(m) for m in args.models]
    model_names = args.model_names or [p.stem for p in model_paths]
    if len(model_names) != len(model_paths):
        raise SystemExit("--model-names length must match --models.")

    adapters: list[ModelAdapter] = []
    for path, name in zip(model_paths, model_names):
        if not path.exists():
            raise SystemExit(f"Model not found: {path}")
        clean_name = _sanitize_name(name)
        log.info("Loading model adapter: %s <- %s", clean_name, path)
        try:
            adapters.append(_load_model_adapter(path, clean_name))
        except Exception as exc:
            msg = f"Skip model {clean_name} ({path}): load failed: {exc}"
            if args.strict_model_load:
                raise SystemExit(msg) from exc
            log.warning(msg)

    out_handles: dict[str, Any] = {}
    if "milp" in selected_modes:
        milp_path = results_dir / f"{args.out_prefix}_milp.jsonl"
        out_handles["milp"] = open(milp_path, "w", encoding="utf-8")
        log.info("Output: %s", milp_path)
    if "random" in selected_modes:
        rnd_path = results_dir / f"{args.out_prefix}_random.jsonl"
        out_handles["random"] = open(rnd_path, "w", encoding="utf-8")
        log.info("Output: %s", rnd_path)
    if "roundrobin" in selected_modes:
        rr_path = results_dir / f"{args.out_prefix}_roundrobin.jsonl"
        out_handles["roundrobin"] = open(rr_path, "w", encoding="utf-8")
        log.info("Output: %s", rr_path)
    for k8s_mode in ("k8s_default_light", "k8s_default_quality"):
        if k8s_mode in selected_modes:
            k8s_path = results_dir / f"{args.out_prefix}_{k8s_mode}.jsonl"
            out_handles[k8s_mode] = open(k8s_path, "w", encoding="utf-8")
            log.info("Output: %s", k8s_path)

    if "drl" in selected_modes or "hybrid" in selected_modes:
        for adapter in adapters:
            if "drl" in selected_modes:
                mp = results_dir / f"{args.out_prefix}_drl_{adapter.model_name}.jsonl"
                out_handles[f"drl:{adapter.model_name}"] = open(mp, "w", encoding="utf-8")
                log.info("Output: %s", mp)
            if "hybrid" in selected_modes:
                hp = results_dir / f"{args.out_prefix}_hybrid_{adapter.model_name}.jsonl"
                out_handles[f"hybrid:{adapter.model_name}"] = open(hp, "w", encoding="utf-8")
                log.info("Output: %s", hp)

    if ("drl" in selected_modes or "hybrid" in selected_modes) and not adapters:
        raise SystemExit("Mode 'drl'/'hybrid' was selected but no model was loaded. Provide valid --models.")
    if not out_handles:
        raise SystemExit("No runnable mode selected.")

    # ── Digital Twin (for hybrid mode) ───────────────────────────────────────
    twin: Optional[DigitalTwinValidator] = None
    if "hybrid" in selected_modes and args.twin_rollouts > 0:
        twin = DigitalTwinValidator(n_rollouts=args.twin_rollouts, seed=args.seed)
        log.info("DigitalTwinValidator ready (n_rollouts=%d)", args.twin_rollouts)
    elif "hybrid" in selected_modes:
        log.info("Twin gate disabled (--twin-rollouts=0); hybrid will use direct J comparison only.")

    _live_states: list[dict] = []
    if args.from_states:
        from_path = Path(args.from_states)
        if not from_path.exists():
            raise SystemExit(f"--from-states file not found: {from_path}")
        with open(from_path, encoding="utf-8") as fh:
            _live_states = [json.loads(line) for line in fh if line.strip()]
        args.cycles = len(_live_states)
        log.info("Loaded %d pre-captured states from %s", len(_live_states), from_path)

    gen = ScenarioGenerator(seed=args.seed)
    fixed_scenario: Optional[ScenarioType] = None
    if args.scenario and args.scenario != "mixed":
        fixed_scenario = ScenarioType(args.scenario)

    _scenario_types = gen._scenario_types
    _scenario_weights = np.asarray(gen._weights, dtype=np.float64)
    _scenario_weights /= _scenario_weights.sum()
    _scenario_rng = np.random.default_rng(args.seed + 1)
    rng = _random.Random(args.seed)

    t_start = time.perf_counter()
    try:
        for cycle in range(1, args.cycles + 1):
            if _live_states:
                rec = _live_states[cycle - 1]
                state = np.asarray(rec["state"], dtype=np.float32)
                scenario_type = rec.get("scenario", "idle_cluster")
            elif fixed_scenario:
                sampled_scenario = fixed_scenario
                state = gen.sample_state(scenario=sampled_scenario)
                scenario_type = sampled_scenario.value
            else:
                idx = int(_scenario_rng.choice(len(_scenario_types), p=_scenario_weights))
                sampled_scenario = _scenario_types[idx]
                state = gen.sample_state(scenario=sampled_scenario)
                scenario_type = sampled_scenario.value

            ds = _build_milp_dataset_from_state(state)
            milp_result = None
            milp_inference_time_ms = None
            milp_start = time.perf_counter()
            try:
                milp_result = solve_placement(ds)
            except Exception as exc:
                log.warning("MILP solve failed cycle=%d: %s", cycle, exc)
            finally:
                milp_inference_time_ms = _elapsed_ms(milp_start)
            milp_decision_time_ms = milp_inference_time_ms
            if milp_result is not None and getattr(milp_result, "solve_time", None) is not None:
                milp_inference_time_ms = float(milp_result.solve_time) * 1000.0

            if "milp" in selected_modes:
                milp_mig = None
                milp_safe = None
                if milp_result is not None:
                    milp_mig = int(sum(1 for v in milp_result.migration_types.values() if v != "Stayed"))
                    milp_safe = bool(milp_mig <= int(ds.v_storm_max))
                row = _milp_result_to_row(
                    milp_result,
                    state,
                    cycle,
                    scenario_type,
                    inference_time_ms=milp_inference_time_ms,
                    decision_time_ms=milp_decision_time_ms,
                    timing_scope="solver_core_ms",
                    decoder_mode="milp_solver_exact",
                    storm_safe=milp_safe,
                    migration_count=milp_mig,
                    storm_max=int(ds.v_storm_max),
                )
                out_handles["milp"].write(json.dumps(row, ensure_ascii=True) + "\n")

            if "random" in selected_modes:
                random_start = time.perf_counter()
                rnd = _random_placement(rng)
                random_inference_time_ms = _elapsed_ms(random_start)
                row = _placement_to_row(
                    rnd,
                    ds,
                    state,
                    cycle,
                    scenario_type,
                    mode="random",
                    inference_time_ms=random_inference_time_ms,
                    decision_time_ms=random_inference_time_ms,
                    timing_scope="baseline_function_ms",
                    decoder_mode="random_baseline",
                    storm_safe=_migration_count_from_x_prev(ds, rnd) <= int(ds.v_storm_max),
                    storm_max=int(ds.v_storm_max),
                )
                out_handles["random"].write(json.dumps(row, ensure_ascii=True) + "\n")

            if "roundrobin" in selected_modes:
                roundrobin_start = time.perf_counter()
                rr = _roundrobin_placement(cycle)
                roundrobin_inference_time_ms = _elapsed_ms(roundrobin_start)
                row = _placement_to_row(
                    rr,
                    ds,
                    state,
                    cycle,
                    scenario_type,
                    mode="roundrobin",
                    inference_time_ms=roundrobin_inference_time_ms,
                    decision_time_ms=roundrobin_inference_time_ms,
                    timing_scope="baseline_function_ms",
                    decoder_mode="roundrobin_baseline",
                    storm_safe=_migration_count_from_x_prev(ds, rr) <= int(ds.v_storm_max),
                    storm_max=int(ds.v_storm_max),
                )
                out_handles["roundrobin"].write(json.dumps(row, ensure_ascii=True) + "\n")

            for k8s_mode in ("k8s_default_light", "k8s_default_quality"):
                if k8s_mode not in selected_modes:
                    continue
                placement = _k8s_default_schedule(ds, _K8S_DEFAULT_VARIANT_PROFILES[k8s_mode])
                row = _placement_to_row(
                    placement,
                    ds,
                    state,
                    cycle,
                    scenario_type,
                    mode=k8s_mode,
                    inference_time_ms=_K8S_DECISION_LATENCY_MS,
                    decision_time_ms=_K8S_DECISION_LATENCY_MS,
                    timing_scope="fixed_scheduler_assumption_ms",
                    decoder_mode=f"{k8s_mode}_heuristic",
                    storm_safe=_migration_count_from_x_prev(ds, placement) <= int(ds.v_storm_max),
                    storm_max=int(ds.v_storm_max),
                )
                out_handles[k8s_mode].write(json.dumps(row, ensure_ascii=True) + "\n")

            if "drl" in selected_modes:
                for adapter in adapters:
                    key = f"drl:{adapter.model_name}"
                    adapter.last_decision_meta = {}
                    adapter._last_inference_core_ms = None
                    drl_inference_time_ms = None
                    drl_start = time.perf_counter()
                    try:
                        placement = adapter.predict_placement(ds, state)
                        drl_decision_time_ms = _elapsed_ms(drl_start)
                        drl_inference_time_ms = adapter.measure_inference_time_ms(ds, state)
                        meta = dict(adapter.last_decision_meta)
                        row = _placement_to_row(
                            placement,
                            ds,
                            state,
                            cycle,
                            scenario_type,
                            mode="drl",
                            model_name=adapter.model_name,
                            inference_time_ms=drl_inference_time_ms,
                            decision_time_ms=drl_decision_time_ms,
                            timing_scope="policy_forward_decode_ms",
                            decoder_mode=meta.get("decoder_mode"),
                            storm_safe=meta.get("storm_safe"),
                            migration_count=meta.get("migration_count"),
                            storm_max=meta.get("storm_max", int(ds.v_storm_max)),
                        )
                    except Exception as exc:
                        if drl_inference_time_ms is None:
                            drl_inference_time_ms = _elapsed_ms(drl_start)
                        drl_decision_time_ms = drl_inference_time_ms
                        log.warning("DRL adapter failed cycle=%d model=%s: %s", cycle, adapter.model_name, exc)
                        e2e = _e2e_from_state(state)
                        row = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "cycle": cycle,
                            "mode": "drl",
                            "model_name": adapter.model_name,
                            "scenario": scenario_type,
                            "inference_time_ms": _round_ms(drl_inference_time_ms),
                            "objective_J": None,
                            "e2e_latency_ms": e2e,
                            "energy_w": None,
                            "migrations": None,
                            "sla_violated": e2e > SLA_LATENCY_MS,
                            "infeasible": True,
                        }
                        _stamp_timing(
                            row,
                            decision_time_ms=drl_decision_time_ms,
                            timing_scope="policy_forward_decode_ms",
                        )
                        _stamp_decoder(
                            row,
                            decoder_mode=(adapter.last_decision_meta or {}).get("decoder_mode"),
                            storm_safe=(adapter.last_decision_meta or {}).get("storm_safe"),
                            migration_count=(adapter.last_decision_meta or {}).get("migration_count"),
                            storm_max=int(getattr(ds, "v_storm_max", 0)),
                        )
                    out_handles[key].write(json.dumps(row, ensure_ascii=True) + "\n")

                    # ── Hybrid: Twin gate → min(J_DRL, J_MILP) ───────────────
                    # Flow:
                    #  1. If Twin available AND adapter exposes action array:
                    #     run Twin safety gate (feasible_rate, std_reward).
                    #     If DRL is unsafe → MILP fallback (skip J compare).
                    #  2. If DRL passes safety gate (or no Twin / no action):
                    #     compare actual J_DRL vs J_MILP, pick lower.
                    if "hybrid" in selected_modes:
                        hkey = f"hybrid:{adapter.model_name}"
                        milp_j = (
                            float(milp_result.objective_value)
                            if milp_result is not None
                            else None
                        )
                        drl_j_val = row.get("objective_J")

                        # Try to get raw action for Twin validation
                        twin_result = None
                        twin_unsafe = False
                        twin_high_variance = False
                        if twin is not None and drl_j_val is not None:
                            try:
                                drl_action, _ = adapter.predict_action_and_placement(ds, state)
                                if drl_action is not None:
                                    twin_result = twin.validate(drl_action, state)
                                    twin_unsafe = not twin_result.is_safe
                                    twin_high_variance = twin_result.std_reward > 0.25
                                    if twin_unsafe or twin_high_variance:
                                        reason = "twin_unsafe" if twin_unsafe else "twin_high_variance"
                                        log.debug(
                                            "Hybrid cycle=%d model=%s: %s "
                                            "(feasible=%.0f%% std=%.3f) → MILP",
                                            cycle, adapter.model_name, reason,
                                            twin_result.feasible_rate * 100,
                                            twin_result.std_reward,
                                        )
                                # Dynamic adapters return action=None → skip Twin gate
                            except Exception as twin_exc:
                                log.debug(
                                    "Hybrid twin validate failed cycle=%d model=%s: %s",
                                    cycle, adapter.model_name, twin_exc,
                                )

                        # Build twin fields for output row
                        twin_fields: dict[str, Any] = {}
                        if twin_result is not None:
                            twin_fields = {
                                "twin_is_safe": twin_result.is_safe,
                                "twin_feasible_rate": round(twin_result.feasible_rate, 3),
                                "twin_std_reward": round(twin_result.std_reward, 4),
                                "twin_predicted_j": round(twin_result.predicted_j, 6),
                            }

                        # Decide source: if Twin rejected → MILP wins by safety;
                        # else compare J directly
                        if twin_unsafe or twin_high_variance:
                            force_milp = True
                            milp_decoder_mode = "milp_fallback_twin_rejected"
                        else:
                            force_milp = False
                            milp_decoder_mode = "milp_wins_j_compare"

                        if drl_j_val is not None and milp_j is not None and not force_milp:
                            # Both available + DRL passed gate: pick lower J
                            if milp_j < drl_j_val:
                                milp_mig = int(sum(
                                    1 for v in milp_result.migration_types.values()
                                    if v != "Stayed"
                                ))
                                hybrid_row = _milp_result_to_row(
                                    milp_result,
                                    state,
                                    cycle,
                                    scenario_type,
                                    inference_time_ms=milp_inference_time_ms,
                                    decision_time_ms=milp_decision_time_ms,
                                    timing_scope="hybrid_milp_wins_ms",
                                    decoder_mode=milp_decoder_mode,
                                    storm_safe=bool(milp_mig <= int(ds.v_storm_max)),
                                    migration_count=milp_mig,
                                    storm_max=int(ds.v_storm_max),
                                )
                                hybrid_row["mode"] = "hybrid"
                                hybrid_row["model_name"] = adapter.model_name
                                hybrid_row["hybrid_source"] = "milp"
                                hybrid_row.update(twin_fields)
                            else:
                                hybrid_row = dict(row)
                                hybrid_row["mode"] = "hybrid"
                                hybrid_row["decoder_mode"] = (
                                    str(row.get("decoder_mode", "")) + "_drl_wins_j_compare"
                                ).lstrip("_")
                                hybrid_row["hybrid_source"] = "drl"
                                hybrid_row.update(twin_fields)
                            log.debug(
                                "Hybrid cycle=%d model=%s: source=%s "
                                "(drl_J=%.4f milp_J=%.4f twin=%s)",
                                cycle, adapter.model_name,
                                hybrid_row["hybrid_source"], drl_j_val, milp_j,
                                "safe" if (twin_result and twin_result.is_safe) else
                                "unsafe" if twin_result else "no_twin",
                            )
                        elif (milp_j is not None) and (force_milp or drl_j_val is None):
                            # Twin rejected DRL or DRL failed → use MILP
                            milp_mig = int(sum(
                                1 for v in milp_result.migration_types.values()
                                if v != "Stayed"
                            ))
                            hybrid_row = _milp_result_to_row(
                                milp_result, state, cycle, scenario_type,
                                inference_time_ms=milp_inference_time_ms,
                                decision_time_ms=milp_decision_time_ms,
                                timing_scope="hybrid_drl_failed_ms",
                                decoder_mode=milp_decoder_mode if force_milp
                                    else "milp_fallback_drl_failed",
                                storm_safe=bool(milp_mig <= int(ds.v_storm_max)),
                                migration_count=milp_mig,
                                storm_max=int(ds.v_storm_max),
                            )
                            hybrid_row["mode"] = "hybrid"
                            hybrid_row["model_name"] = adapter.model_name
                            hybrid_row["hybrid_source"] = "milp"
                            hybrid_row.update(twin_fields)
                        elif drl_j_val is not None:
                            # MILP failed — use DRL
                            hybrid_row = dict(row)
                            hybrid_row["mode"] = "hybrid"
                            hybrid_row["hybrid_source"] = "drl"
                            hybrid_row.update(twin_fields)
                        else:
                            # Both failed
                            e2e = _e2e_from_state(state)
                            hybrid_row = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "cycle": cycle,
                                "mode": "hybrid",
                                "model_name": adapter.model_name,
                                "scenario": scenario_type,
                                "objective_J": None,
                                "e2e_latency_ms": e2e,
                                "energy_w": None,
                                "migrations": None,
                                "sla_violated": e2e > SLA_LATENCY_MS,
                                "infeasible": True,
                                "hybrid_source": None,
                            }
                        out_handles[hkey].write(json.dumps(hybrid_row, ensure_ascii=True) + "\n")

            if cycle % 25 == 0 or cycle == args.cycles:
                elapsed = time.perf_counter() - t_start
                log.info("cycle %d/%d elapsed=%.1fs scenario=%s", cycle, args.cycles, elapsed, scenario_type)

        elapsed = time.perf_counter() - t_start
        log.info(
            "Done. %d cycles, %d models in %.1f s (%.1f ms/cycle)",
            args.cycles,
            len(adapters),
            elapsed,
            elapsed / max(args.cycles, 1) * 1000,
        )
        log.info("Results written to: %s", results_dir)

    finally:
        for h in out_handles.values():
            h.close()


if __name__ == "__main__":
    main()
