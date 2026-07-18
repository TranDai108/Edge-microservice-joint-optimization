"""DRL inference agent — FastAPI service, 5 s control loop.

Phase 3 (D3.1):
  - FastAPI service on port 8001.
  - GET /health  → {"status": "ok", "mode": "<current_mode>"}
  - Background thread: _inference_loop() runs every 5 s.
      * Reads system:mode from Redis.
      * In "shadow" / "drl" / "hybrid" modes: runs PPO inference.
      * In "drl" / "hybrid": may commit drl:placement (TTL 35 s).
      * In "shadow": publish proposal only (observe-only, no commit).
      * In "drl" mode: also pushes SARSA tuple to drl:training_buffer
        (capped at 1000 entries — D3.6).
      * In "milp" mode: sleeps, no inference.

HA additions (Controller HA and Failover Plan):
  - Leader election via K8s Lease (only the elected leader writes
    drl:placement and drl:placement:proposed; standby still runs inference but
    discards placement outputs).
  - Epoch-fenced placement/proposal writes via placement_write_guarded()
    prevent split-brain writes from a stale pod that has not yet detected
    leader loss.
  - Sentinel-aware Redis client from src.ha.redis_client.

Usage (local):
    uvicorn src.drl.drl_agent:app --host 0.0.0.0 --port 8001
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import logging
import math
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import redis as redis_lib
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from src.drl.digital_twin import DigitalTwinValidator
except ImportError:
    from drl.digital_twin import DigitalTwinValidator

try:
    from src.ha.leader_election import placement_write_guarded
except ImportError:
    from ha.leader_election import placement_write_guarded

try:
    from src.drl.legacy44_stormsafe_decoder import (
        DEFAULT_ACTION_DIMS as LEGACY_ACTION_DIMS,
        DEFAULT_TOPK_PER_HEAD as LEGACY_TOPK_PER_HEAD,
        apply_flat_mask_to_head_logits,
        decode_legacy44_action,
        select_stormsafe_action,
    )
except ImportError:
    from drl.legacy44_stormsafe_decoder import (
        DEFAULT_ACTION_DIMS as LEGACY_ACTION_DIMS,
        DEFAULT_TOPK_PER_HEAD as LEGACY_TOPK_PER_HEAD,
        apply_flat_mask_to_head_logits,
        decode_legacy44_action,
        select_stormsafe_action,
    )

# ── Resolve model path relative to this file ─────────────────────────────────
_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = str(_HERE / "models" / "ppo_simulator.zip")
_DEFAULT_REDIS_HOST = os.getenv("REDIS_HOST", "redis.default.svc.cluster.local")
_DEFAULT_REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
_INFERENCE_INTERVAL_S = max(
    0.5,
    float(os.getenv("DRL_INFERENCE_INTERVAL_S", "5")),
)  # seconds between inference cycles
_TWIN_N_ROLLOUTS = int(os.getenv("DRL_TWIN_N_ROLLOUTS", "20"))
_TWIN_SAFETY_THRESHOLD = float(os.getenv("DRL_TWIN_SAFETY_THRESHOLD", "-0.5"))
_TWIN_MIN_FEASIBLE_RATE = float(os.getenv("DRL_TWIN_MIN_FEASIBLE_RATE", "0.8"))
_EPSILON_FALLBACK = float(os.getenv("DRL_EPSILON_FALLBACK", "0.22"))
_EPSILON_WINDOW = int(os.getenv("DRL_EPSILON_WINDOW", "50"))
_REJECT_RETRY_AFTER = int(os.getenv("DRL_REJECT_RETRY_AFTER", "3"))
_REJECT_CANDIDATES = int(os.getenv("DRL_REJECT_CANDIDATES", "8"))
_ACTION_STICKY_WINDOW = int(os.getenv("DRL_ACTION_STICKY_WINDOW", "12"))
_ACTION_STICKY_MIN_IMPROVEMENT = float(os.getenv("DRL_ACTION_STICKY_MIN_IMPROVEMENT", "0.005"))
_ACTION_STICKY_GUARD_ENABLED = os.getenv("DRL_ACTION_STICKY_GUARD_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
_CANDIDATE_RERANK_ENABLED = os.getenv("DRL_CANDIDATE_RERANK_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
_CANDIDATE_RERANK_MODES = {
    x.strip().lower()
    for x in os.getenv("DRL_CANDIDATE_RERANK_MODES", "hybrid").split(",")
    if x.strip()
}
_RERANK_MIN_IMPROVEMENT = float(os.getenv("DRL_RERANK_MIN_IMPROVEMENT", "0.001"))
_RERANK_INCLUDE_MILP = os.getenv("DRL_RERANK_INCLUDE_MILP", "true").lower() in {
    "1", "true", "yes", "on"
}
_RERANK_INCLUDE_CONFIRMED = os.getenv("DRL_RERANK_INCLUDE_CONFIRMED", "true").lower() in {
    "1", "true", "yes", "on"
}
_SKIP_LEADER_ELECTION = bool(os.getenv("DRL_SKIP_LEADER_ELECTION", ""))
_STORM_PROJECTION_BUDGET = int(os.getenv("DRL_STORM_PROJECTION_BUDGET", "2"))
_STORM_CORRECTION_BUFFER_KEY = os.getenv("DRL_STORM_CORRECTION_BUFFER_KEY", "drl:storm_corrections")
_STORM_CORRECTION_MAXLEN = int(os.getenv("DRL_STORM_CORRECTION_MAXLEN", "5000"))
_DRL_DYNAMIC_DECODE_MODE = os.getenv("DRL_DYNAMIC_DECODE_MODE", "greedy").strip().lower()
_DRL_DYNAMIC_BEAM_WIDTH = max(1, int(os.getenv("DRL_DYNAMIC_BEAM_WIDTH", "24")))
_DRL_DYNAMIC_TOPK_ACTIONS = max(1, int(os.getenv("DRL_DYNAMIC_TOPK_ACTIONS", "8")))
_DRL_TORCH_NUM_THREADS = max(1, int(os.getenv("DRL_TORCH_NUM_THREADS", "1")))
_DRL_ALLOWED_CONTRACTS = {
    x.strip() for x in os.getenv("DRL_ALLOWED_CONTRACTS", "legacy_44,dynamic_96").split(",")
    if x.strip()
}
_DRL_DISABLE_ON_UNSUPPORTED = os.getenv("DRL_DISABLE_ON_UNSUPPORTED", "true").lower() in {
    "1", "true", "yes", "on"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("drl-agent")

# ── Module-level Redis client (set at startup) ────────────────────────────────
_rdb: Optional[redis_lib.Redis] = None

# ── Module-level leader elector (set at startup, may be None in dev) ──────────
_elector = None

_runtime_state: dict[str, object] = {
    "model_contract": "unknown",
    "adapter_name": "none",
    "inference_enabled": False,
    "inference_error_count": 0,
    "obs_dim": None,
    "action_space": "unknown",
}


def _write_proposed_placement(rdb: redis_lib.Redis, payload: dict) -> bool:
    """Publish the Tier-2 proposal only from the current DRL leader.

    Local/single-pod runs have no elector and retain the direct-write path.
    HA runs use the same epoch-fenced Redis write as committed placement so a
    standby or stale leader cannot overwrite the proposal being verified.
    """
    if _elector is None:
        rdb.setex("drl:placement:proposed", 60, json.dumps(payload))
        return True

    try:
        return placement_write_guarded(
            rdb,
            "drl:placement:proposed",
            payload,
            _elector,
            ttl=60,
        )
    except Exception as exc:
        # Never fall back to a direct HA write: doing so would restore the
        # multi-writer race this guard exists to prevent.
        log.warning("Proposed placement leader guard failed: %s", exc)
        return False


class ModelContract(str, Enum):
    LEGACY_44 = "legacy_44"
    DYNAMIC_96 = "dynamic_96"
    UNSUPPORTED = "unsupported"


@dataclass
class PlacementResult:
    placement: dict[str, dict[str, str]]
    action: np.ndarray
    adapter_name: str
    model_contract: ModelContract
    obs_dim: int
    notes: str = ""
    inference_core_ms: float | None = None
    decode_mode: str = "n/a"
    storm_safe: bool | None = None
    migration_count: int | None = None
    storm_max: int | None = None
    fallback_full_enumeration: bool = False


class BaseInferenceAdapter(ABC):
    name: str = "base"
    contract: ModelContract = ModelContract.UNSUPPORTED

    def __init__(self, model: object, obs_dim: int) -> None:
        self.model = model
        self.obs_dim = obs_dim

    @abstractmethod
    def predict(self, rdb: redis_lib.Redis) -> PlacementResult:
        raise NotImplementedError


# ── Lazy PPO import (avoids slow torch import at module level) ────────────────
def _load_model(model_path: str):
    import torch  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    try:
        torch.set_num_threads(_DRL_TORCH_NUM_THREADS)
    except Exception:
        pass
    try:
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        return MaskablePPO.load(model_path)
    except TypeError as exc:
        if "use_sde" not in str(exc):
            raise
        # sb3-contrib >= 2.4 removed use_sde from MaskableActorCriticPolicy.__init__.
        # Models saved with older versions carry this kwarg in the data dict.
        # Patch the constructor to absorb it, load, then restore.
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # noqa: PLC0415
        _orig_init = MaskableActorCriticPolicy.__init__

        def _compat_init(self, *args, use_sde: bool = False, **kwargs):  # noqa: ANN001
            _orig_init(self, *args, **kwargs)

        MaskableActorCriticPolicy.__init__ = _compat_init
        try:
            return MaskablePPO.load(model_path)
        finally:
            MaskableActorCriticPolicy.__init__ = _orig_init


def _normalize_model_path(model_path: str) -> str:
    """Keep env path canonical while supporting relative paths."""
    p = Path(model_path)
    if p.is_absolute():
        return str(p)
    return str((_HERE / p).resolve()) if not str(p).startswith("models/") else str((_HERE / p).resolve())


def _load_model_with_fallback(model_path: str):
    """Load model from canonical path; optionally try '.zip' if suffix is omitted."""
    normalized = _normalize_model_path(model_path)
    candidates = [normalized]
    if not normalized.endswith(".zip"):
        candidates.append(f"{normalized}.zip")
    last_exc: Exception | None = None
    for cand in candidates:
        try:
            model = _load_model(cand)
            return model, cand
        except Exception as exc:  # pragma: no cover - runtime compatibility branch
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"Unable to load model path: {model_path}")


def _resolve_model_contract(model: object) -> tuple[ModelContract, int, str]:
    """Resolve runtime contract from model spaces."""
    obs_dim = -1
    action_name = "unknown"
    try:
        obs_dim = int(model.observation_space.shape[0])  # type: ignore[attr-defined]
    except Exception:
        obs_dim = -1
    try:
        action_name = model.action_space.__class__.__name__  # type: ignore[attr-defined]
    except Exception:
        action_name = "unknown"
    if obs_dim == 44 and action_name == "MultiDiscrete":
        return ModelContract.LEGACY_44, obs_dim, action_name
    if obs_dim == 96 and action_name == "Discrete":
        return ModelContract.DYNAMIC_96, obs_dim, action_name
    return ModelContract.UNSUPPORTED, obs_dim, action_name


# ── Service/node name→id maps (must match edge_env.py order) ─────────────────
_SVC_NAME_TO_ID = {
    "api-gateway": "m0", "ingest": "m1", "preprocess": "m2",
    "detection": "m3", "gen-ai": "m4", "postprocess": "m5",
}
_HOST_TO_NODE_ID = {
    "edge-nodes-1": "n0", "edge-nodes-2": "n1",
    "edge-nodes-3": "n2", "edge-nodes-4": "n3",
}


def _placement_to_ids(placement: dict) -> dict[str, tuple[str, str]]:
    placement_ids: dict[str, tuple[str, str]] = {}
    for svc_name, info in placement.items():
        svc_id = _SVC_NAME_TO_ID.get(svc_name)
        node_id = _HOST_TO_NODE_ID.get(info.get("node", ""), "n0")
        variant = info.get("variant", "standard")
        if svc_id:
            placement_ids[svc_id] = (variant, node_id)
    return placement_ids


def _placement_matches_payload(
    placement: dict[str, dict[str, str]],
    payload: dict | None,
) -> bool:
    if not payload:
        return False
    other = payload.get("placement")
    if not isinstance(other, dict):
        return False
    if set(other.keys()) != set(placement.keys()):
        return False
    for svc, info in placement.items():
        o = other.get(svc, {})
        if not isinstance(o, dict):
            return False
        if o.get("node") != info.get("node"):
            return False
        if o.get("variant", "standard") != info.get("variant", "standard"):
            return False
    return True


def _compute_objective_breakdown(
    placement: dict,
    state: np.ndarray,
) -> dict[str, float] | None:
    """Compute full objective decomposition for a DRL placement.

    Uses the same reward decomposition used by DRL training/evaluation.
    """
    try:
        try:
            from src.drl.edge_env import EdgeEnv  # noqa: PLC0415
            from src.drl.reward import evaluate_objective_for_placement  # noqa: PLC0415
        except ImportError:
            from drl.edge_env import EdgeEnv  # noqa: PLC0415
            from drl.reward import evaluate_objective_for_placement  # noqa: PLC0415

        env = EdgeEnv(mock=True)
        env._state = state.copy()
        env._step_count = 0
        dataset = env._build_reward_dataset()
        placement_ids = _placement_to_ids(placement)

        if len(placement_ids) != 6:
            return None

        breakdown = evaluate_objective_for_placement(dataset, placement_ids)
        return {
            "objective_reward": round(float(breakdown.objective_j), 6),
            "cost_energy": round(float(breakdown.cost_energy), 6),
            "cost_disruption": round(float(breakdown.cost_disruption), 6),
            "gain_accuracy": round(float(breakdown.gain_accuracy), 6),
            "norm_cost_energy": round(float(breakdown.norm_cost_energy), 6),
            "norm_cost_disruption": round(float(breakdown.norm_cost_disruption), 6),
            "norm_gain_accuracy": round(float(breakdown.norm_gain_accuracy), 6),
        }
    except Exception as exc:
        log.debug("objective breakdown computation skipped: %s", exc)
        return None


# ── State builder — delegates to EdgeEnv to avoid logic duplication ───────────
def _build_state_from_redis(rdb: redis_lib.Redis) -> np.ndarray:
    """Return a 44-dim float32 state vector from live Redis data.

    Delegates to EdgeEnv._build_live_state() which already mirrors
    milp_agent._build_state_vector() exactly.
    """
    try:
        from src.drl.edge_env import EdgeEnv  # noqa: PLC0415
    except ImportError:
        from drl.edge_env import EdgeEnv  # noqa: PLC0415

    env = EdgeEnv(mock=False)
    env.rdb = rdb  # inject the shared client directly
    return env._build_live_state()


# ── Action → placement dict ───────────────────────────────────────────────────
def _action_to_placement(action: np.ndarray) -> dict:
    """Decode MultiDiscrete action to placement dict matching milp:placement schema.

    Returns:
        {
          "api-gateway":  {"node": "edge-nodes-X", "variant": "standard"},
          "ingest":       {"node": "edge-nodes-X", "variant": "standard"},
          "preprocess":   {"node": "edge-nodes-X", "variant": "standard"},
          "detection":    {"node": "edge-nodes-X", "variant": "<det_variant>"},
          "gen-ai":       {"node": "edge-nodes-X", "variant": "<gen_variant>"},
          "postprocess":  {"node": "edge-nodes-X", "variant": "standard"},
        }
    """
    NODE_HOSTNAMES = [
        "edge-nodes-1", "edge-nodes-2", "edge-nodes-3", "edge-nodes-4",
    ]
    DET_VARIANTS = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
    GEN_VARIANTS = ["qwen-1.5b-nano", "llama-3b-small", "gemma2-2b-medium"]

    a = np.asarray(action, dtype=np.int64).tolist()

    def _decode_vn(value: int, variants: list[str]) -> tuple[str, str]:
        n_nodes = len(NODE_HOSTNAMES)
        var_idx = int(value) // n_nodes
        node_idx = int(value) % n_nodes
        var_idx = min(max(var_idx, 0), len(variants) - 1)
        node_idx = min(max(node_idx, 0), n_nodes - 1)
        return variants[var_idx], NODE_HOSTNAMES[node_idx]

    det_var, det_node = _decode_vn(int(a[3]), DET_VARIANTS)
    gen_var, gen_node = _decode_vn(int(a[4]), GEN_VARIANTS)

    return {
        "api-gateway": {"node": NODE_HOSTNAMES[min(int(a[0]), 3)], "variant": "standard"},
        "ingest":      {"node": NODE_HOSTNAMES[min(int(a[1]), 3)], "variant": "standard"},
        "preprocess":  {"node": NODE_HOSTNAMES[min(int(a[2]), 3)], "variant": "standard"},
        "detection":   {"node": det_node, "variant": det_var},
        "gen-ai":      {"node": gen_node, "variant": gen_var},
        "postprocess": {"node": NODE_HOSTNAMES[min(int(a[5]), 3)], "variant": "standard"},
    }


def _decode_legacy_action_ids(action: np.ndarray) -> dict[str, tuple[str, str]]:
    try:
        from src.drl.edge_env import NODE_IDS  # noqa: PLC0415
    except ImportError:
        from drl.edge_env import NODE_IDS  # noqa: PLC0415
    from variant_catalog import DETECTION_VARIANTS, GEN_AI_VARIANTS  # noqa: PLC0415

    return decode_legacy44_action(
        action,
        node_ids=list(NODE_IDS),
        detection_variants=list(DETECTION_VARIANTS),
        gen_ai_variants=list(GEN_AI_VARIANTS),
    )


def _placement_to_action(placement: dict | None) -> np.ndarray | None:
    """Encode confirmed placement dict back to MultiDiscrete action."""
    if not placement:
        return None
    node_to_idx = {
        "edge-nodes-1": 0, "edge-nodes-2": 1, "edge-nodes-3": 2, "edge-nodes-4": 3,
    }
    det_variants = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
    gen_variants = ["qwen-1.5b-nano", "llama-3b-small", "gemma2-2b-medium"]
    try:
        m0 = node_to_idx[placement["api-gateway"]["node"]]
        m1 = node_to_idx[placement["ingest"]["node"]]
        m2 = node_to_idx[placement["preprocess"]["node"]]
        det_node = node_to_idx[placement["detection"]["node"]]
        det_var = det_variants.index(placement["detection"].get("variant", det_variants[0]))
        gen_node = node_to_idx[placement["gen-ai"]["node"]]
        gen_var = gen_variants.index(placement["gen-ai"].get("variant", gen_variants[0]))
        m5 = node_to_idx[placement["postprocess"]["node"]]
    except Exception:
        return None
    return np.asarray(
        [m0, m1, m2, det_var * 4 + det_node, gen_var * 4 + gen_node, m5],
        dtype=np.int64,
    )


def _placement_complete(placement: dict[str, dict[str, str]]) -> bool:
    required = {"api-gateway", "ingest", "preprocess", "detection", "gen-ai", "postprocess"}
    if set(placement.keys()) != required:
        return False
    for info in placement.values():
        if not isinstance(info, dict):
            return False
        if not info.get("node") or not info.get("variant"):
            return False
    return True


@dataclass
class CandidateRerankOutcome:
    action: np.ndarray
    placement: dict[str, dict[str, str]]
    twin_result: Any
    applied: bool
    reason: str
    selected_source: str
    candidate_count: int


def _normalize_action(action: np.ndarray | list[int] | tuple[int, ...] | None) -> np.ndarray | None:
    if action is None:
        return None
    try:
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] != 6:
        return None
    return arr


def _append_unique_candidate(
    candidates: list[tuple[str, np.ndarray]],
    seen: set[tuple[int, ...]],
    source: str,
    action: np.ndarray | list[int] | tuple[int, ...] | None,
) -> None:
    arr = _normalize_action(action)
    if arr is None:
        return
    key = tuple(int(x) for x in arr.tolist())
    if key in seen:
        return
    seen.add(key)
    candidates.append((source, arr))


def _build_rerank_candidates(
    policy_action: np.ndarray | list[int] | tuple[int, ...],
    milp_payload: dict | None,
    confirmed_placement: dict | None,
    *,
    include_milp: bool = True,
    include_confirmed: bool = True,
) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    _append_unique_candidate(candidates, seen, "policy", policy_action)

    if include_milp and isinstance(milp_payload, dict):
        _append_unique_candidate(
            candidates,
            seen,
            "milp",
            _placement_to_action(milp_payload.get("placement")),
        )
    if include_confirmed:
        _append_unique_candidate(
            candidates,
            seen,
            "confirmed",
            _placement_to_action(confirmed_placement),
        )
    return candidates


def _source_for_action(candidates: list[tuple[str, np.ndarray]], action: np.ndarray) -> str:
    arr = _normalize_action(action)
    if arr is None:
        return "unknown"
    for source, candidate in candidates:
        if np.array_equal(candidate, arr):
            return source
    return "unknown"


def _candidate_rerank_mode_settings(mode: str) -> tuple[bool, bool]:
    mode = str(mode).lower()
    enabled = _CANDIDATE_RERANK_ENABLED and mode in _CANDIDATE_RERANK_MODES
    include_milp = enabled and _RERANK_INCLUDE_MILP and mode == "hybrid"
    return enabled, include_milp


def _rerank_action_candidates(
    *,
    validator: DigitalTwinValidator,
    state: np.ndarray,
    confirmed_placement: dict | None,
    current_action: np.ndarray,
    current_placement: dict[str, dict[str, str]],
    current_twin_result: Any,
    milp_payload: dict | None,
    enabled: bool,
    min_improvement: float,
    include_milp: bool,
    include_confirmed: bool,
) -> CandidateRerankOutcome:
    current_action_arr = _normalize_action(current_action)
    if current_action_arr is None:
        current_action_arr = np.asarray(current_action, dtype=np.int64).reshape(-1)

    if not enabled:
        return CandidateRerankOutcome(
            action=current_action_arr,
            placement=current_placement,
            twin_result=current_twin_result,
            applied=False,
            reason="disabled",
            selected_source="policy",
            candidate_count=1,
        )

    candidates = _build_rerank_candidates(
        current_action_arr,
        milp_payload,
        confirmed_placement,
        include_milp=include_milp,
        include_confirmed=include_confirmed,
    )
    if len(candidates) <= 1:
        return CandidateRerankOutcome(
            action=current_action_arr,
            placement=current_placement,
            twin_result=current_twin_result,
            applied=False,
            reason="single_candidate",
            selected_source="policy",
            candidate_count=len(candidates),
        )

    best_action, best_result = validator.find_best_action(
        [candidate for _, candidate in candidates],
        state,
        confirmed_placement=confirmed_placement,
    )
    selected_source = _source_for_action(candidates, best_action)
    changed = not np.array_equal(current_action_arr, np.asarray(best_action, dtype=np.int64))
    improved = (
        bool(getattr(best_result, "is_safe", False))
        and float(getattr(best_result, "objective_j_raw", float("inf")))
        < (float(getattr(current_twin_result, "objective_j_raw", float("inf"))) - float(min_improvement))
    )

    if not changed:
        reason = "policy_best"
    elif not getattr(best_result, "is_safe", False):
        reason = "best_not_safe"
    elif not improved:
        reason = "below_threshold"
    else:
        best_placement = _action_to_placement(np.asarray(best_action, dtype=np.int64))
        if not isinstance(best_placement, dict) or not _placement_complete(best_placement):
            reason = "best_incomplete"
        else:
            return CandidateRerankOutcome(
                action=np.asarray(best_action, dtype=np.int64),
                placement=best_placement,
                twin_result=best_result,
                applied=True,
                reason="selected_better_safe_candidate",
                selected_source=selected_source,
                candidate_count=len(candidates),
            )

    return CandidateRerankOutcome(
        action=current_action_arr,
        placement=current_placement,
        twin_result=current_twin_result,
        applied=False,
        reason=reason,
        selected_source=selected_source,
        candidate_count=len(candidates),
    )


def _count_migrations_from_confirmed(
    proposed: dict[str, dict[str, str]],
    confirmed: dict | None,
) -> int:
    if not confirmed:
        return 0
    count = 0
    for svc, p in proposed.items():
        c = confirmed.get(svc, {})
        if not c:
            count += 1
            continue
        if c.get("node") != p.get("node") or c.get("variant", "standard") != p.get("variant", "standard"):
            count += 1
    return count


def _project_placement_to_storm_budget(
    proposed: dict[str, dict[str, str]],
    confirmed: dict | None,
    budget: int,
) -> tuple[dict[str, dict[str, str]], bool, int, int]:
    """Project placement to at most `budget` migrations vs confirmed placement.

    Priority keeps AI services first when trimming:
      detection > gen-ai > preprocess > ingest > api-gateway > postprocess
    """
    if not confirmed or budget < 0:
        mc = _count_migrations_from_confirmed(proposed, confirmed)
        return proposed, False, mc, mc
    priority = {
        "detection": 0,
        "gen-ai": 1,
        "preprocess": 2,
        "ingest": 3,
        "api-gateway": 4,
        "postprocess": 5,
    }
    proposed_copy = {k: dict(v) for k, v in proposed.items()}
    changed: list[str] = []
    for svc, p in proposed_copy.items():
        c = confirmed.get(svc, {})
        if not c:
            changed.append(svc)
            continue
        if c.get("node") != p.get("node") or c.get("variant", "standard") != p.get("variant", "standard"):
            changed.append(svc)
    before = len(changed)
    if before <= budget:
        return proposed_copy, False, before, before
    changed.sort(key=lambda s: priority.get(s, 99))
    keep = set(changed[:budget])
    for svc in changed:
        if svc in keep:
            continue
        c = confirmed.get(svc, {})
        if c and c.get("node"):
            proposed_copy[svc] = {
                "node": c.get("node"),
                "variant": c.get("variant", "standard"),
            }
    after = _count_migrations_from_confirmed(proposed_copy, confirmed)
    return proposed_copy, True, before, after


def _push_storm_correction_sample(
    rdb: redis_lib.Redis,
    *,
    mode: str,
    state: np.ndarray,
    confirmed_placement: dict | None,
    raw_action: np.ndarray,
    projected_action: np.ndarray,
    raw_placement: dict[str, dict[str, str]],
    projected_placement: dict[str, dict[str, str]],
    migration_before: int,
    migration_after: int,
    storm_budget: int,
    model_contract: str,
    adapter_name: str,
) -> None:
    """Persist projection-correction sample for later fine-tuning.

    These samples represent the exact cases where policy proposes
    storm-heavy transitions and runtime has to trim to stay safe.
    """
    try:
        payload = {
            "ts": int(time.time()),
            "mode": mode,
            "model_contract": model_contract,
            "adapter_name": adapter_name,
            "storm_budget": int(storm_budget),
            "migration_before": int(migration_before),
            "migration_after": int(migration_after),
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "confirmed_placement": confirmed_placement or {},
            "raw_action": np.asarray(raw_action, dtype=np.int64).tolist(),
            "projected_action": np.asarray(projected_action, dtype=np.int64).tolist(),
            "raw_placement": raw_placement,
            "projected_placement": projected_placement,
        }
        rdb.lpush(_STORM_CORRECTION_BUFFER_KEY, json.dumps(payload))
        rdb.ltrim(_STORM_CORRECTION_BUFFER_KEY, 0, max(0, _STORM_CORRECTION_MAXLEN - 1))
    except Exception as exc:
        log.debug("Unable to persist storm correction sample: %s", exc)


@dataclass
class _DynamicEpisode:
    storm_max: int
    nodes: list[dict[str, float | str]]
    services: list[dict[str, object]]


def _build_milp_dataset_from_state(state: np.ndarray):
    try:
        from src.drl.edge_env import NODE_IDS, build_mock_dataset  # noqa: PLC0415
    except ImportError:
        from drl.edge_env import NODE_IDS, build_mock_dataset  # noqa: PLC0415

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


def _extract_prev_placement(ds) -> dict[str, tuple[str, str]]:
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


def _build_dynamic_episode(ds) -> _DynamicEpisode:
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
    services: list[dict[str, object]] = []
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
    return _DynamicEpisode(storm_max=int(ds.v_storm_max), nodes=nodes, services=services)


def _action_index(variant_idx: int, node_idx: int, max_nodes: int) -> int:
    return variant_idx * max_nodes + node_idx


def _decode_action_index(action_idx: int, max_nodes: int) -> tuple[int, int]:
    return action_idx // max_nodes, action_idx % max_nodes


def _infer_dynamic_shape(obs_dim: int, action_dim: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for n in range(1, 65):
        if action_dim % n != 0:
            continue
        v = action_dim // n
        if 12 + 9 * n + 4 * v == obs_dim:
            candidates.append((n, v))
    if not candidates:
        raise ValueError(f"Cannot infer dynamic shape from obs_dim={obs_dim}, action_dim={action_dim}")
    candidates.sort(key=lambda x: (abs(x[0] - 8), x[1]))
    return candidates[0]


def _build_mask_for_service(ep: _DynamicEpisode, service: dict[str, object], residual_cpu: list[float], residual_mem: list[float], migration_used: int, max_nodes: int, max_variants: int) -> np.ndarray:
    action_dim = max_nodes * max_variants
    mask = np.zeros((action_dim,), dtype=np.float32)
    node_index = {n["node_id"]: i for i, n in enumerate(ep.nodes)}
    variants = service["variants"]  # type: ignore[index]
    for vi, var in enumerate(variants):  # type: ignore[arg-type]
        if vi >= max_variants:
            break
        for ni, node in enumerate(ep.nodes):
            if ni >= max_nodes:
                break
            delta_mig = 0
            if service["prev_variant"] != var["variant_id"] or service["prev_node"] != node["node_id"]:  # type: ignore[index]
                delta_mig = 1
            if migration_used + delta_mig > ep.storm_max:
                continue
            if residual_cpu[ni] + 1e-9 < float(var["cpu"]):
                continue
            if residual_mem[ni] + 1e-9 < float(var["mem_gb"]):
                continue
            mask[_action_index(vi, ni, max_nodes)] = 1.0
    if float(mask.sum()) <= 0.0:
        prev_node_idx = node_index.get(str(service["prev_node"]), 0)
        prev_variant_idx = 0
        for i, v in enumerate(variants):  # type: ignore[arg-type]
            if v["variant_id"] == service["prev_variant"]:
                prev_variant_idx = i
                break
        prev_variant_idx = min(prev_variant_idx, max_variants - 1)
        prev_node_idx = min(prev_node_idx, max_nodes - 1)
        mask[_action_index(prev_variant_idx, prev_node_idx, max_nodes)] = 1.0
    return mask


def _vectorize_state_for_service(ep: _DynamicEpisode, service: dict[str, object], step_idx: int, residual_cpu: list[float], residual_mem: list[float], migration_used: int, max_nodes: int, max_variants: int, ds) -> np.ndarray:
    max_cap_cpu = max(float(n["cap_cpu"]) for n in ep.nodes)
    max_cap_mem = max(float(n["cap_mem_gb"]) for n in ep.nodes)
    max_energy = max(float(n["energy_cost"]) for n in ep.nodes)
    max_mem_energy = max(max(float(n["e_mem_unit"]) for n in ep.nodes), 1e-6)
    max_mig_cost = max(float(s["migration_cost"]) for s in ep.services)
    feats: list[float] = []
    feats.extend([
        len(ep.nodes) / max(max_nodes, 1),
        len(ep.services) / max(len(ep.services), 1),
        float(ds.w_c),
        float(ds.w_d),
        float(ds.w_a),
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
    variants = service["variants"]  # type: ignore[index]
    for i in range(max_variants):
        if i < len(variants):  # type: ignore[arg-type]
            v = variants[i]  # type: ignore[index]
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
        1.0 if len(variants) > 1 else 0.0,  # type: ignore[arg-type]
        prev_node_norm,
        1.0 if service["service_type"] in {"detection", "gen_ai"} else 0.0,
        len(variants) / max(max_variants, 1),  # type: ignore[arg-type]
    ])
    return np.asarray(feats, dtype=np.float32)


class Legacy44Adapter(BaseInferenceAdapter):
    name = "legacy44_adapter"
    contract = ModelContract.LEGACY_44

    def __init__(self, model: object, obs_dim: int) -> None:
        super().__init__(model, obs_dim)
        try:
            from src.drl.edge_env import compute_action_masks  # noqa: PLC0415
        except ImportError:
            from drl.edge_env import compute_action_masks  # noqa: PLC0415
        self.compute_action_masks = compute_action_masks
        try:
            self.action_dims = [int(x) for x in np.asarray(model.action_space.nvec, dtype=np.int64).tolist()]  # type: ignore[attr-defined]
        except Exception:
            self.action_dims = [int(x) for x in LEGACY_ACTION_DIMS]

    def _head_logits(self, obs: np.ndarray, flat_mask: np.ndarray | None) -> list[np.ndarray]:
        try:
            import torch  # noqa: PLC0415

            with torch.inference_mode():
                obs_t, _ = self.model.policy.obs_to_tensor(obs)  # type: ignore[attr-defined]
                try:
                    features = self.model.policy.extract_features(obs_t)  # type: ignore[attr-defined]
                    latent_pi, _ = self.model.policy.mlp_extractor(features)  # type: ignore[attr-defined]
                    flat_logits = (
                        self.model.policy.action_net(latent_pi)  # type: ignore[attr-defined]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    out = split_head_logits(flat_logits, action_dims=self.action_dims)
                    if len(out) == len(self.action_dims):
                        return out
                except Exception:
                    pass

                dist = None
                if flat_mask is not None:
                    try:
                        dist = self.model.policy.get_distribution(  # type: ignore[attr-defined]
                            obs_t,
                            action_masks=flat_mask.reshape(1, -1).astype(bool),
                        )
                    except Exception:
                        dist = self.model.policy.get_distribution(obs_t)  # type: ignore[attr-defined]
                else:
                    dist = self.model.policy.get_distribution(obs_t)  # type: ignore[attr-defined]
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

    def predict(self, rdb: redis_lib.Redis) -> PlacementResult:
        state = _build_state_from_redis(rdb)
        ds = _build_milp_dataset_from_state(state)
        action_masks = self.compute_action_masks(state)

        try:
            from src.drl.reward import evaluate_objective_for_placement  # noqa: PLC0415
        except ImportError:
            from drl.reward import evaluate_objective_for_placement  # noqa: PLC0415

        t_policy = time.perf_counter()
        head_logits = self._head_logits(state, action_masks)
        masked_logits = apply_flat_mask_to_head_logits(
            head_logits=head_logits,
            flat_mask=action_masks,
            action_dims=self.action_dims,
        )
        outcome = select_stormsafe_action(
            dataset=ds,
            head_logits=masked_logits,
            decode_action_fn=_decode_legacy_action_ids,
            objective_fn=evaluate_objective_for_placement,
            action_dims=self.action_dims,
            topk_per_head=LEGACY_TOPK_PER_HEAD,
        )
        policy_ms = (time.perf_counter() - t_policy) * 1000.0
        action = np.asarray(outcome.action, dtype=np.int64)
        placement = _action_to_placement(action)
        return PlacementResult(
            placement=placement,
            action=action,
            adapter_name=self.name,
            model_contract=self.contract,
            obs_dim=self.obs_dim,
            notes="legacy_44_multidiscrete_stormsafe",
            inference_core_ms=policy_ms,
            decode_mode=outcome.decoder_mode,
            storm_safe=True,
            migration_count=int(outcome.migration_count),
            storm_max=int(outcome.storm_max),
            fallback_full_enumeration=bool(outcome.fallback_full_enumeration),
        )


class Dynamic96Adapter(BaseInferenceAdapter):
    name = "dynamic96_adapter"
    contract = ModelContract.DYNAMIC_96

    def __init__(self, model: object, obs_dim: int) -> None:
        super().__init__(model, obs_dim)
        action_dim = int(model.action_space.n)  # type: ignore[attr-defined]
        self.max_nodes, self.max_variants = _infer_dynamic_shape(obs_dim, action_dim)
        self.decode_mode = _DRL_DYNAMIC_DECODE_MODE if _DRL_DYNAMIC_DECODE_MODE in {"greedy", "beam"} else "greedy"
        self.beam_width = _DRL_DYNAMIC_BEAM_WIDTH
        self.topk_actions = _DRL_DYNAMIC_TOPK_ACTIONS
        if _DRL_DYNAMIC_DECODE_MODE not in {"greedy", "beam"}:
            log.warning(
                "Invalid DRL_DYNAMIC_DECODE_MODE=%s; fallback to greedy",
                _DRL_DYNAMIC_DECODE_MODE,
            )

    def _action_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            import torch  # noqa: PLC0415

            with torch.inference_mode():
                obs_t, _ = self.model.policy.obs_to_tensor(obs)  # type: ignore[attr-defined]
                dist = self.model.policy.get_distribution(  # type: ignore[attr-defined]
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

    def _predict_greedy_action_index(self, obs: np.ndarray, mask: np.ndarray) -> int | None:
        valid_idx = np.where(mask > 0.5)[0]
        if len(valid_idx) <= 0:
            return None
        try:
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask.astype(bool))
            action_idx = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
            if mask[action_idx] > 0.5:
                return action_idx
        except Exception:
            pass
        probs = self._action_probs(obs, mask)
        masked_probs = np.where(mask > 0.5, probs, -np.inf)
        return int(np.argmax(masked_probs))

    def _greedy_policy_placement(
        self,
        ep: object,
        services: list[dict],
        ds: object,
    ) -> tuple[dict[str, tuple[str, str]], float]:
        residual_cpu = [float(n["cap_cpu"]) for n in ep.nodes]
        residual_mem = [float(n["cap_mem_gb"]) for n in ep.nodes]
        migration_used = 0
        placement_id: dict[str, tuple[str, str]] = {}
        policy_ms = 0.0

        for step_idx, svc in enumerate(services):
            mask = _build_mask_for_service(
                ep, svc, residual_cpu, residual_mem, migration_used, self.max_nodes, self.max_variants
            )
            obs = _vectorize_state_for_service(
                ep, svc, step_idx, residual_cpu, residual_mem, migration_used, self.max_nodes, self.max_variants, ds
            )
            t_policy = time.perf_counter()
            action_idx = self._predict_greedy_action_index(obs, mask)
            policy_ms += (time.perf_counter() - t_policy) * 1000.0
            if action_idx is None:
                chosen_variant = str(svc["prev_variant"])
                chosen_node = str(svc["prev_node"])
            else:
                vi, ni = _decode_action_index(action_idx, self.max_nodes)
                variants = svc["variants"]  # type: ignore[index]
                if ni >= len(ep.nodes) or vi >= len(variants):  # type: ignore[arg-type]
                    chosen_variant = str(svc["prev_variant"])
                    chosen_node = str(svc["prev_node"])
                else:
                    chosen_variant = str(variants[vi]["variant_id"])  # type: ignore[index]
                    chosen_node = str(ep.nodes[ni]["node_id"])
            placement_id[str(svc["service_id"])] = (chosen_variant, chosen_node)
            var_obj = next(
                (v for v in svc["variants"] if str(v["variant_id"]) == chosen_variant),
                svc["variants"][0],
            )
            node_idx = next((i for i, n in enumerate(ep.nodes) if str(n["node_id"]) == chosen_node), 0)
            residual_cpu[node_idx] -= float(var_obj["cpu"])
            residual_mem[node_idx] -= float(var_obj["mem_gb"])
            if svc["prev_variant"] != chosen_variant or svc["prev_node"] != chosen_node:
                migration_used += 1

        for svc in services:
            sid = str(svc["service_id"])
            if sid not in placement_id:
                placement_id[sid] = (str(svc["prev_variant"]), str(svc["prev_node"]))
        return placement_id, policy_ms

    def _beam_policy_placement(
        self,
        ep: object,
        services: list[dict],
        ds: object,
    ) -> tuple[dict[str, tuple[str, str]], float]:
        @dataclass
        class _BeamState:
            placement: dict[str, tuple[str, str]]
            residual_cpu: list[float]
            residual_mem: list[float]
            migration_used: int
            logp: float

        beams = [
            _BeamState(
                placement={},
                residual_cpu=[float(n["cap_cpu"]) for n in ep.nodes],
                residual_mem=[float(n["cap_mem_gb"]) for n in ep.nodes],
                migration_used=0,
                logp=0.0,
            )
        ]
        policy_ms = 0.0

        for step_idx, svc in enumerate(services):
            expanded: list[_BeamState] = []
            for b in beams:
                mask = _build_mask_for_service(
                    ep, svc, b.residual_cpu, b.residual_mem, b.migration_used, self.max_nodes, self.max_variants
                )
                obs = _vectorize_state_for_service(
                    ep, svc, step_idx, b.residual_cpu, b.residual_mem, b.migration_used, self.max_nodes, self.max_variants, ds
                )
                t_policy = time.perf_counter()
                probs = self._action_probs(obs, mask)
                policy_ms += (time.perf_counter() - t_policy) * 1000.0
                valid_count = int((mask > 0.5).sum())
                if valid_count <= 0:
                    continue
                k = max(1, min(self.topk_actions, valid_count))
                top_idx = np.argpartition(-probs, k - 1)[:k]
                top_idx = top_idx[np.argsort(-probs[top_idx])]
                for action_idx in top_idx.tolist():
                    if mask[action_idx] < 0.5:
                        continue
                    vi, ni = _decode_action_index(int(action_idx), self.max_nodes)
                    variants = svc["variants"]  # type: ignore[index]
                    if ni >= len(ep.nodes) or vi >= len(variants):  # type: ignore[arg-type]
                        continue
                    chosen_var = variants[vi]  # type: ignore[index]
                    chosen_node = str(ep.nodes[ni]["node_id"])
                    nxt = _BeamState(
                        placement=dict(b.placement),
                        residual_cpu=list(b.residual_cpu),
                        residual_mem=list(b.residual_mem),
                        migration_used=b.migration_used,
                        logp=b.logp + math.log(max(float(probs[action_idx]), 1e-12)),
                    )
                    nxt.placement[str(svc["service_id"])] = (str(chosen_var["variant_id"]), chosen_node)
                    nxt.residual_cpu[ni] -= float(chosen_var["cpu"])
                    nxt.residual_mem[ni] -= float(chosen_var["mem_gb"])
                    if svc["prev_variant"] != chosen_var["variant_id"] or svc["prev_node"] != chosen_node:
                        nxt.migration_used += 1
                    expanded.append(nxt)
            if not expanded:
                break
            expanded.sort(key=lambda x: x.logp, reverse=True)
            beams = expanded[: max(1, self.beam_width)]

        try:
            from src.drl.reward import evaluate_objective_for_placement  # noqa: PLC0415
        except ImportError:
            from drl.reward import evaluate_objective_for_placement  # noqa: PLC0415

        for b in beams:
            for svc in services:
                sid = str(svc["service_id"])
                if sid not in b.placement:
                    b.placement[sid] = (str(svc["prev_variant"]), str(svc["prev_node"]))
        best = None
        best_key = None
        for b in beams:
            try:
                obj = evaluate_objective_for_placement(ds, b.placement).objective_j
            except Exception:
                obj = float("inf")
            key = (obj, -b.logp)
            if best is None or key < best_key:
                best = b
                best_key = key
        placement_id = best.placement if best is not None else {
            str(s["service_id"]): (str(s["prev_variant"]), str(s["prev_node"])) for s in services
        }
        return placement_id, policy_ms

    def predict(self, rdb: redis_lib.Redis) -> PlacementResult:
        state_44 = _build_state_from_redis(rdb)
        ds = _build_milp_dataset_from_state(state_44)
        ep = _build_dynamic_episode(ds)
        services = sorted(ep.services, key=lambda s: str(s["service_id"]))

        if self.decode_mode == "beam":
            placement_id, policy_ms = self._beam_policy_placement(ep, services, ds)
            notes = "dynamic_96_discrete_beam_proxy"
        else:
            placement_id, policy_ms = self._greedy_policy_placement(ep, services, ds)
            notes = "dynamic_96_discrete_greedy_live"
        node_name = {
            "n0": "edge-nodes-1",
            "n1": "edge-nodes-2",
            "n2": "edge-nodes-3",
            "n3": "edge-nodes-4",
        }
        svc_name = {
            "m0": "api-gateway",
            "m1": "ingest",
            "m2": "preprocess",
            "m3": "detection",
            "m4": "gen-ai",
            "m5": "postprocess",
        }
        placement: dict[str, dict[str, str]] = {}
        for sid, (variant, nid) in placement_id.items():
            placement[svc_name[sid]] = {"node": node_name[nid], "variant": variant}
        action = _placement_to_action(placement)
        if action is None:
            raise ValueError("dynamic96 adapter produced non-encodable placement")
        return PlacementResult(
            placement=placement,
            action=np.asarray(action, dtype=np.int64),
            adapter_name=self.name,
            model_contract=self.contract,
            obs_dim=self.obs_dim,
            notes=notes,
            inference_core_ms=policy_ms,
            decode_mode=self.decode_mode,
        )


def _build_adapter_for_model(model: object, contract: ModelContract, obs_dim: int) -> BaseInferenceAdapter | None:
    if contract == ModelContract.LEGACY_44:
        return Legacy44Adapter(model, obs_dim)
    if contract == ModelContract.DYNAMIC_96:
        return Dynamic96Adapter(model, obs_dim)
    return None


# ── Inference loop ────────────────────────────────────────────────────────────
def _inference_loop(
    rdb: redis_lib.Redis,
    adapter: BaseInferenceAdapter | None,
    validator: DigitalTwinValidator,
    inference_enabled: bool,
) -> None:
    """Background thread: runs PPO inference every 5 s.

    Digital Twin gate (Phase 5):
        Before committing any drl:placement, the proposed action is validated
        through N_ROLLOUTS Monte Carlo simulations in a Digital Twin
        synchronised with the current Redis cluster state.  Only actions that
        are feasible in ≥MIN_FEASIBLE_RATE of perturbed rollouts AND exceed
        SAFETY_THRESHOLD are written to drl:placement.  Rejected actions leave
        drl:placement unwritten; the scheduler extender (/prioritize) then falls
        back to milp:placement so MILP node scores govern scheduling.
        Validation stats are always written to drl:twin_stats (TTL 35 s).
    """
    log.info(
        "Digital Twin validator: n_rollouts=%d safety_threshold=%.2f "
        "min_feasible_rate=%.0f%%",
        validator.n_rollouts,
        validator.safety_threshold,
        validator.min_feasible_rate * 100,
    )

    milp_j_window: deque = deque(maxlen=_EPSILON_WINDOW)
    projection_window: deque = deque(maxlen=_EPSILON_WINDOW)
    reject_streak = 0
    last_action_tuple: tuple[int, ...] | None = None
    repeated_action_streak = 0

    while True:
        t0 = time.perf_counter()
        try:
            mode = rdb.get("system:mode") or "milp"
            if not inference_enabled or adapter is None:
                if mode in ("shadow", "drl", "hybrid"):
                    log.warning(
                        "Inference disabled (adapter=%s contract=%s mode=%s)",
                        _runtime_state.get("adapter_name"),
                        _runtime_state.get("model_contract"),
                        mode,
                    )
                rdb.setex(
                    "drl:twin_stats",
                    35,
                    json.dumps(
                        {
                            "is_safe": False,
                            "reason": "inference_disabled",
                            "model_contract": _runtime_state.get("model_contract"),
                            "adapter_name": _runtime_state.get("adapter_name"),
                            "inference_enabled": False,
                            "inference_error_count": _runtime_state.get("inference_error_count", 0),
                            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                        }
                    ),
                )
                time.sleep(max(0.0, _INFERENCE_INTERVAL_S - (time.perf_counter() - t0)))
                continue

            if mode in ("shadow", "drl", "hybrid"):
                placement_result = adapter.predict(rdb)
                placement = placement_result.placement
                action = np.asarray(placement_result.action, dtype=np.int64)
                state = _build_state_from_redis(rdb)
                raw_action = np.asarray(action, dtype=np.int64).copy()
                raw_placement = {k: dict(v) for k, v in placement.items()}

                # ── Digital Twin validation gate ──────────────────────────
                # Use confirmed live cluster placement as single source of truth.
                _raw_cp = rdb.get("milp:confirmed_placement")
                try:
                    _confirmed = json.loads(_raw_cp) if _raw_cp else None
                except Exception:
                    _confirmed = None

                # Track the current MILP snapshot early so every guard in this
                # cycle compares candidates against the same reference.
                _milp_ref_ts: str | None = None
                _milp_ref_obj: float | None = None
                _milp_payload: dict | None = None
                try:
                    _raw_milp = rdb.get("milp:placement")
                    if _raw_milp:
                        _milp_payload = json.loads(_raw_milp)
                        _milp_j = _milp_payload.get("objective")
                        _milp_ref_ts = _milp_payload.get("timestamp")
                        if _milp_j is not None:
                            _milp_ref_obj = float(_milp_j)
                            milp_j_window.append(_milp_ref_obj)
                except Exception:
                    pass
                rolling_milp_j = float(np.mean(list(milp_j_window))) if milp_j_window else None

                projected, projected_applied, mig_before, mig_after = _project_placement_to_storm_budget(
                    placement,
                    _confirmed,
                    _STORM_PROJECTION_BUDGET,
                )
                if projected_applied:
                    projected_action = _placement_to_action(projected)
                    if projected_action is not None:
                        placement = projected
                        action = np.asarray(projected_action, dtype=np.int64)
                        log.info(
                            "Storm projection applied before Twin: migrations %d -> %d (budget=%d)",
                            mig_before,
                            mig_after,
                            _STORM_PROJECTION_BUDGET,
                        )
                        _push_storm_correction_sample(
                            rdb,
                            mode=mode,
                            state=state,
                            confirmed_placement=_confirmed,
                            raw_action=raw_action,
                            projected_action=action,
                            raw_placement=raw_placement,
                            projected_placement=placement,
                            migration_before=mig_before,
                            migration_after=mig_after,
                            storm_budget=_STORM_PROJECTION_BUDGET,
                            model_contract=placement_result.model_contract.value,
                            adapter_name=placement_result.adapter_name,
                        )
                projection_window.append(1 if projected_applied else 0)
                projection_rate = float(np.mean(list(projection_window))) if projection_window else 0.0

                action_tuple = tuple(int(x) for x in np.asarray(action, dtype=np.int64).tolist())
                if action_tuple == last_action_tuple:
                    repeated_action_streak += 1
                else:
                    repeated_action_streak = 1
                    last_action_tuple = action_tuple

                twin_result = validator.validate(action, state, confirmed_placement=_confirmed)
                candidate_rerank_mode_enabled, candidate_rerank_include_milp = _candidate_rerank_mode_settings(mode)
                candidate_rerank = _rerank_action_candidates(
                    validator=validator,
                    state=state,
                    confirmed_placement=_confirmed,
                    current_action=action,
                    current_placement=placement,
                    current_twin_result=twin_result,
                    milp_payload=_milp_payload,
                    enabled=candidate_rerank_mode_enabled,
                    min_improvement=_RERANK_MIN_IMPROVEMENT,
                    include_milp=candidate_rerank_include_milp,
                    include_confirmed=_RERANK_INCLUDE_CONFIRMED,
                )
                if candidate_rerank.applied:
                    action = np.asarray(candidate_rerank.action, dtype=np.int64)
                    placement = candidate_rerank.placement
                    twin_result = candidate_rerank.twin_result
                    action_tuple = tuple(int(x) for x in action.tolist())
                    last_action_tuple = action_tuple
                    repeated_action_streak = 1
                    log.info(
                        "Candidate rerank selected %s (reason=%s, candidates=%d, J=%.4f)",
                        candidate_rerank.selected_source,
                        candidate_rerank.reason,
                        candidate_rerank.candidate_count,
                        float(getattr(twin_result, "objective_j_raw", 0.0)),
                    )
                sticky_override_applied = False
                sticky_override_reason = ""
                if (
                    _ACTION_STICKY_GUARD_ENABLED
                    and twin_result.is_safe
                    and repeated_action_streak >= max(2, _ACTION_STICKY_WINDOW)
                ):
                    sticky_candidates: list[np.ndarray] = [np.asarray(action, dtype=np.int64)]
                    confirmed_action = _placement_to_action(_confirmed)
                    if (
                        confirmed_action is not None
                        and not any(np.array_equal(confirmed_action, c) for c in sticky_candidates)
                    ):
                        sticky_candidates.append(confirmed_action)
                    milp_action = None
                    try:
                        _raw_milp_for_action = rdb.get("milp:placement")
                        _milp_for_action = json.loads(_raw_milp_for_action) if _raw_milp_for_action else None
                        if isinstance(_milp_for_action, dict):
                            milp_action = _placement_to_action(_milp_for_action.get("placement"))
                    except Exception:
                        milp_action = None
                    if (
                        milp_action is not None
                        and not any(np.array_equal(milp_action, c) for c in sticky_candidates)
                    ):
                        sticky_candidates.append(milp_action)

                    if len(sticky_candidates) > 1:
                        prev_j = twin_result.objective_j_raw
                        best_action, best_result = validator.find_best_action(
                            sticky_candidates,
                            state,
                            confirmed_placement=_confirmed,
                        )
                        improved = (
                            best_result.is_safe
                            and best_result.objective_j_raw
                            < (twin_result.objective_j_raw - _ACTION_STICKY_MIN_IMPROVEMENT)
                        )
                        changed = not np.array_equal(np.asarray(action, dtype=np.int64), np.asarray(best_action, dtype=np.int64))
                        if improved and changed:
                            best_placement = _action_to_placement(np.asarray(best_action, dtype=np.int64))
                            if _placement_complete(best_placement):
                                action = np.asarray(best_action, dtype=np.int64)
                                placement = best_placement
                                twin_result = best_result
                                sticky_override_applied = True
                                sticky_override_reason = "sticky_action_replaced_by_better_safe_candidate"
                                repeated_action_streak = 1
                                last_action_tuple = tuple(int(x) for x in action.tolist())
                                log.info(
                                    "Sticky-action guard switched action after streak=%d "
                                    "(J_raw %.3f -> %.3f)",
                                    _ACTION_STICKY_WINDOW,
                                    prev_j,
                                    twin_result.objective_j_raw,
                                )

                twin_stats_payload = {
                    **twin_result.to_dict(),
                    "model_contract": placement_result.model_contract.value,
                    "adapter_name": placement_result.adapter_name,
                    "obs_dim": placement_result.obs_dim,
                    "decode_mode": placement_result.decode_mode,
                    "storm_safe": placement_result.storm_safe,
                    "migration_count_policy": placement_result.migration_count,
                    "storm_max_policy": placement_result.storm_max,
                    "fallback_full_enumeration": placement_result.fallback_full_enumeration,
                    "inference_core_ms": placement_result.inference_core_ms,
                    "inference_enabled": inference_enabled,
                    "inference_error_count": int(_runtime_state.get("inference_error_count", 0)),
                    "adapter_notes": placement_result.notes,
                    "storm_projection_applied": projected_applied,
                    "storm_projection_budget": _STORM_PROJECTION_BUDGET,
                    "migration_count_before_projection": mig_before,
                    "migration_count_after_projection": mig_after,
                    "projection_rate_window": projection_rate,
                    "projection_window_size": len(projection_window),
                    "candidate_rerank_enabled": _CANDIDATE_RERANK_ENABLED,
                    "candidate_rerank_mode_enabled": candidate_rerank_mode_enabled,
                    "candidate_rerank_modes": sorted(_CANDIDATE_RERANK_MODES),
                    "candidate_rerank_include_milp": candidate_rerank_include_milp,
                    "candidate_rerank_applied": candidate_rerank.applied,
                    "candidate_rerank_reason": candidate_rerank.reason,
                    "candidate_rerank_selected_source": candidate_rerank.selected_source,
                    "candidate_count": candidate_rerank.candidate_count,
                    "repeated_action_streak": repeated_action_streak,
                    "sticky_override_applied": sticky_override_applied,
                    "sticky_override_reason": sticky_override_reason,
                    "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                }
                rdb.setex("drl:twin_stats", 35, json.dumps(twin_stats_payload))

                _committed = False

                if not twin_result.is_safe:
                    reject_streak += 1
                    if reject_streak >= _REJECT_RETRY_AFTER:
                        candidates: list[np.ndarray] = [np.asarray(action, dtype=np.int64)]
                        confirmed_action = _placement_to_action(_confirmed)
                        if confirmed_action is not None and not any(
                            np.array_equal(confirmed_action, c) for c in candidates
                        ):
                            candidates.append(confirmed_action)
                        for _ in range(max(0, _REJECT_CANDIDATES - 1)):
                            # Adapter may be non-stochastic or sequential; keep current action
                            # as fallback candidate to avoid contract-specific random sampling bugs.
                            alt_action = np.asarray(action, dtype=np.int64)
                            if not any(np.array_equal(alt_action, c) for c in candidates):
                                candidates.append(alt_action)
                        if len(candidates) > 1:
                            rescue_action, rescue_result = validator.find_best_action(
                                candidates,
                                state,
                                confirmed_placement=_confirmed,
                            )
                            if rescue_result.is_safe:
                                action = np.asarray(rescue_action, dtype=np.int64)
                                placement = _action_to_placement(action)
                                twin_result = rescue_result
                                twin_stats_payload = {
                                    **twin_result.to_dict(),
                                    "model_contract": placement_result.model_contract.value,
                                    "adapter_name": placement_result.adapter_name,
                                    "obs_dim": placement_result.obs_dim,
                                    "decode_mode": placement_result.decode_mode,
                                    "storm_safe": placement_result.storm_safe,
                                    "migration_count_policy": placement_result.migration_count,
                                    "storm_max_policy": placement_result.storm_max,
                                    "fallback_full_enumeration": placement_result.fallback_full_enumeration,
                                    "inference_core_ms": placement_result.inference_core_ms,
                                    "inference_enabled": inference_enabled,
                                    "inference_error_count": int(_runtime_state.get("inference_error_count", 0)),
                                    "adapter_notes": placement_result.notes,
                                    "storm_projection_applied": projected_applied,
                                    "storm_projection_budget": _STORM_PROJECTION_BUDGET,
                                    "migration_count_before_projection": mig_before,
                                    "migration_count_after_projection": mig_after,
                                    "projection_rate_window": projection_rate,
                                    "projection_window_size": len(projection_window),
                                    "candidate_rerank_enabled": _CANDIDATE_RERANK_ENABLED,
                                    "candidate_rerank_mode_enabled": candidate_rerank_mode_enabled,
                                    "candidate_rerank_modes": sorted(_CANDIDATE_RERANK_MODES),
                                    "candidate_rerank_include_milp": candidate_rerank_include_milp,
                                    "candidate_rerank_applied": candidate_rerank.applied,
                                    "candidate_rerank_reason": candidate_rerank.reason,
                                    "candidate_rerank_selected_source": candidate_rerank.selected_source,
                                    "candidate_count": candidate_rerank.candidate_count,
                                    "repeated_action_streak": repeated_action_streak,
                                    "sticky_override_applied": sticky_override_applied,
                                    "sticky_override_reason": sticky_override_reason,
                                    "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                                }
                                rdb.setex("drl:twin_stats", 35, json.dumps(twin_stats_payload))
                                reject_streak = 0
                                log.info(
                                    "Twin rescue accepted candidate after reject streak=%d "
                                    "(J_raw=%.3f migrations=%d/%d)",
                                    _REJECT_RETRY_AFTER,
                                    twin_result.objective_j_raw,
                                    twin_result.migration_count,
                                    twin_result.v_storm_max,
                                )
                            else:
                                log.warning(
                                    "Twin rescue failed after reject streak=%d "
                                    "(no safe candidate in %d samples)",
                                    reject_streak,
                                    len(candidates),
                                )
                else:
                    reject_streak = 0

                proposed_payload = {
                    "placement": placement,
                    "action": action.tolist(),
                    "twin_stats": twin_stats_payload,
                    "accepted": False,
                    "committed": False,
                    "commit_reason": "",
                }

                if not twin_result.is_safe:
                    proposed_payload["commit_reason"] = f"rejected:{twin_result.reason}"
                    log.warning(
                        "Twin REJECTED action (J_raw=%.3f J_pen=%.3f reason=%s streak=%d) "
                        "— MILP retains control (mode=%s)",
                        twin_result.objective_j_raw,
                        twin_result.objective_j_penalized,
                        twin_result.reason,
                        reject_streak,
                        mode,
                    )
                elif (
                    rolling_milp_j is not None
                    and twin_result.objective_j_raw > rolling_milp_j * (1 - _EPSILON_FALLBACK)
                ):
                    proposed_payload["commit_reason"] = "epsilon_fallback_to_milp"
                    gap_pct = (twin_result.objective_j_raw - rolling_milp_j) / abs(rolling_milp_j) * 100
                    log.warning(
                        "Epsilon fallback: DRL J_raw=%.3f gap=+%.1f%% > eps=%.0f%% "
                        "\u2014 MILP retains control (mode=%s)",
                        twin_result.objective_j_raw, gap_pct, _EPSILON_FALLBACK * 100, mode,
                    )
                else:
                    if not _placement_complete(placement):
                        proposed_payload["commit_reason"] = "invalid_incomplete_placement"
                        log.error(
                            "Adapter produced incomplete placement; publish blocked "
                            "(adapter=%s contract=%s)",
                            placement_result.adapter_name,
                            placement_result.model_contract.value,
                        )
                        _runtime_state["inference_error_count"] = int(_runtime_state.get("inference_error_count", 0)) + 1
                        _write_proposed_placement(rdb, proposed_payload)
                        continue
                    proposed_payload["accepted"] = True
                    reject_streak = 0

                    # `shadow` stays observe-only: accept by Twin, but never commit.
                    if mode == "shadow":
                        proposed_payload["committed"] = False
                        proposed_payload["commit_reason"] = "shadow_mode_observe_only"
                        log.info(
                            "Twin accepted proposal in shadow mode; commit skipped (observe-only)"
                        )
                    else:
                        # ── Compute migration_types from confirmed placement ────────
                        try:
                            confirmed_raw = rdb.get("milp:confirmed_placement")
                            confirmed = json.loads(confirmed_raw) if confirmed_raw else {}
                        except Exception:
                            confirmed = {}
                        migration_types: dict[str, str] = {}
                        for svc_name, info in placement.items():
                            prev = confirmed.get(svc_name, {})
                            if not prev:
                                migration_types[svc_name] = "New"
                            elif (prev.get("node") == info["node"]
                                  and prev.get("variant") == info.get("variant", "standard")):
                                migration_types[svc_name] = "Stayed"
                            elif prev.get("variant") != info.get("variant", "standard"):
                                migration_types[svc_name] = "AI Model Changed"
                            else:
                                migration_types[svc_name] = "Moved"

                        # ── Build basic node_scores (DRL-preferred nodes get bonus) ─
                        node_scores: dict[str, int] = {}
                        for info in placement.values():
                            node = info["node"]
                            node_scores[node] = min(100, node_scores.get(node, 50) + 10)

                        breakdown = _compute_objective_breakdown(placement, state)
                        aligned_with_milp = _placement_matches_payload(placement, _milp_payload)
                        if aligned_with_milp and _milp_payload:
                            breakdown_source = "milp_snapshot_aligned"
                            cost_energy = _milp_payload.get("cost_energy")
                            cost_disruption = _milp_payload.get("cost_disruption")
                            gain_accuracy = _milp_payload.get("gain_accuracy")
                            norm_cost_energy = _milp_payload.get("norm_cost_energy")
                            norm_cost_disruption = _milp_payload.get("norm_cost_disruption")
                            norm_gain_accuracy = _milp_payload.get("norm_gain_accuracy")
                        else:
                            breakdown_source = "reward_estimate"
                            cost_energy = (breakdown or {}).get("cost_energy")
                            cost_disruption = (breakdown or {}).get("cost_disruption")
                            gain_accuracy = (breakdown or {}).get("gain_accuracy")
                            norm_cost_energy = (breakdown or {}).get("norm_cost_energy")
                            norm_cost_disruption = (breakdown or {}).get("norm_cost_disruption")
                            norm_gain_accuracy = (breakdown or {}).get("norm_gain_accuracy")

                        # Keep the published `objective` comparable with MILP.
                        #
                        # Digital Twin MC mean is useful as a robustness/risk estimate,
                        # but it is not the same deterministic objective emitted by MILP.
                        # When both payloads describe the same placement and MILP snapshot,
                        # use MILP's exact objective so Hybrid/dashboard cannot report a
                        # fake DRL win caused only by Twin perturbation noise.
                        twin_expected_objective = float(twin_result.objective_j_raw)
                        twin_expected_objective_penalized = float(twin_result.objective_j_penalized)
                        comparable_objective = twin_expected_objective
                        comparable_source = "twin_mc_mean_fallback"
                        if aligned_with_milp and _milp_ref_obj is not None:
                            comparable_objective = float(_milp_ref_obj)
                            comparable_source = "milp_snapshot_exact"
                        elif breakdown and breakdown.get("objective_reward") is not None:
                            comparable_objective = float(breakdown["objective_reward"])
                            comparable_source = "reward_breakdown"

                        # ── Full payload (milp:placement-compatible schema) ─────────
                        decision_ms = max(1.0, (time.perf_counter() - t0) * 1000.0)
                        inference_core_ms = float(placement_result.inference_core_ms or decision_ms)
                        drl_payload = {
                            "placement": placement,
                            "objective": comparable_objective,
                            "objective_penalized": comparable_objective,
                            "comparable_objective": comparable_objective,
                            "comparable_objective_source": comparable_source,
                            "twin_expected_objective": twin_expected_objective,
                            "twin_expected_objective_penalized": twin_expected_objective_penalized,
                            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                            "milp_ref_timestamp": _milp_ref_ts,
                            "milp_ref_objective": _milp_ref_obj,
                            "objective_source": comparable_source,
                            "twin_objective_source": "twin_mc_mean",
                            "objective_breakdown_source": breakdown_source,
                            "cost_energy": cost_energy,
                            "cost_disruption": cost_disruption,
                            "gain_accuracy": gain_accuracy,
                            "norm_cost_energy": norm_cost_energy,
                            "norm_cost_disruption": norm_cost_disruption,
                            "norm_gain_accuracy": norm_gain_accuracy,
                            "migration_types": migration_types,
                            "inference_time_ms": round(inference_core_ms, 3),
                            "decision_time_ms": round(decision_ms, 3),
                            "timing_scope": "policy_core_vs_full_decision",
                            "node_scores": node_scores,
                            "model_contract": placement_result.model_contract.value,
                            "adapter_name": placement_result.adapter_name,
                            "decode_mode": placement_result.decode_mode,
                            "storm_safe": placement_result.storm_safe,
                            "migration_count_policy": placement_result.migration_count,
                            "storm_max_policy": placement_result.storm_max,
                            "fallback_full_enumeration": placement_result.fallback_full_enumeration,
                        }

                        # ── Leader-gated, epoch-fenced write ───────────────────────────
                        if _elector is not None:
                            try:
                                from ha.leader_election import placement_write_guarded  # noqa: PLC0415
                                written = placement_write_guarded(
                                    rdb, "drl:placement", drl_payload, _elector, ttl=35
                                )
                            except Exception as le_exc:
                                log.warning("Leader election write guard failed: %s", le_exc)
                                written = False
                                rdb.setex("drl:placement", 35, json.dumps(drl_payload))
                        else:
                            # No elector (single-pod dev/test) — write directly
                            rdb.setex("drl:placement", 35, json.dumps(drl_payload))
                            written = True

                        if written:
                            _committed = True
                            proposed_payload["committed"] = True
                            proposed_payload["commit_reason"] = "committed"
                            log.info(
                                "Twin VALIDATED drl:placement written "
                                "(J_raw=%.3f J_pen=%.3f, feasible=%.0f%%, mode=%s)",
                                twin_result.objective_j_raw,
                                twin_result.objective_j_penalized,
                                twin_result.feasible_rate * 100,
                                mode,
                            )
                        else:
                            proposed_payload["committed"] = False
                            proposed_payload["commit_reason"] = "leader_gate_skipped"
                            log.info(
                                "drl:placement write skipped (standby/stale-epoch mode=%s)",
                                mode,
                            )
                proposed_written = _write_proposed_placement(rdb, proposed_payload)
                log.debug(
                    "drl:placement:proposed %s (TTL 60s accepted=%s)",
                    "written" if proposed_written else "skipped by leader gate",
                    proposed_payload["accepted"],
                )
                # ─────────────────────────────────────────────────────────

                if mode == "drl" and _committed:
                    # D3.6 — push online SARSA tuple to training buffer (cap 1000)
                    rdb.lpush(
                        "drl:training_buffer",
                        json.dumps({
                            "state": state.tolist(),
                            "action": action.tolist(),
                            "twin_predicted_j": twin_result.predicted_j,
                            "twin_objective_j_raw": twin_result.objective_j_raw,
                        }),
                    )
                    rdb.ltrim("drl:training_buffer", 0, 999)
                    log.debug(
                        "drl:training_buffer len=%s",
                        rdb.llen("drl:training_buffer"),
                    )
            else:
                log.debug("mode=%s — inference skipped", mode)

        except redis_lib.RedisError as exc:
            log.warning("Redis error in inference loop: %s", exc)
        except Exception as exc:
            _runtime_state["inference_error_count"] = int(_runtime_state.get("inference_error_count", 0)) + 1
            log.error("Unexpected error in inference loop: %s", exc, exc_info=True)

        # ── Heartbeat: written every cycle so alerts detect total loss ────────
        try:
            rdb.setex(
                "drl:heartbeat",
                _INFERENCE_INTERVAL_S * 6,  # TTL = 3 × write interval (5 s × 6 = 30 s)
                __import__("datetime").datetime.utcnow().isoformat(),
            )
            rdb.setex(
                "drl:runtime_status",
                _INFERENCE_INTERVAL_S * 6,
                json.dumps(
                    {
                        "model_contract": _runtime_state.get("model_contract"),
                        "adapter_name": _runtime_state.get("adapter_name"),
                        "inference_enabled": _runtime_state.get("inference_enabled"),
                        "inference_error_count": _runtime_state.get("inference_error_count"),
                        "storm_correction_buffer_key": _STORM_CORRECTION_BUFFER_KEY,
                        "storm_correction_buffer_len": rdb.llen(_STORM_CORRECTION_BUFFER_KEY),
                        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                    }
                ),
            )
        except Exception:
            pass  # non-critical; don't mask inference loop errors

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, _INFERENCE_INTERVAL_S - elapsed))



# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start inference background thread on startup."""
    global _rdb, _elector

    redis_host = os.getenv("REDIS_HOST", _DEFAULT_REDIS_HOST)
    redis_port = int(os.getenv("REDIS_PORT", str(_DEFAULT_REDIS_PORT)))
    model_path = os.getenv("DRL_MODEL_PATH", _DEFAULT_MODEL_PATH)

    log.info(
        "DRL Agent starting — redis=%s:%s model=%s",
        redis_host, redis_port, model_path,
    )

    adapter: BaseInferenceAdapter | None = None
    inference_enabled = True
    try:
        model, loaded_path = _load_model_with_fallback(model_path)
        contract, obs_dim, action_space = _resolve_model_contract(model)
        adapter = _build_adapter_for_model(model, contract, obs_dim)
        allowed = contract.value in _DRL_ALLOWED_CONTRACTS
        if contract == ModelContract.UNSUPPORTED or adapter is None or not allowed:
            msg = (
                f"Model contract unsupported/blocked: contract={contract.value} "
                f"obs_dim={obs_dim} action_space={action_space} allowed={allowed}"
            )
            if _DRL_DISABLE_ON_UNSUPPORTED:
                inference_enabled = False
                log.error("%s — inference disabled", msg)
            else:
                log.warning("%s — continuing without publish guarantees", msg)
        _runtime_state.update(
            {
                "model_contract": contract.value,
                "adapter_name": adapter.name if adapter else "none",
                "inference_enabled": inference_enabled and adapter is not None,
                "obs_dim": obs_dim,
                "action_space": action_space,
            }
        )
        log.info(
            "PPO model loaded OK — path=%s contract=%s obs_dim=%s action_space=%s adapter=%s inference_enabled=%s",
            loaded_path,
            _runtime_state["model_contract"],
            _runtime_state["obs_dim"],
            _runtime_state["action_space"],
            _runtime_state["adapter_name"],
            _runtime_state["inference_enabled"],
        )
    except Exception as exc:
        inference_enabled = False
        _runtime_state.update(
            {
                "model_contract": ModelContract.UNSUPPORTED.value,
                "adapter_name": "none",
                "inference_enabled": False,
            }
        )
        log.error("Failed to load PPO model: %s — inference disabled", exc)

    # ── Sentinel-aware Redis client ──────────────────────────────────────────
    try:
        from ha.redis_client import make_redis_client  # noqa: PLC0415
        _rdb = make_redis_client()
        log.info("Redis (Sentinel-aware) connected")
    except Exception:
        _rdb = redis_lib.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        try:
            _rdb.ping()
            log.info("Redis connected (plain): %s:%s", redis_host, redis_port)
        except redis_lib.RedisError as exc:
            log.warning("Redis not reachable at startup: %s — inference loop will retry", exc)

    # ── Leader election ──────────────────────────────────────────────────────
    if _SKIP_LEADER_ELECTION:
        log.info("Leader election disabled via DRL_SKIP_LEADER_ELECTION — single-pod mode")
        _elector = None
    else:
        try:
            from ha.leader_election import LeaderElector  # noqa: PLC0415
            _elector = LeaderElector(lease_name="drl-leader")
            _elector.start()
            log.info("LeaderElector started: drl-leader")
        except Exception as exc:
            log.warning("Leader election unavailable (%s) — single-pod mode", exc)
            _elector = None

    validator = DigitalTwinValidator(
        n_rollouts=_TWIN_N_ROLLOUTS,
        safety_threshold=_TWIN_SAFETY_THRESHOLD,
        min_feasible_rate=_TWIN_MIN_FEASIBLE_RATE,
    )
    log.info(
        "DigitalTwinValidator initialised: n_rollouts=%d safety_threshold=%.2f "
        "min_feasible_rate=%.2f",
        _TWIN_N_ROLLOUTS, _TWIN_SAFETY_THRESHOLD, _TWIN_MIN_FEASIBLE_RATE,
    )

    thread = threading.Thread(
        target=_inference_loop,
        args=(_rdb, adapter, validator, bool(_runtime_state.get("inference_enabled", inference_enabled))),
        daemon=True,
        name="drl-inference",
    )
    thread.start()
    log.info("Inference loop thread started")

    yield  # app is running

    if _elector is not None:
        _elector.stop()
    log.info("DRL Agent shutting down")


app = FastAPI(title="DRL Placement Agent", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    """Health check endpoint.

    Returns:
        {"status": "ok", "mode": "<current system:mode>"}
    """
    mode = _rdb.get("system:mode") if _rdb else "unknown"
    return {
        "status": "ok",
        "mode": mode or "milp",
        "model_contract": _runtime_state.get("model_contract"),
        "adapter_name": _runtime_state.get("adapter_name"),
        "inference_enabled": _runtime_state.get("inference_enabled"),
        "inference_error_count": _runtime_state.get("inference_error_count"),
    }


@app.get("/placement")
def get_placement() -> dict:
    """Return latest DRL placement decision from Redis."""
    if _rdb is None:
        return {"error": "redis not connected"}
    raw = _rdb.get("drl:placement")
    if raw is None:
        return {"placement": None, "message": "no drl:placement in Redis yet"}
    return {"placement": json.loads(raw)}


@app.get("/stats")
def get_stats() -> dict:
    """Return basic runtime statistics."""
    if _rdb is None:
        return {"error": "redis not connected"}
    return {
        "mode": _rdb.get("system:mode") or "milp",
        "drl_placement_ttl": _rdb.ttl("drl:placement"),
        "training_buffer_len": _rdb.llen("drl:training_buffer"),
        "storm_correction_buffer_key": _STORM_CORRECTION_BUFFER_KEY,
        "storm_correction_buffer_len": _rdb.llen(_STORM_CORRECTION_BUFFER_KEY),
        "model_contract": _runtime_state.get("model_contract"),
        "adapter_name": _runtime_state.get("adapter_name"),
        "inference_enabled": _runtime_state.get("inference_enabled"),
        "inference_error_count": _runtime_state.get("inference_error_count"),
    }


@app.get("/twin-stats")
def get_twin_stats() -> dict:
    """Return latest Digital Twin validation result from Redis."""
    if _rdb is None:
        return {"error": "redis not connected"}
    raw = _rdb.get("drl:twin_stats")
    if raw is None:
        return {"twin_stats": None, "message": "no drl:twin_stats in Redis yet"}
    return {"twin_stats": json.loads(raw)}


@app.get("/proposed-placement")
def get_proposed_placement() -> dict:
    """Return latest Tier-2 proposed placement payload from Redis."""
    if _rdb is None:
        return {"error": "redis not connected"}
    raw = _rdb.get("drl:placement:proposed")
    if raw is None:
        return {"proposed": None}
    return {"proposed": json.loads(raw)}


@app.get("/verification-status")
def get_verification_status() -> dict:
    """Return latest Tier-2 verification verdict from Redis."""
    if _rdb is None:
        return {"error": "redis not connected"}
    verified = _rdb.get("drl:placement:verified")
    revoked = _rdb.get("drl:placement:revoked")
    return {
        "verified": json.loads(verified) if verified else None,
        "revoked": json.loads(revoked) if revoked else None,
    }


@app.get("/leader")
def get_leader_status() -> dict:
    """Return current leader-election status for this pod."""
    if _elector is None:
        return {"leader": True, "mode": "single-pod", "epoch": 0}
    return {
        "leader": _elector.is_leader(),
        "epoch": _elector.current_epoch(),
        "identity": os.getenv("POD_NAME", "unknown"),
    }


# ── Scheduler-extender models ─────────────────────────────────────────────────
# Mirrors milp_agent's ExtenderArgs / HostPriority so drl-agent can serve
# as an independent Kubernetes scheduler extender.

class _ExtenderArgs(BaseModel):
    Pod: dict
    Nodes: Optional[dict] = None
    NodeNames: Optional[list[str]] = None


class _HostPriority(BaseModel):
    Host: str
    Score: int


# Grace period (seconds) before considering MILP heartbeat stale.
# milp-agent writes milp:heartbeat every CONTROL_INTERVAL (30 s) with
# TTL = CONTROL_INTERVAL × 3 (90 s).  Use the same margin so we only
# activate DRL scores when MILP has genuinely stopped writing.
MILP_HEARTBEAT_GRACE_S = float(os.getenv("MILP_HEARTBEAT_GRACE_S", "90"))


def _milp_is_alive() -> bool:
    """Return True if milp:heartbeat was written within MILP_HEARTBEAT_GRACE_S."""
    if _rdb is None:
        return False
    try:
        raw = _rdb.get("milp:heartbeat")
        if not raw:
            return False
        # Heartbeat exists — also check wall-clock age in case TTL is wrong
        from datetime import datetime, timezone
        s = raw.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return age <= MILP_HEARTBEAT_GRACE_S
    except Exception:
        # Key exists but unparseable timestamp — treat as alive to be safe
        return True


def _drl_extract_node_names(args: _ExtenderArgs) -> list[str]:
    node_names = args.NodeNames or []
    if args.Nodes and isinstance(args.Nodes, dict):
        node_names = [
            n.get("metadata", {}).get("name", "")
            for n in args.Nodes.get("items", [])
            if n.get("metadata", {}).get("name")
        ]
    return node_names


@app.post("/filter")
def drl_filter_nodes(args: _ExtenderArgs) -> dict:
    """Pass-through filter — DRL extender does not exclude any candidate nodes.

    Filtering decisions (node feasibility) remain with milp-agent and the
    built-in kube-scheduler plugins.  DRL only influences node *scoring*.
    """
    node_names = _drl_extract_node_names(args)
    return {
        "Nodes": {"items": [{"metadata": {"name": n}} for n in node_names]},
        "FailedNodes": {},
        "Error": "",
    }


@app.post("/prioritize")
def drl_prioritize_nodes(args: _ExtenderArgs) -> list[_HostPriority]:
    """Return DRL-based node scores for the scheduler extender.

    Failover design:
    • When milp:heartbeat is fresh (MILP alive): return Score=50 for every
      node so DRL does not interfere with MILP's own scoring.
    • When milp:heartbeat is absent/stale (MILP dead): return node scores
      derived from drl:placement so the scheduler still places pods on the
      DRL-preferred nodes even without MILP.

    Weight in extender-config is 100 (same as MILP). The combined math:
      MILP alive  → MILP preferred=100 + DRL neutral=50 = 150 vs 100 ✓
      MILP dead   → MILP ignorable → DRL preferred=100 vs others=50   ✓
    """
    node_names = _drl_extract_node_names(args)

    if _milp_is_alive():
        # MILP is healthy — return neutral scores to avoid double-counting
        log.debug("drl /prioritize: MILP heartbeat alive — returning neutral scores")
        return [_HostPriority(Host=n, Score=50) for n in node_names]

    # MILP is down — activate DRL scores as primary scheduler signal
    log.info("drl /prioritize: MILP heartbeat stale/absent — activating DRL node scores")
    scores: dict[str, int] = {}
    if _rdb is not None:
        try:
            raw = _rdb.get("drl:placement")
            if raw:
                payload = json.loads(raw)
                scores = payload.get("node_scores", {})
        except Exception as exc:
            log.warning("drl /prioritize: failed to read drl:placement: %s", exc)

    out: list[_HostPriority] = []
    for node_name in node_names:
        score = max(0, min(100, int(scores.get(node_name, 50))))
        out.append(_HostPriority(Host=node_name, Score=score))
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
