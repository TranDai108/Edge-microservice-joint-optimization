"""Gymnasium environment for edge placement control.

Phase 2:
- D2.1: mock mode with 44-dim obs, MultiDiscrete action space.
- D2.2: reward mirrors MILP objective J.
- D2.3: used for MO-PPO simulator training (50 k steps).
- D2.4: live Redis mode — _build_live_state() reads milp:confirmed_placement
         and milp:placement and builds the identical 44-dim state vector
         that milp_agent._build_state_vector() produces.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import gymnasium as gym
import numpy as np
import redis

try:
    from src.config import MILP_W_A, MILP_W_C, MILP_W_D
    from src.variant_catalog import (
        DETECTION_CPU_SCALE,
        DETECTION_MEM_SCALE,
        DETECTION_OFFLINE_ACCURACY,
        DETECTION_VARIANTS,
        GEN_AI_CPU_SCALE,
        GEN_AI_MEM_SCALE,
        GEN_AI_OFFLINE_QUALITY,
        GEN_AI_VARIANTS,
    )
    from src.drl.reward import compute_reward_for_placement
except ImportError:
    from config import MILP_W_A, MILP_W_C, MILP_W_D
    from variant_catalog import (
        DETECTION_CPU_SCALE,
        DETECTION_MEM_SCALE,
        DETECTION_OFFLINE_ACCURACY,
        DETECTION_VARIANTS,
        GEN_AI_CPU_SCALE,
        GEN_AI_MEM_SCALE,
        GEN_AI_OFFLINE_QUALITY,
        GEN_AI_VARIANTS,
    )
    from drl.reward import compute_reward_for_placement

try:
    from src.solver.dataset_generator import MILPDataset, NodeSpec, ServiceSpec
except ImportError:
    from solver.dataset_generator import MILPDataset, NodeSpec, ServiceSpec

try:
    from src.drl.scenario_generator import ScenarioGenerator
except ImportError:
    try:
        from drl.scenario_generator import ScenarioGenerator
    except ImportError:
        ScenarioGenerator = None  # type: ignore[assignment,misc]


OBS_DIM = 44
ACTION_DIMS = [4, 4, 4, 12, 12, 4]
SERVICE_IDS = ["m0", "m1", "m2", "m3", "m4", "m5"]
NODE_IDS = ["n0", "n1", "n2", "n3"]

# Episode length (number of control steps per episode).
# Set to 1 to align training with evaluation: each episode is a single
# placement decision, and x_prev always comes from the scenario's state[15:39]
# rather than the agent's own prior action.  This matches the live deployment
# model (one placement decision per 30-s control cycle).
DEFAULT_MAX_STEPS = 1

# SLA latency threshold (ms). It is a guardrail, not the primary objective.
SLA_LATENCY_MS = float(os.getenv("SLA_LATENCY_MS", "1500.0"))
# Mock e2e latency used in simulator mode (no real Prometheus data)
_MOCK_E2E_LATENCY_MS = 150.0

# ── Action masking: resource requirements per variant ───────────────────────
# CPU in cores and memory in GB — derived from k8s/manifests/ resource requests.
# Used by action_masks() to prevent infeasible placements (pod would stay Pending).
_VARIANT_CPU_REQ: dict[str, float] = {
    "standard":         0.20,   # 200m  (api-gateway, ingest, preprocess, postprocess)
    "yolo26-nano":      0.50,   # 500m  (DETECTION_CPU_SCALE ×0.5)
    "yolo26-small":     1.00,   # 1000m (DETECTION_CPU_SCALE ×1.0)
    "yolo26-medium":    2.00,   # 2000m (DETECTION_CPU_SCALE ×2.0)
    "qwen-1.5b-nano":   0.30,   # 300m  (gen-ai base request)
    "llama-3b-small":   0.60,   # 600m  (GEN_AI_CPU_SCALE ×2.0)
    "gemma2-2b-medium": 0.40,   # 400m  (GEN_AI_CPU_SCALE ×1.33)
}
_VARIANT_MEM_REQ: dict[str, float] = {
    "standard":         0.25,   # ~256Mi
    "yolo26-nano":      0.78,   # ~800Mi (DETECTION_MEM_SCALE ×0.7)
    "yolo26-small":     1.10,   # ~1124Mi (DETECTION_MEM_SCALE ×1.0)
    "yolo26-medium":    1.65,   # ~1690Mi (DETECTION_MEM_SCALE ×1.5)
    "qwen-1.5b-nano":   0.50,   # ~512Mi
    "llama-3b-small":   1.00,   # ~1024Mi (GEN_AI_MEM_SCALE ×2.0)
    "gemma2-2b-medium": 0.70,   # ~716Mi  (GEN_AI_MEM_SCALE ×1.4)
}
# Node capacities — must mirror NODE_CAP_CPU / NODE_CAP_MEM in _build_live_state
_NODE_CAP_CPU: list[float] = [4.0, 2.0, 4.0, 4.0]    # allocatable cores
_NODE_CAP_MEM: list[float] = [3.82, 3.82, 7.75, 7.75]  # GB

# Penalty applied to reward for each node whose capacity is exceeded
INFEASIBILITY_PENALTY: float = -1.0
STORM_EXCESS_PENALTY: float = float(os.getenv("DRL_STORM_EXCESS_PENALTY", "15.0"))


def build_mock_dataset() -> MILPDataset:
    """Create a deterministic mock MILP dataset for simulator training."""
    nodes = [
        NodeSpec(node_id="n0", cap_cpu=4.0, cap_mem_gb=3.82, energy_cost=19.87, e_mem_unit=4.90),
        NodeSpec(node_id="n1", cap_cpu=2.0, cap_mem_gb=3.82, energy_cost=41.02, e_mem_unit=4.90),
        NodeSpec(node_id="n2", cap_cpu=4.0, cap_mem_gb=7.75, energy_cost=19.81, e_mem_unit=1.20),
        NodeSpec(node_id="n3", cap_cpu=4.0, cap_mem_gb=7.75, energy_cost=9.99, e_mem_unit=1.20),
    ]

    services = [
        ServiceSpec(service_id="m0", service_type="gateway", migration_cost=1.0, valid_variants=["standard"]),
        ServiceSpec(service_id="m1", service_type="ingest", migration_cost=1.0, valid_variants=["standard"]),
        ServiceSpec(service_id="m2", service_type="preprocess", migration_cost=1.0, valid_variants=["standard"]),
        ServiceSpec(service_id="m3", service_type="detection", migration_cost=3.0, valid_variants=list(DETECTION_VARIANTS)),
        ServiceSpec(service_id="m4", service_type="gen_ai", migration_cost=2.0, valid_variants=list(GEN_AI_VARIANTS)),
        ServiceSpec(service_id="m5", service_type="postprocess", migration_cost=1.0, valid_variants=["standard"]),
    ]

    r_req: dict[tuple[str, str], float] = {}
    r_mem: dict[tuple[str, str], float] = {}
    acc: dict[tuple[str, str], float] = {}

    standard_services = ["m0", "m1", "m2", "m5"]
    for svc in standard_services:
        r_req[(svc, "standard")] = _VARIANT_CPU_REQ["standard"]
        r_mem[(svc, "standard")] = _VARIANT_MEM_REQ["standard"]
        acc[(svc, "standard")] = 1.0

    for var in DETECTION_VARIANTS:
        r_req[("m3", var)] = _VARIANT_CPU_REQ[var]
        r_mem[("m3", var)] = _VARIANT_MEM_REQ[var]
        acc[("m3", var)] = float(DETECTION_OFFLINE_ACCURACY[var])

    for var in GEN_AI_VARIANTS:
        r_req[("m4", var)] = _VARIANT_CPU_REQ[var]
        r_mem[("m4", var)] = _VARIANT_MEM_REQ[var]
        acc[("m4", var)] = float(GEN_AI_OFFLINE_QUALITY[var])

    x_prev: dict[tuple[str, str, str], float] = {}
    base = {
        "m0": ("standard", "n0"),
        "m1": ("standard", "n0"),
        "m2": ("standard", "n0"),
        "m3": (DETECTION_VARIANTS[0], "n0"),
        "m4": (GEN_AI_VARIANTS[0], "n0"),
        "m5": ("standard", "n0"),
    }
    for svc in services:
        for var in svc.valid_variants:
            for node in NODE_IDS:
                x_prev[(svc.service_id, var, node)] = 0.0
    for svc, (var, node) in base.items():
        x_prev[(svc, var, node)] = 1.0

    return MILPDataset(
        nodes=nodes,
        services=services,
        r_req=r_req,
        r_mem=r_mem,
        acc=acc,
        x_prev=x_prev,
        theta_max=1.3,
        v_storm_max=2,
        w_c=float(MILP_W_C),
        w_d=float(MILP_W_D),
        w_a=float(MILP_W_A),
    )


def compute_action_masks(
    state: np.ndarray,
    last_placement: dict | None = None,
) -> np.ndarray:
    """Compute a flat boolean action-validity mask from a raw state vector.

    This is the module-level version of ``EdgeEnv.action_masks()``.  It can be
    called directly (e.g. during offline inference in ``demo_scenarios``) without
    requiring a live ``EdgeEnv`` instance.

    Parameters
    ----------
    state:
        44-dim observation vector (same layout as ``EdgeEnv`` observations).
    last_placement:
        Optional current placement dict ``{svc_id: (variant, node_id)}``.
        When provided, resources of services staying on the same node are
        credited back to that node's headroom to avoid masking larger variants.
    """
    cpu_util = state[0:4].tolist()
    mem_used = state[4:8].tolist()

    headroom_cpu = [
        max(0.0, _NODE_CAP_CPU[i] * (1.0 - float(cpu_util[i])))
        for i in range(4)
    ]
    headroom_mem = [
        max(0.0, _NODE_CAP_MEM[i] - float(mem_used[i]))
        for i in range(4)
    ]

    cur_node_per_svc: list[int | None] = []
    for svc_idx in range(len(SERVICE_IDS)):
        slot = 15 + svc_idx * 4
        ps = state[slot : slot + 4]
        cur_node_per_svc.append(
            int(np.argmax(ps)) if float(ps.max()) > 0.0 else None
        )

    cur_var_per_svc: dict[str, str | None] = {
        "m0": "standard", "m1": "standard",
        "m2": "standard", "m5": "standard",
    }
    det_slice = state[12:15]
    det_vi = int(np.argmax(det_slice)) if float(det_slice.max()) > 0.0 else 0
    cur_var_per_svc["m3"] = list(DETECTION_VARIANTS)[det_vi]
    cur_var_per_svc["m4"] = (
        last_placement["m4"][0] if last_placement is not None else None
    )

    mask = np.ones(sum(ACTION_DIMS), dtype=bool)
    offset = 0

    for svc_idx, (svc_id, head_size) in enumerate(zip(SERVICE_IDS, ACTION_DIMS)):
        svc_cur_node = cur_node_per_svc[svc_idx]
        svc_cur_var = cur_var_per_svc.get(svc_id)
        if svc_id in ("m0", "m1", "m2", "m5"):
            req_cpu = _VARIANT_CPU_REQ["standard"]
            req_mem = _VARIANT_MEM_REQ["standard"]
            for node_idx in range(4):
                eff_cpu = headroom_cpu[node_idx]
                eff_mem = headroom_mem[node_idx]
                if svc_cur_node == node_idx and svc_cur_var is not None:
                    eff_cpu += _VARIANT_CPU_REQ.get(svc_cur_var, 0.0)
                    eff_mem += _VARIANT_MEM_REQ.get(svc_cur_var, 0.0)
                if eff_cpu < req_cpu or eff_mem < req_mem:
                    mask[offset + node_idx] = False
        elif svc_id == "m3":
            for vi, var in enumerate(DETECTION_VARIANTS):
                req_cpu = _VARIANT_CPU_REQ[var]
                req_mem = _VARIANT_MEM_REQ[var]
                for node_idx in range(4):
                    action_idx = vi * len(NODE_IDS) + node_idx
                    eff_cpu = headroom_cpu[node_idx]
                    eff_mem = headroom_mem[node_idx]
                    if svc_cur_node == node_idx and svc_cur_var is not None:
                        eff_cpu += _VARIANT_CPU_REQ.get(svc_cur_var, 0.0)
                        eff_mem += _VARIANT_MEM_REQ.get(svc_cur_var, 0.0)
                    if eff_cpu < req_cpu or eff_mem < req_mem:
                        mask[offset + action_idx] = False
        elif svc_id == "m4":
            for vi, var in enumerate(GEN_AI_VARIANTS):
                req_cpu = _VARIANT_CPU_REQ[var]
                req_mem = _VARIANT_MEM_REQ[var]
                for node_idx in range(4):
                    action_idx = vi * len(NODE_IDS) + node_idx
                    eff_cpu = headroom_cpu[node_idx]
                    eff_mem = headroom_mem[node_idx]
                    if svc_cur_node == node_idx and svc_cur_var is not None:
                        eff_cpu += _VARIANT_CPU_REQ.get(svc_cur_var, 0.0)
                        eff_mem += _VARIANT_MEM_REQ.get(svc_cur_var, 0.0)
                    if eff_cpu < req_cpu or eff_mem < req_mem:
                        mask[offset + action_idx] = False
        offset += head_size

    offset = 0
    for head_size in ACTION_DIMS:
        head_slice = mask[offset:offset + head_size]
        if not head_slice.any():
            head_slice[0] = True
        offset += head_size

    return mask


class EdgeEnv(gym.Env[np.ndarray, np.ndarray]):
    """Edge control environment for DRL experiments.

    Observation:
        44-dim continuous vector.
    Action:
        MultiDiscrete([4, 4, 4, 12, 12, 4]).
    Action masking:
        ``action_masks()`` returns a flat boolean array compatible with
        sb3-contrib MaskablePPO / ActionMasker wrapper.
        ``_compute_infeasibility_penalty()`` provides a hard reward penalty
        for standard PPO when the proposed placement overloads a node.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mock: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        max_steps: int = DEFAULT_MAX_STEPS,
        scenario_generator=None,
    ) -> None:
        super().__init__()
        self.mock = mock
        self.max_steps = max_steps
        self._step_count: int = 0
        self._rng = np.random.default_rng()
        self._state = np.zeros(OBS_DIM, dtype=np.float32)
        self._mock_dataset = build_mock_dataset()
        self._scenario_generator = scenario_generator
        self._last_placement: dict[str, tuple[str, str]] | None = None

        self.action_space = gym.spaces.MultiDiscrete(ACTION_DIMS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        self.rdb: redis.Redis | None = None
        if not mock:
            self.rdb = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._last_placement = None
        self._state = self._build_state()
        return self._state.copy(), {"mock": self.mock}

    def action_masks(self) -> np.ndarray:
        """Return a flat boolean validity mask — delegates to module-level helper."""
        return compute_action_masks(self._state, self._last_placement)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (len(ACTION_DIMS),):
            raise ValueError(
                f"Expected action shape {(len(ACTION_DIMS),)}, got {action.shape}."
            )

        reward = self._compute_reward(action)
        self._last_placement = self._action_to_placement(action)
        self._step_count += 1
        self._state = self._build_state()

        terminated = False
        # Truncate after max_steps — allows Monitor/SB3 to log episode rewards
        truncated = self._step_count >= self.max_steps
        info: dict[str, Any] = {"mock": self.mock}
        return self._state.copy(), reward, terminated, truncated, info

    def _build_state(self) -> np.ndarray:
        if self.mock:
            return self._build_mock_state()
        return self._build_live_state()

    def _build_mock_state(self) -> np.ndarray:
        if self._scenario_generator is not None:
            # Use structured scenario for training augmentation
            state = self._scenario_generator.sample_state()
            return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        # Keep mock observations finite and bounded to realistic magnitudes.
        state = self._rng.normal(loc=0.0, scale=0.5, size=(OBS_DIM,)).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_live_state(self) -> np.ndarray:
        """Build the 44-dim state vector from live Redis data.

        Mirrors ``milp_agent._build_state_vector()`` exactly.
        Layout (44 floats total):
          [0:4]   node CPU utilisation ratio  (4 nodes)
          [4:8]   node memory usage in GB     (4 nodes)
          [8:12]  node CPU energy unit W/core (4 nodes)
          [12:15] detection variant one-hot   (3 variants)
          [15:39] placement one-hot 6 svcs×4 nodes (24)
          [39]    e2e latency ms
          [40]    migrations last cycle (from placement payload)
          [41:44] w_c, w_d, w_a

        All Redis / JSON failures fall back to finite zeros so the env
        never raises (D2.4 acceptance: no crash when Redis is live).
        """
        zeros = np.zeros(OBS_DIM, dtype=np.float32)
        if self.rdb is None:
            return zeros

        # ── helpers ───────────────────────────────────────────────────────────
        # Static node order: edge-nodes-1 → n0, …, edge-nodes-4 → n3
        NODE_HOSTNAMES = [
            "edge-nodes-1", "edge-nodes-2", "edge-nodes-3", "edge-nodes-4",
        ]
        # Real cluster profiles from metrics_collector.py constants
        # (read-only reference — we never import metrics_collector at runtime
        #  to keep the DRL package self-contained)
        NODE_E_CPU = [19.87, 41.02, 19.81, 9.99]   # W/core  (approx linear)
        NODE_CAP_CPU = [4.0, 2.0, 4.0, 4.0]        # allocatable cores
        NODE_CAP_MEM = [3.82, 3.82, 7.75, 7.75]  # GB

        DET_VARIANTS = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
        SVC_DEPLOY_NAMES = {
            "m0": "api-gateway",
            "m1": "ingest",
            "m2": "preprocess",
            "m3": "detection",
            "m4": "gen-ai",
            "m5": "postprocess",
        }

        try:
            # ── 1. Read Redis keys ────────────────────────────────────────────
            raw_confirmed = self.rdb.get("milp:confirmed_placement")
            raw_placement = self.rdb.get("milp:placement")
            raw_drl = self.rdb.get("drl:placement")

            confirmed: dict = json.loads(raw_confirmed) if raw_confirmed else {}
            placement: dict = json.loads(raw_placement) if raw_placement else {}
            # When DRL has an active placement use it as the x_prev reference so
            # migration counting is relative to DRL's own continuity, not MILP's.
            if raw_drl:
                try:
                    _drl_pl = json.loads(raw_drl).get("placement", {})
                    if _drl_pl:
                        confirmed = _drl_pl
                except Exception:
                    pass

            # ── 2. Node CPU utilisation [0:4] ────────────────────────────────
            # milp:placement carries per-node resource_usage when present
            resource_usage: dict = placement.get("resource_usage", {})
            cpu_util = [0.0] * 4
            for i, host in enumerate(NODE_HOSTNAMES):
                node_id = f"n{i}"
                used = float(resource_usage.get(node_id, 0.0))
                cap = NODE_CAP_CPU[i] if NODE_CAP_CPU[i] > 0 else 1.0
                cpu_util[i] = max(0.0, min(1.0, used / cap))

            # ── 2b. Saturate offline nodes from milp:node_down ───────────────
            # MILP watchdog writes milp:node_down (TTL 60s) listing failed node
            # hostnames. Without this, an offline node shows cpu_util=0.0 and
            # looks maximally free — causing DRL to keep placing there.
            raw_down = self.rdb.get("milp:node_down")
            if raw_down:
                for down_host in raw_down.split(","):
                    down_host = down_host.strip()
                    if down_host in NODE_HOSTNAMES:
                        i = NODE_HOSTNAMES.index(down_host)
                        cpu_util[i] = 1.0            # fully saturated → headroom = 0
                        # mem_gb handled below after mem_usage is read; store index
                        # We mark it here; mem override applied after mem_gb is built.

            # ── 3. Node memory usage in GB [4:8] ─────────────────────────────
            mem_usage: dict = placement.get("mem_usage", {})
            mem_gb = [0.0] * 4
            for i in range(4):
                node_id = f"n{i}"
                mem_gb[i] = float(mem_usage.get(node_id, 0.0))

            # Apply mem cap for offline nodes (mirrors cpu_util saturation above)
            if raw_down:
                for down_host in raw_down.split(","):
                    down_host = down_host.strip()
                    if down_host in NODE_HOSTNAMES:
                        i = NODE_HOSTNAMES.index(down_host)
                        mem_gb[i] = NODE_CAP_MEM[i]  # fully occupied → headroom = 0

            # ── 4. Node CPU energy unit [8:12] ───────────────────────────────
            # Prefer live Kepler W/core from milp:placement["e_cpu_unit"],
            # fall back to calibrated hardware constants if key is absent.
            e_cpu_raw = placement.get("e_cpu_unit", [])
            if (
                isinstance(e_cpu_raw, list)
                and len(e_cpu_raw) == 4
                and all(isinstance(v, (int, float)) and v > 0 for v in e_cpu_raw)
            ):
                e_cpu_unit = [max(0.1, float(v)) for v in e_cpu_raw]
            else:
                e_cpu_unit = list(NODE_E_CPU)  # fallback: static calibration constants

            # ── 5. Detection variant one-hot [12:15] ─────────────────────────
            det_onehot = [0.0, 0.0, 0.0]
            det_variant = "yolo26-nano"  # default
            # Try confirmed placement first ("detection" service)
            if "detection" in confirmed:
                det_variant = confirmed["detection"].get("variant", det_variant)
            elif placement.get("placement"):
                det_variant = placement["placement"].get(
                    "m3", [det_variant, ""])[0]
            if det_variant in DET_VARIANTS:
                det_onehot[DET_VARIANTS.index(det_variant)] = 1.0

            # ── 6. Placement one-hot 6 svcs × 4 nodes [15:39] ────────────────
            placement_onehot = [0.0] * 24
            for svc_idx, svc_id in enumerate(SERVICE_IDS):
                deploy_name = SVC_DEPLOY_NAMES[svc_id]
                host = ""
                if deploy_name in confirmed:
                    host = confirmed[deploy_name].get("node", "")
                slot = svc_idx * 4
                # Map hostname → node index
                node_idx = None
                if host in NODE_HOSTNAMES:
                    node_idx = NODE_HOSTNAMES.index(host)
                elif host.startswith("edge-nodes-"):
                    try:
                        idx = int(host.rsplit("-", 1)[-1]) - 1
                        if 0 <= idx < 4:
                            node_idx = idx
                    except ValueError:
                        pass
                if node_idx is not None:
                    placement_onehot[slot + node_idx] = 1.0

            # ── 7. e2e latency ms [39] ───────────────────────────────────────
            e2e_ms = float(placement.get("e2e_latency_ms", _MOCK_E2E_LATENCY_MS))
            if not math.isfinite(e2e_ms):
                e2e_ms = 0.0

            # ── 8. Migrations last cycle [40] ────────────────────────────────
            migration_types: dict = placement.get("migration_types", {})
            n_migrations = float(
                sum(1 for v in migration_types.values() if v != "Stayed")
            )

            # ── 9. MILP weights [41:44] ──────────────────────────────────────
            weights = [float(MILP_W_C), float(MILP_W_D), float(MILP_W_A)]

            # ── Assemble ──────────────────────────────────────────────────────
            vec = (
                cpu_util
                + mem_gb
                + e_cpu_unit
                + det_onehot
                + placement_onehot
                + [e2e_ms, n_migrations]
                + weights
            )
            # Pad / clip to exactly 44
            if len(vec) < OBS_DIM:
                vec.extend([0.0] * (OBS_DIM - len(vec)))
            vec = vec[:OBS_DIM]

            # Final NaN/Inf guard
            arr = np.array(vec, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr

        except redis.RedisError:
            # Redis unreachable — return zeros, do not crash (D2.4 requirement)
            return zeros
        except Exception:
            # Any other parsing failure — return zeros, do not crash
            return zeros

    def _build_reward_dataset(self) -> MILPDataset:
        """Build a scenario-aware MILPDataset from the current state vector.

        Extracts live energy costs (state[8:12]) and objective weights
        (state[41:44]) so that ``_compute_reward`` correctly penalises
        placements that are expensive *for the current scenario* (e.g.
        energy_inversion where n3 costs 50+ W/core).
        """
        e_cpu_unit = [max(0.5, float(self._state[8 + i])) for i in range(4)]
        w_c = max(1e-4, float(self._state[41]))
        w_d = max(1e-4, float(self._state[42]))
        w_a = max(1e-4, float(self._state[43]))
        updated_nodes = [
            NodeSpec(
                node_id=nd.node_id,
                cap_cpu=nd.cap_cpu,
                cap_mem_gb=nd.cap_mem_gb,
                energy_cost=e_cpu_unit[i],
                e_mem_unit=nd.e_mem_unit,
            )
            for i, nd in enumerate(self._mock_dataset.nodes)
        ]
        # ── Derive x_prev from state vector or last action ───────────────────
        # _mock_dataset.x_prev is always n0 (build_mock_dataset default).
        # In scenario training the scenario generator randomises state[15:39]
        # (placement one-hot) and state[12:15] (detection variant), so x_prev
        # must be extracted from the *current* state to give a correct
        # migration-cost signal.  After the first step we prefer _last_placement
        # which is exact; on reset (first step) we fall back to the state vector.
        if self._last_placement is not None:
            x_prev: dict[tuple[str, str, str], float] = {
                k: 0.0 for k in self._mock_dataset.x_prev
            }
            for svc_id, (var, node_id) in self._last_placement.items():
                x_prev[(svc_id, var, node_id)] = 1.0
        else:
            # Decode from state[15:39] (placement one-hot) + state[12:15] (det variant)
            det_vars = list(DETECTION_VARIANTS)
            gen_vars = list(GEN_AI_VARIANTS)
            det_var_idx = int(np.argmax(self._state[12:15])) if self._state[12:15].sum() > 0 else 0
            x_prev = {k: 0.0 for k in self._mock_dataset.x_prev}
            for svc_idx, svc_id in enumerate(SERVICE_IDS):
                slot = self._state[15 + svc_idx * 4: 15 + svc_idx * 4 + 4]
                node_idx = int(np.argmax(slot)) if slot.sum() > 0 else 0
                node_id  = NODE_IDS[node_idx]
                if svc_id == "m3":
                    var = det_vars[det_var_idx]
                elif svc_id == "m4":
                    var = gen_vars[0]   # gen-ai variant not in state; use lightest
                else:
                    var = "standard"
                x_prev[(svc_id, var, node_id)] = 1.0

        return MILPDataset(
            nodes=updated_nodes,
            services=self._mock_dataset.services,
            r_req=self._mock_dataset.r_req,
            r_mem=self._mock_dataset.r_mem,
            acc=self._mock_dataset.acc,
            x_prev=x_prev,
            theta_max=self._mock_dataset.theta_max,
            v_storm_max=self._mock_dataset.v_storm_max,
            w_c=w_c,
            w_d=w_d,
            w_a=w_a,
        )

    def _compute_infeasibility_penalty(self, action: np.ndarray) -> float:
        """Return a hard negative penalty when the proposed placement overloads a node.

        With MaskablePPO the agent never selects infeasible actions, so this
        penalty is effectively always 0.0 during normal training and inference.
        It is kept as a safety net for non-masked evaluation paths.
        """
        placement = self._action_to_placement(action)

        node_cpu_load: dict[str, float] = {n: 0.0 for n in NODE_IDS}
        node_mem_load: dict[str, float] = {n: 0.0 for n in NODE_IDS}
        for _svc, (variant, node_id) in placement.items():
            node_cpu_load[node_id] += _VARIANT_CPU_REQ.get(variant, 0.0)
            node_mem_load[node_id] += _VARIANT_MEM_REQ.get(variant, 0.0)

        penalty = 0.0
        for i, node_id in enumerate(NODE_IDS):
            if node_cpu_load[node_id] > _NODE_CAP_CPU[i]:
                penalty += INFEASIBILITY_PENALTY
            if node_mem_load[node_id] > _NODE_CAP_MEM[i]:
                penalty += INFEASIBILITY_PENALTY
        return penalty

    def _compute_storm_penalty(
        self,
        placement: dict[str, tuple[str, str]],
        reward_ds: MILPDataset,
    ) -> float:
        """Penalise placements that exceed v_storm_max migrations."""
        migration_count = sum(
            1 for svc, (var, node) in placement.items()
            if reward_ds.x_prev.get((svc, var, node), 0.0) < 0.5
        )
        if migration_count > reward_ds.v_storm_max:
            excess = migration_count - reward_ds.v_storm_max
            return -float(STORM_EXCESS_PENALTY) * float(excess)
        return 0.0

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward = -J + infeasibility penalty + storm penalty.

        Uses a scenario-aware dataset (``_build_reward_dataset``) so that the
        energy cost term in J reflects the *current* node power prices from the
        state vector rather than the fixed calibration constants.  This ensures
        the agent learns to avoid expensive nodes (e.g. n3 at 54 W/core in
        energy_inversion) and feasibility is checked against available headroom.
        """
        placement = self._action_to_placement(action)
        reward_ds = self._build_reward_dataset()
        base_reward = compute_reward_for_placement(reward_ds, placement)
        infeasibility = self._compute_infeasibility_penalty(action)
        storm_penalty = self._compute_storm_penalty(placement, reward_ds)
        return base_reward + infeasibility + storm_penalty

    @staticmethod
    def _decode_variant_node(value: int, variants: list[str]) -> tuple[str, str]:
        variant_idx = int(value) // len(NODE_IDS)
        node_idx = int(value) % len(NODE_IDS)
        variant_idx = min(max(variant_idx, 0), len(variants) - 1)
        node_idx = min(max(node_idx, 0), len(NODE_IDS) - 1)
        return variants[variant_idx], NODE_IDS[node_idx]

    def _action_to_placement(self, action: np.ndarray) -> dict[str, tuple[str, str]]:
        a = np.asarray(action, dtype=np.int64).tolist()
        return {
            "m0": ("standard", NODE_IDS[int(a[0])]),
            "m1": ("standard", NODE_IDS[int(a[1])]),
            "m2": ("standard", NODE_IDS[int(a[2])]),
            "m3": self._decode_variant_node(int(a[3]), list(DETECTION_VARIANTS)),
            "m4": self._decode_variant_node(int(a[4]), list(GEN_AI_VARIANTS)),
            "m5": ("standard", NODE_IDS[int(a[5])]),
        }
