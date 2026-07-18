"""Digital Twin validator for safe DRL action commitment.

Tier 1 of the Digital Twin stack: pure Python, no new infrastructure.

Architecture
────────────
1. Take the current 44-dim cluster state from Redis (built by EdgeEnv._build_live_state).
2. Run N_ROLLOUTS Monte-Carlo simulations with Gaussian perturbations on:
     • CPU utilisation   [0:4]   – simulates load-spike uncertainty
     • Memory usage      [4:8]   – simulates memory-pressure uncertainty
     • e2e latency ms    [39]    – simulates network-jitter uncertainty
3. For each perturbed state check:
     a) Feasibility: variant resource requirements must fit within available headroom.
     b) SLA safety: predicted latency must stay under SLA_LATENCY_MS.
4. Compute mean predicted reward across rollouts (mirrors MILP objective J).
5. Report is_safe when:
     • feasible_fraction >= min_feasible_rate  (default 0.80)
     • mean_reward >= safety_threshold         (default -0.50)

Thesis claim
────────────
"DRL actions are pre-validated in N=20 Monte Carlo simulation rollouts via a
Digital Twin synchronised with real cluster state.  Only actions that are
feasible in ≥80 % of perturbed scenarios are committed to the live cluster."
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("digital-twin")

# ── Default hyper-parameters ─────────────────────────────────────────────────
DEFAULT_N_ROLLOUTS: int = 20
DEFAULT_SAFETY_THRESHOLD: float = -0.50   # mean reward must exceed this
DEFAULT_MIN_FEASIBLE_RATE: float = 0.80   # ≥80 % of rollouts must be feasible
DEFAULT_PERTURBATION_STD: float = 0.05    # σ for CPU/mem noise (fraction of capacity)
DEFAULT_LATENCY_SCALE: float = 50.0       # scale of exponential latency noise (ms)

# ── Constants mirrored from edge_env.py (kept local to avoid circular import) ─
_SLA_LATENCY_MS: float = float(os.getenv("SLA_LATENCY_MS", "1500.0"))
_NODE_IDS: list[str] = ["n0", "n1", "n2", "n3"]
_V_STORM_MAX: int = 2    # must mirror build_mock_dataset() v_storm_max

# Mapping from live cluster names → mock dataset IDs
_SVC_NAME_TO_ID: dict[str, str] = {
    "api-gateway": "m0", "ingest": "m1", "preprocess": "m2",
    "detection": "m3", "gen-ai": "m4", "postprocess": "m5",
}
_NODE_NAME_TO_ID: dict[str, str] = {
    "edge-nodes-1": "n0", "edge-nodes-2": "n1",
    "edge-nodes-3": "n2", "edge-nodes-4": "n3",
}
_NODE_CAP_CPU: list[float] = [4.0, 2.0, 4.0, 4.0]       # allocatable cores
_NODE_CAP_MEM: list[float] = [3.82, 3.82, 7.75, 7.75]  # GB
_DETECTION_VARIANTS: list[str] = ["yolo26-nano", "yolo26-small", "yolo26-medium"]
_GEN_AI_VARIANTS: list[str] = ["qwen-1.5b-nano", "llama-3b-small", "gemma2-2b-medium"]
_VARIANT_CPU_REQ: dict[str, float] = {
    "standard":         0.20,
    "yolo26-nano":      0.50,
    "yolo26-small":     1.00,
    "yolo26-medium":    2.00,
    "qwen-1.5b-nano":   0.30,
    "llama-3b-small":   0.60,
    "gemma2-2b-medium": 0.40,
}
_VARIANT_MEM_REQ: dict[str, float] = {
    "standard":         0.25,
    "yolo26-nano":      0.78,
    "yolo26-small":     1.10,
    "yolo26-medium":    1.65,
    "qwen-1.5b-nano":   0.50,
    "llama-3b-small":   1.00,
    "gemma2-2b-medium": 0.70,
}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of a Digital Twin validation run."""
    is_safe: bool
    mean_reward: float
    predicted_j: float
    objective_j_raw: float
    objective_j_penalized: float
    mean_raw_reward: float
    mean_storm_penalty: float
    mean_infeas_penalty: float
    std_reward: float
    min_reward: float
    n_rollouts: int
    n_feasible: int
    feasible_rate: float
    sla_safe_rate: float
    safety_threshold: float
    min_feasible_rate: float
    action: list[int] = field(default_factory=list)
    storm_ok: bool = True
    migration_count: int = 0
    v_storm_max: int = _V_STORM_MAX
    reason: str = "accepted"

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "mean_reward": round(self.mean_reward, 4),
            "predicted_j": round(self.predicted_j, 4),
            "objective_j_raw": round(self.objective_j_raw, 4),
            "objective_j_penalized": round(self.objective_j_penalized, 4),
            "mean_raw_reward": round(self.mean_raw_reward, 4),
            "mean_storm_penalty": round(self.mean_storm_penalty, 4),
            "mean_infeas_penalty": round(self.mean_infeas_penalty, 4),
            "std_reward": round(self.std_reward, 4),
            "min_reward": round(self.min_reward, 4),
            "n_rollouts": self.n_rollouts,
            "n_feasible": self.n_feasible,
            "feasible_rate": round(self.feasible_rate, 3),
            "sla_safe_rate": round(self.sla_safe_rate, 3),
            "safety_threshold": self.safety_threshold,
            "min_feasible_rate": self.min_feasible_rate,
            "action": self.action,
            "storm_ok": self.storm_ok,
            "migration_count": self.migration_count,
            "v_storm_max": self.v_storm_max,
            "reason": self.reason,
        }


# ── Placement decoder (mirrors EdgeEnv._action_to_placement) ─────────────────

def _count_migrations(
    action: np.ndarray,
    confirmed_placement: dict | None,
) -> int:
    """Count how many services would migrate relative to confirmed_placement."""
    if not confirmed_placement:
        return 0
    proposed = _decode_action(action)  # {svc_id: (variant, node_id)}
    _svc_id_to_name = {v: k for k, v in _SVC_NAME_TO_ID.items()}
    count = 0
    for svc_id, (new_var, new_node) in proposed.items():
        svc_name = _svc_id_to_name.get(svc_id)
        if not svc_name:
            continue
        prev = confirmed_placement.get(svc_name, {})
        if not prev:
            count += 1
            continue
        prev_node_id = _NODE_NAME_TO_ID.get(prev.get("node", ""), "")
        prev_var = prev.get("variant", "standard")
        if prev_node_id != new_node or prev_var != new_var:
            count += 1
    return count


def _confirmed_to_action_space(
    confirmed_placement: dict | None,
) -> dict[str, tuple[str, str]] | None:
    """Convert milp:confirmed_placement format to EdgeEnv action placement format."""
    if not confirmed_placement:
        return None
    placement: dict[str, tuple[str, str]] = {}
    for svc_name, info in confirmed_placement.items():
        svc_id = _SVC_NAME_TO_ID.get(svc_name)
        node_id = _NODE_NAME_TO_ID.get(info.get("node", ""))
        variant = info.get("variant", "standard")
        if svc_id and node_id:
            placement[svc_id] = (variant, node_id)
    return placement or None


def _decode_action(action: np.ndarray) -> dict[str, tuple[str, str]]:
    a = np.asarray(action, dtype=np.int64).tolist()

    def _vn(value: int, variants: list[str]) -> tuple[str, str]:
        n = len(_NODE_IDS)
        vi = min(max(int(value) // n, 0), len(variants) - 1)
        ni = min(max(int(value) % n, 0), n - 1)
        return variants[vi], _NODE_IDS[ni]

    return {
        "m0": ("standard", _NODE_IDS[min(int(a[0]), 3)]),
        "m1": ("standard", _NODE_IDS[min(int(a[1]), 3)]),
        "m2": ("standard", _NODE_IDS[min(int(a[2]), 3)]),
        "m3": _vn(int(a[3]), _DETECTION_VARIANTS),
        "m4": _vn(int(a[4]), _GEN_AI_VARIANTS),
        "m5": ("standard", _NODE_IDS[min(int(a[5]), 3)]),
    }


# ── Feasibility check against a given state ──────────────────────────────────

def _check_feasibility(
    action: np.ndarray,
    state: np.ndarray,
) -> tuple[bool, int]:
    """Check if action fits within AVAILABLE node headroom in the given state.

    Uses current CPU utilisation [0:4] and memory usage [4:8] from state to
    compute remaining headroom per node.  When a service stays on the SAME
    node, only the net resource delta (new − old) is charged against that
    node's headroom, because the old variant's resources are freed first.

    Returns:
        (is_feasible, n_violations) — n_violations==0 means fully feasible.
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

    # ── Extract current placement from state[15:39] ───────────────────────
    _SVC_IDS = ["m0", "m1", "m2", "m3", "m4", "m5"]
    cur_node_for_svc: dict[str, int | None] = {}
    for svc_idx, svc_id in enumerate(_SVC_IDS):
        slot = 15 + svc_idx * 4
        ps = state[slot : slot + 4]
        cur_node_for_svc[svc_id] = (
            int(np.argmax(ps)) if float(ps.max()) > 0.0 else None
        )

    cur_var_for_svc: dict[str, str | None] = {
        svc: "standard" for svc in ("m0", "m1", "m2", "m5")
    }
    det_slice = state[12:15]
    det_vi = int(np.argmax(det_slice)) if float(det_slice.max()) > 0.0 else 0
    cur_var_for_svc["m3"] = _DETECTION_VARIANTS[det_vi]
    cur_var_for_svc["m4"] = None  # gen-ai variant not recoverable from state alone

    # ── Compute net resource delta per node (fix double-counting) ─────────
    # Services staying on the same node only need the incremental resource
    # difference; services moving to a new node need their full requirement.
    placement = _decode_action(action)
    node_cpu_load: dict[str, float] = {n: 0.0 for n in _NODE_IDS}
    node_mem_load: dict[str, float] = {n: 0.0 for n in _NODE_IDS}
    for svc_id, (variant, node_id) in placement.items():
        new_cpu = _VARIANT_CPU_REQ.get(variant, 0.0)
        new_mem = _VARIANT_MEM_REQ.get(variant, 0.0)
        node_idx = _NODE_IDS.index(node_id)
        if cur_node_for_svc.get(svc_id) == node_idx:
            old_var = cur_var_for_svc.get(svc_id)
            if old_var is not None:
                new_cpu = max(0.0, new_cpu - _VARIANT_CPU_REQ.get(old_var, 0.0))
                new_mem = max(0.0, new_mem - _VARIANT_MEM_REQ.get(old_var, 0.0))
        node_cpu_load[node_id] += new_cpu
        node_mem_load[node_id] += new_mem

    n_violations = 0
    for i, node_id in enumerate(_NODE_IDS):
        if node_cpu_load[node_id] > headroom_cpu[i]:
            n_violations += 1
        if node_mem_load[node_id] > headroom_mem[i]:
            n_violations += 1

    return n_violations == 0, n_violations


# ── Main validator class ──────────────────────────────────────────────────────

class DigitalTwinValidator:
    """Monte Carlo safety validator for DRL placement actions.

    Initialise once at startup and call validate() before each live commit.

    Example usage in drl_agent.py::

        validator = DigitalTwinValidator(n_rollouts=20, safety_threshold=-0.5)
        state = _build_state_from_redis(rdb)
        action, _ = model.predict(state, deterministic=True)
        result = validator.validate(action, state)
        if result.is_safe:
            rdb.setex("drl:placement", 35, json.dumps(_action_to_placement(action)))
        else:
            log.warning("Twin rejected action, falling back to MILP")
        rdb.setex("drl:twin_stats", 35, json.dumps(result.to_dict()))
    """

    def __init__(
        self,
        n_rollouts: int = DEFAULT_N_ROLLOUTS,
        safety_threshold: float = DEFAULT_SAFETY_THRESHOLD,
        min_feasible_rate: float = DEFAULT_MIN_FEASIBLE_RATE,
        perturbation_std: float = DEFAULT_PERTURBATION_STD,
        latency_scale: float = DEFAULT_LATENCY_SCALE,
        seed: int | None = None,
    ) -> None:
        self.n_rollouts = n_rollouts
        self.safety_threshold = safety_threshold
        self.min_feasible_rate = min_feasible_rate
        self.perturbation_std = perturbation_std
        self.latency_scale = latency_scale
        self._rng = np.random.default_rng(seed)
        self._env_cls = None

    # ── Lazy EdgeEnv import ───────────────────────────────────────────────────

    def _get_env(self):
        """Return a fresh mock EdgeEnv instance (lazy import avoids torch overhead)."""
        if self._env_cls is None:
            try:
                from src.drl.edge_env import EdgeEnv  # noqa: PLC0415
            except ImportError:
                from drl.edge_env import EdgeEnv  # noqa: PLC0415
            self._env_cls = EdgeEnv
        return self._env_cls(mock=True)

    # ── State perturbation ────────────────────────────────────────────────────

    def _perturb_state(self, base_state: np.ndarray) -> np.ndarray:
        """Apply structured Gaussian/exponential noise to simulate uncertainty.

        - CPU utilisation [0:4]:  additive Gaussian noise, clipped to [0, 1]
        - Memory usage   [4:8]:   additive Gaussian noise, clipped to ≥ 0
        - e2e latency    [39]:    additive exponential noise (latency only spikes up)
        - Energy W/core  [8:12]:  multiplicative ±10 % Gaussian (thermal variance)
                                  Simulates Kepler measurement jitter and
                                  transient thermal throttling between scrapes.
        """
        state = base_state.copy()
        cpu_noise = self._rng.normal(0.0, self.perturbation_std, size=4).astype(np.float32)
        state[0:4] = np.clip(state[0:4] + cpu_noise, 0.0, 1.0)
        mem_noise = self._rng.normal(0.0, self.perturbation_std * 2.0, size=4).astype(np.float32)
        state[4:8] = np.clip(state[4:8] + mem_noise, 0.0, None)
        lat_noise = float(self._rng.exponential(scale=self.latency_scale))
        state[39] = float(state[39]) + lat_noise
        # ±10 % multiplicative noise on energy W/core to simulate thermal variance
        energy_noise = self._rng.normal(0.0, 0.10, size=4).astype(np.float32)
        state[8:12] = np.clip(state[8:12] * (1.0 + energy_noise), 0.1, None)
        return state

    # ── Single-rollout reward estimate ────────────────────────────────────────

    def _estimate_reward(
        self,
        action: np.ndarray,
        state: np.ndarray,
        confirmed_placement: dict | None = None,
    ) -> tuple[float, float, float, float, float, bool]:
        """Compute reward for (action, state) via a mock EdgeEnv step.

        Uses live Kepler W/core values from state[8:12] to update the
        mock dataset's energy_cost per node before computing reward, so
        the objective J reflects actual thermal conditions rather than
        hardcoded power-profile constants.

        If confirmed_placement is provided (from milp:confirmed_placement in
        Redis), x_prev in the mock dataset is updated to reflect the real
        current cluster placement.  This prevents spurious migration costs
        for services that are already on their target node.

        Returns:
            (penalized_reward, raw_objective_j, raw_reward, storm_penalty,
             infeas_penalty, sla_safe)
        """
        try:
            from src.drl.reward import evaluate_objective_for_placement  # noqa: PLC0415
        except ImportError:
            from drl.reward import evaluate_objective_for_placement  # noqa: PLC0415

        env = self._get_env()
        # Inject live energy costs from state[8:12] into the env's dataset
        # so _compute_reward() evaluates J with Kepler-derived W/core values.
        nodes_sorted = sorted(env._mock_dataset.nodes, key=lambda n: n.node_id)
        for i, node in enumerate(nodes_sorted[:4]):
            node.energy_cost = max(0.1, float(state[8 + i]))
        # Update x_prev to reflect real confirmed placement so disruption cost
        # is computed relative to what is actually deployed, not the default mock.
        if confirmed_placement:
            ds = env._mock_dataset
            # Zero out all x_prev entries first
            for key in list(ds.x_prev.keys()):
                ds.x_prev[key] = 0.0
            # Set x_prev=1 for each confirmed service placement
            for svc_name, info in confirmed_placement.items():
                svc_id = _SVC_NAME_TO_ID.get(svc_name)
                node_id = _NODE_NAME_TO_ID.get(info.get("node", ""))
                variant = info.get("variant", "standard")
                if svc_id and node_id:
                    ds.x_prev[(svc_id, variant, node_id)] = 1.0
        # Always score against confirmed live placement for fair migration cost.
        # This avoids artificially setting disruption/storm penalties to zero.
        env._last_placement = _confirmed_to_action_space(confirmed_placement)
        env._state = state.copy()
        env._step_count = 0
        placement = env._action_to_placement(action)
        reward_ds = env._build_reward_dataset()
        breakdown = evaluate_objective_for_placement(reward_ds, placement)
        raw_reward = float(breakdown.reward)
        storm_penalty = float(env._compute_storm_penalty(placement, reward_ds))
        infeas_penalty = float(env._compute_infeasibility_penalty(action))
        reward = raw_reward + storm_penalty + infeas_penalty
        sla_safe = float(state[39]) < _SLA_LATENCY_MS
        return (
            float(reward),
            float(breakdown.objective_j),
            raw_reward,
            storm_penalty,
            infeas_penalty,
            sla_safe,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        action: np.ndarray,
        base_state: np.ndarray,
        confirmed_placement: dict | None = None,
    ) -> ValidationResult:
        """Run N_ROLLOUTS Monte Carlo simulations and return a safety verdict.

        Args:
            action:               Proposed MultiDiscrete action [6].
            base_state:           Current 44-dim cluster state from Redis.
            confirmed_placement:  Dict from milp:confirmed_placement (optional).
                                  When provided, x_prev is initialised from real
                                  cluster state so disruption cost is accurate.

        Returns:
            ValidationResult — inspect .is_safe before committing to live cluster.
        """
        action = np.asarray(action, dtype=np.int64)
        base_state = np.asarray(base_state, dtype=np.float32)

        rewards: list[float] = []
        raw_objectives: list[float] = []
        raw_rewards: list[float] = []
        storm_penalties: list[float] = []
        infeas_penalties: list[float] = []
        n_feasible = 0
        n_sla_safe = 0

        for _ in range(self.n_rollouts):
            perturbed = self._perturb_state(base_state)
            is_feasible, _ = _check_feasibility(action, perturbed)
            (
                reward,
                raw_objective_j,
                raw_reward,
                storm_penalty,
                infeas_penalty,
                sla_safe,
            ) = self._estimate_reward(action, perturbed, confirmed_placement)

            rewards.append(reward)
            raw_objectives.append(raw_objective_j)
            raw_rewards.append(raw_reward)
            storm_penalties.append(storm_penalty)
            infeas_penalties.append(infeas_penalty)
            if is_feasible:
                n_feasible += 1
            if sla_safe:
                n_sla_safe += 1

        arr = np.array(rewards, dtype=np.float64)
        mean_reward = float(np.mean(arr))
        mean_raw_objective = float(np.mean(np.array(raw_objectives, dtype=np.float64)))
        mean_raw_reward = float(np.mean(np.array(raw_rewards, dtype=np.float64)))
        mean_storm_penalty = float(np.mean(np.array(storm_penalties, dtype=np.float64)))
        mean_infeas_penalty = float(np.mean(np.array(infeas_penalties, dtype=np.float64)))
        feasible_rate = n_feasible / self.n_rollouts
        migration_count = _count_migrations(action, confirmed_placement)
        storm_ok = migration_count <= _V_STORM_MAX
        checks: list[str] = []
        if mean_reward < self.safety_threshold:
            checks.append(
                f"predicted_reward_below_threshold({mean_reward:.3f}<{self.safety_threshold:.3f})"
            )
        if feasible_rate < self.min_feasible_rate:
            checks.append(
                f"feasible_rate_below_min({feasible_rate:.2f}<{self.min_feasible_rate:.2f})"
            )
        if not storm_ok:
            checks.append(
                f"storm_max_exceeded({migration_count}>{_V_STORM_MAX})"
            )
        is_safe = len(checks) == 0
        reason = "accepted" if is_safe else "; ".join(checks)

        result = ValidationResult(
            is_safe=is_safe,
            mean_reward=mean_reward,
            predicted_j=-mean_reward,
            objective_j_raw=mean_raw_objective,
            objective_j_penalized=-mean_reward,
            mean_raw_reward=mean_raw_reward,
            mean_storm_penalty=mean_storm_penalty,
            mean_infeas_penalty=mean_infeas_penalty,
            std_reward=float(np.std(arr)),
            min_reward=float(np.min(arr)),
            n_rollouts=self.n_rollouts,
            n_feasible=n_feasible,
            feasible_rate=feasible_rate,
            sla_safe_rate=n_sla_safe / self.n_rollouts,
            safety_threshold=self.safety_threshold,
            min_feasible_rate=self.min_feasible_rate,
            action=action.tolist(),
            storm_ok=storm_ok,
            migration_count=migration_count,
            v_storm_max=_V_STORM_MAX,
            reason=reason,
        )

        log.debug(
            "Twin validate: is_safe=%s mean_r=%.4f J_raw=%.4f J_pen=%.4f "
            "feasible=%.0f%% sla_ok=%.0f%% storm_ok=%s migrations=%d/%d reason=%s",
            is_safe, mean_reward, mean_raw_objective, -mean_reward,
            feasible_rate * 100, result.sla_safe_rate * 100,
            storm_ok, migration_count, _V_STORM_MAX, reason,
        )
        return result

    def find_best_action(
        self,
        candidates: list[np.ndarray],
        base_state: np.ndarray,
        confirmed_placement: dict | None = None,
    ) -> tuple[np.ndarray, ValidationResult]:
        """Evaluate multiple candidate actions, return the safest/best one.

        Prefers safe candidates by mean_reward.  Falls back to the least-bad
        candidate when none pass the safety gate.

        Args:
            candidates:           List of MultiDiscrete action arrays to evaluate.
            base_state:           Current 44-dim cluster state from Redis.
            confirmed_placement:  Forwarded to validate() for accurate disruption cost.

        Returns:
            (best_action, ValidationResult for best_action).
        """
        if not candidates:
            raise ValueError("candidates list must not be empty")

        results: list[tuple[np.ndarray, ValidationResult]] = [
            (c, self.validate(c, base_state, confirmed_placement)) for c in candidates
        ]

        safe = [(a, r) for a, r in results if r.is_safe]
        pool = safe if safe else results
        best_action, best_result = max(pool, key=lambda x: x[1].mean_reward)

        log.info(
            "find_best_action: %d candidates, %d safe — "
            "best J=%.4f feasible=%.0f%%",
            len(candidates), len(safe),
            best_result.predicted_j, best_result.feasible_rate * 100,
        )
        return best_action, best_result
