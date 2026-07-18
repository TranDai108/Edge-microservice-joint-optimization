"""Traffic scenario generator for EdgeEnv training augmentation.

Provides diverse starting-state perturbations so the PPO policy learns to
handle real-world conditions beyond the static mock baseline.

Scenarios (13 total)
────────────────────
Original 5 — perturb load without targeting n3's dominance:
- nominal          : balanced load, latency ~150 ms
- traffic_burst    : high CPU on AI nodes, e2e_latency near SLA limit
- node_failure     : one node fully saturated (cpu_util ≈ 1.0)
- thermal_throttle : energy cost dims spike (node gets hot)  [fixed: scale reaches >41 W/core]
- load_imbalance   : one node heavily loaded, others nearly idle

Adversarial 5 — each breaks n3's dominance via a different axis:
- n3_saturated     : n3 CPU 93-98% → MILP must distribute across n0/n2
- energy_inversion : n0 on green power (3 W/core), n3 on peak diesel (50+ W/core)
- memory_pressure  : n3 RAM 95% used → memory-heavy services forced to n2
- pipeline_split   : n2+n3 both 70-80% loaded → n0 absorbs lightweight services
- cascade_failure  : n1 AND n3 saturated → only n0+n2 available

Usage in train_simulator.py (--use-scenarios)
─────────────────────────────────────────────
    gen = ScenarioGenerator(seed=42)
    state = gen.sample_state()          # random scenario each call
    env._state = state
    obs, _ = env.reset()
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np

log = logging.getLogger("scenario-gen")

# ── State vector layout (must mirror edge_env.OBS_DIM = 44) ──────────────────
OBS_DIM: int = 44
#  [0:4]   cpu_util per node
#  [4:8]   mem_gb used per node
#  [8:12]  e_cpu_unit W/core per node
#  [12:15] detection variant one-hot
#  [15:39] placement one-hot (6 svcs × 4 nodes)
#  [39]    e2e_latency_ms
#  [40]    migrations_last_cycle
#  [41:44] w_c, w_d, w_a

# Real cluster constants from metrics_collector.py
_NODE_E_CPU = [19.87, 41.02, 19.81, 9.99]   # W/core
_NODE_CAP_CPU = [4.0, 2.0, 4.0, 4.0]
_NODE_CAP_MEM = [3.82, 3.82, 7.75, 7.75]


class ScenarioType(str, Enum):
    NOMINAL = "nominal"
    TRAFFIC_BURST = "traffic_burst"
    NODE_FAILURE = "node_failure"
    THERMAL_THROTTLE = "thermal_throttle"
    LOAD_IMBALANCE = "load_imbalance"
    # ── Adversarial: break n3 dominance ──────────────────────────────────────
    N3_SATURATED = "n3_saturated"
    ENERGY_INVERSION = "energy_inversion"
    MEMORY_PRESSURE = "memory_pressure"
    PIPELINE_SPLIT = "pipeline_split"
    CASCADE_FAILURE = "cascade_failure"
    # ── Hard objective scenarios ──────────────────────────────────────────────
    ENERGY_SAVING = "energy_saving"     # extreme w_c weight, cheap-vs-expensive nodes
    STORM_TEST = "storm_test"           # high disruption churn, strong w_d signal
    IDLE_CLUSTER = "idle_cluster"       # zero utilization, high w_a — matches live KWOK state


class ScenarioGenerator:
    """Generates diverse 44-dim state vectors for EdgeEnv.reset() augmentation.

    Initialise once and call sample_state() in each episode reset to inject
    a different traffic scenario into the environment.
    """

    # Sampling weights for random scenario selection (sum to 1)
    # Original 5 share 0.36; adversarial 5 share 0.40; hard-objective 3 share 0.24.
    _SCENARIO_WEIGHTS: dict[ScenarioType, float] = {
        ScenarioType.NOMINAL:          0.08,
        ScenarioType.TRAFFIC_BURST:    0.08,
        ScenarioType.NODE_FAILURE:     0.07,
        ScenarioType.THERMAL_THROTTLE: 0.06,
        ScenarioType.LOAD_IMBALANCE:   0.07,
        # adversarial — each breaks n3 via a different mechanism
        ScenarioType.N3_SATURATED:     0.10,
        ScenarioType.ENERGY_INVERSION: 0.09,
        ScenarioType.MEMORY_PRESSURE:  0.08,
        ScenarioType.PIPELINE_SPLIT:   0.06,
        ScenarioType.CASCADE_FAILURE:  0.07,
        # hard-objective — extreme weight settings DRL historically struggles with
        ScenarioType.ENERGY_SAVING:    0.08,
        ScenarioType.STORM_TEST:       0.06,
        ScenarioType.IDLE_CLUSTER:     0.10,  # zero-util live pattern, high priority
    }

    # Scenarios used per curriculum stage (stages 1-4 in ascending difficulty)
    _CURRICULUM_STAGES: dict[int, list["ScenarioType"]] = {
        1: [
            ScenarioType.NOMINAL, ScenarioType.TRAFFIC_BURST, ScenarioType.LOAD_IMBALANCE,
        ],
        2: [
            ScenarioType.NOMINAL, ScenarioType.TRAFFIC_BURST, ScenarioType.LOAD_IMBALANCE,
            ScenarioType.MEMORY_PRESSURE, ScenarioType.N3_SATURATED, ScenarioType.THERMAL_THROTTLE,
            ScenarioType.NODE_FAILURE,
        ],
        3: [
            ScenarioType.NOMINAL, ScenarioType.TRAFFIC_BURST, ScenarioType.LOAD_IMBALANCE,
            ScenarioType.MEMORY_PRESSURE, ScenarioType.N3_SATURATED, ScenarioType.THERMAL_THROTTLE,
            ScenarioType.NODE_FAILURE, ScenarioType.ENERGY_INVERSION,
            ScenarioType.CASCADE_FAILURE, ScenarioType.STORM_TEST, ScenarioType.PIPELINE_SPLIT,
        ],
        4: [],  # all scenarios, hard ones at 65% (handled in _update_curriculum_weights)
    }

    # Objective weight ranges per profile for randomization during training
    _OBJECTIVE_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
        "quality":  {"w_c": (0.05, 0.12), "w_d": (0.04, 0.12), "w_a": (0.80, 0.93)},
        "energy":   {"w_c": (0.40, 0.70), "w_d": (0.05, 0.15), "w_a": (0.20, 0.50)},
        "storm":    {"w_c": (0.08, 0.20), "w_d": (0.28, 0.50), "w_a": (0.35, 0.60)},
        "balanced": {"w_c": (0.10, 0.25), "w_d": (0.05, 0.20), "w_a": (0.55, 0.80)},
        "idle":     {"w_c": (0.08, 0.20), "w_d": (0.04, 0.14), "w_a": (0.65, 0.82)},
    }

    # Scenarios considered "hard" for curriculum stage 3+
    _HARD_SCENARIOS: frozenset["ScenarioType"] = frozenset({
        ScenarioType.ENERGY_INVERSION, ScenarioType.CASCADE_FAILURE,
        ScenarioType.ENERGY_SAVING, ScenarioType.STORM_TEST,
        ScenarioType.MEMORY_PRESSURE, ScenarioType.N3_SATURATED,
        ScenarioType.IDLE_CLUSTER,
    })

    def __init__(
        self,
        seed: Optional[int] = None,
        curriculum_stage: Optional[int] = None,
        objective_profile: Optional[str] = None,
        fixed_scenario: Optional[str] = None,
        storm_boost: Optional[float] = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.curriculum_stage = curriculum_stage
        self.objective_profile = objective_profile
        self._fixed_scenario: Optional[ScenarioType] = (
            ScenarioType(fixed_scenario) if fixed_scenario else None
        )
        self._scenario_types = list(self._SCENARIO_WEIGHTS.keys())
        self._weights = [self._SCENARIO_WEIGHTS[s] for s in self._scenario_types]
        if curriculum_stage is not None:
            self._update_curriculum_weights(curriculum_stage)
        # Optionally boost storm_test sampling weight and redistribute the excess
        if storm_boost is not None and storm_boost > 0.0:
            idx = self._scenario_types.index(ScenarioType.STORM_TEST)
            old_w = self._weights[idx]
            boost = min(float(storm_boost), 0.50)  # cap at 50%
            delta = boost - old_w
            if delta > 0:
                # Reduce other scenarios proportionally
                others = [i for i in range(len(self._weights)) if i != idx]
                total_others = sum(self._weights[i] for i in others)
                if total_others > 1e-9:
                    for i in others:
                        self._weights[i] -= delta * (self._weights[i] / total_others)
                self._weights[idx] = boost
            total = sum(self._weights)
            self._weights = [w / total for w in self._weights]
            log.info("storm_boost applied: storm_test weight %.2f → %.2f",
                     old_w, self._weights[idx])

    # ── Public API ────────────────────────────────────────────────────────────

    def _update_curriculum_weights(self, stage: int) -> None:
        """Recompute _scenario_types and _weights for the given curriculum stage."""
        if stage in (1, 2, 3):
            allowed = self._CURRICULUM_STAGES[stage]
            hard = [s for s in allowed if s in self._HARD_SCENARIOS]
            easy = [s for s in allowed if s not in self._HARD_SCENARIOS]
            if stage <= 2:
                # Uniform weights — no hard-scenario boosting yet
                self._scenario_types = allowed
                w = 1.0 / len(allowed)
                self._weights = [w] * len(allowed)
            else:
                # Stage 3: hard 50%, easy 50%
                self._scenario_types = allowed
                hard_w = 0.50 / max(len(hard), 1)
                easy_w = 0.50 / max(len(easy), 1)
                wmap = {s: (hard_w if s in self._HARD_SCENARIOS else easy_w) for s in allowed}
                self._weights = [wmap[s] for s in allowed]
        else:
            # Stage 4: all scenarios, hard at 65%
            all_types = list(self._SCENARIO_WEIGHTS.keys())
            hard = [s for s in all_types if s in self._HARD_SCENARIOS]
            easy = [s for s in all_types if s not in self._HARD_SCENARIOS]
            self._scenario_types = all_types
            hard_w = 0.65 / max(len(hard), 1)
            easy_w = 0.35 / max(len(easy), 1)
            wmap = {s: (hard_w if s in self._HARD_SCENARIOS else easy_w) for s in all_types}
            self._weights = [wmap[s] for s in all_types]
        # Re-normalise to guard against floating-point drift
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

    def _apply_objective_profile(self, state: np.ndarray, profile: str) -> np.ndarray:
        """Randomize objective weights [41:44] according to the named profile."""
        if profile not in self._OBJECTIVE_PROFILES:
            return state
        ranges = self._OBJECTIVE_PROFILES[profile]
        w_c = float(self._rng.uniform(*ranges["w_c"]))
        w_d = float(self._rng.uniform(*ranges["w_d"]))
        w_a = float(self._rng.uniform(*ranges["w_a"]))
        total = w_c + w_d + w_a
        state = state.copy()
        state[41] = w_c / total
        state[42] = w_d / total
        state[43] = w_a / total
        return state

    def sample_state(
        self,
        scenario: Optional[ScenarioType] = None,
    ) -> np.ndarray:
        """Return a 44-dim state vector for the given (or random) scenario.

        Args:
            scenario: Force a specific scenario type.  If None, samples
                      randomly according to _SCENARIO_WEIGHTS (or the active
                      curriculum stage weights if curriculum_stage was set at init).

        Returns:
            Float32 array of shape (44,) ready to assign to env._state.
        """
        if scenario is None:
            if self._fixed_scenario is not None:
                scenario = self._fixed_scenario
            else:
                idx = int(self._rng.choice(len(self._scenario_types), p=self._weights))
                scenario = self._scenario_types[idx]

        builders = {
            ScenarioType.NOMINAL:          self._nominal,
            ScenarioType.TRAFFIC_BURST:    self._traffic_burst,
            ScenarioType.NODE_FAILURE:     self._node_failure,
            ScenarioType.THERMAL_THROTTLE: self._thermal_throttle,
            ScenarioType.LOAD_IMBALANCE:   self._load_imbalance,
            ScenarioType.N3_SATURATED:     self._n3_saturated,
            ScenarioType.ENERGY_INVERSION: self._energy_inversion,
            ScenarioType.MEMORY_PRESSURE:  self._memory_pressure,
            ScenarioType.PIPELINE_SPLIT:   self._pipeline_split,
            ScenarioType.CASCADE_FAILURE:  self._cascade_failure,
            ScenarioType.ENERGY_SAVING:    self._energy_saving,
            ScenarioType.STORM_TEST:       self._storm_test,
            ScenarioType.IDLE_CLUSTER:     self._idle_cluster,
        }
        state = builders[scenario]()  # type: ignore[index]
        assert state.shape == (OBS_DIM,), f"scenario returned shape {state.shape}"
        # Apply objective profile randomization if configured at init time
        if self.objective_profile is not None:
            state = self._apply_objective_profile(state, self.objective_profile)
        log.debug("ScenarioGenerator: sampled scenario=%s profile=%s", scenario, self.objective_profile)
        return state

    # ── Scenario builders ─────────────────────────────────────────────────────

    def _base_state(
        self,
        cpu_util: list[float],
        mem_frac: list[float],
        e2e_ms: float,
        migrations: float = 0.0,
        e_cpu_scale: Optional[list[float]] = None,
    ) -> np.ndarray:
        """Assemble a state vector from high-level scenario parameters."""
        state = np.zeros(OBS_DIM, dtype=np.float32)

        # [0:4] cpu_util (ratio [0, 1])
        state[0:4] = np.clip(cpu_util, 0.0, 1.0)

        # [4:8] mem used in GB
        for i in range(4):
            state[4 + i] = max(0.0, mem_frac[i] * _NODE_CAP_MEM[i])

        # [8:12] e_cpu_unit W/core (optionally scaled for thermal throttle)
        scales = e_cpu_scale or [1.0] * 4
        for i in range(4):
            state[8 + i] = float(_NODE_E_CPU[i]) * scales[i]

        # [12:15] detection variant one-hot: default yolo26-nano active
        state[12] = 1.0

        # [15:39] placement one-hot: default all services on n0 (node 0)
        # (can be overridden per-scenario after calling _base_state)
        for svc_idx in range(6):
            state[15 + svc_idx * 4 + 0] = 1.0  # n0

        # [39] e2e latency ms
        state[39] = float(e2e_ms)

        # [40] migrations last cycle
        state[40] = float(migrations)

        # [41:44] weights — defaults match config.py
        state[41] = 0.15   # w_c
        state[42] = 0.10   # w_d
        state[43] = 0.75   # w_a

        return state

    def _add_jitter(self, state: np.ndarray, std: float = 0.02) -> np.ndarray:
        """Add small Gaussian noise to make states non-deterministic."""
        state = state.copy()
        noise = self._rng.normal(0.0, std, size=OBS_DIM).astype(np.float32)
        state[0:4] = np.clip(state[0:4] + noise[0:4], 0.0, 1.0)
        state[4:8] = np.clip(state[4:8] + noise[4:8] * 0.5, 0.0, None)
        state[39] = max(0.0, state[39] + float(self._rng.normal(0.0, 10.0)))
        return state

    def _nominal(self) -> np.ndarray:
        """Balanced load — typical daytime operation."""
        cpu = [0.30, 0.45, 0.25, 0.20]
        mem = [0.40, 0.55, 0.20, 0.15]
        state = self._base_state(cpu, mem, e2e_ms=150.0, migrations=0.0)
        return self._add_jitter(state)

    def _traffic_burst(self) -> np.ndarray:
        """Sudden traffic spike: AI nodes saturated, latency approaching SLA."""
        burst_intensity = float(self._rng.uniform(0.7, 0.95))
        cpu = [0.60, burst_intensity, 0.55, burst_intensity * 0.9]
        mem = [0.60, 0.80, 0.50, 0.70]
        e2e = float(self._rng.uniform(350.0, 490.0))  # near SLA=500ms
        state = self._base_state(cpu, mem, e2e_ms=e2e, migrations=2.0)
        return self._add_jitter(state, std=0.03)

    def _node_failure(self) -> np.ndarray:
        """One node fully saturated (simulates hardware fault / eviction cascade)."""
        failed_node = int(self._rng.integers(0, 4))
        cpu = [0.25, 0.35, 0.20, 0.30]
        cpu[failed_node] = float(self._rng.uniform(0.92, 1.0))  # saturated
        mem = [0.30, 0.40, 0.25, 0.35]
        mem[failed_node] = float(self._rng.uniform(0.85, 0.98))
        state = self._base_state(cpu, mem, e2e_ms=220.0, migrations=3.0)
        return self._add_jitter(state)

    def _thermal_throttle(self) -> np.ndarray:
        """Node overheating: energy cost spikes, placement decisions change.

        BUG FIX: original scale (1.5–2.5×) kept n3 (base 9.99 W/core) below n1
        (41.02 W/core), so MILP never avoided the hot node.  New scale (4–6×)
        guarantees the hot node exceeds ALL other nodes' energy cost.
        """
        hot_node = int(self._rng.integers(0, 4))
        cpu = [0.35, 0.40, 0.30, 0.25]
        mem = [0.35, 0.45, 0.30, 0.25]
        e_cpu_scale = [1.0, 1.0, 1.0, 1.0]
        # Scale must make hot_node cost > max(_NODE_E_CPU) = 41.02 regardless of base.
        # Worst case: n3 base = 9.99 → need scale > 41.02/9.99 ≈ 4.1; use 4.5–6.0.
        e_cpu_scale[hot_node] = float(self._rng.uniform(4.5, 6.0))
        state = self._base_state(cpu, mem, e2e_ms=180.0, e_cpu_scale=e_cpu_scale)
        return self._add_jitter(state)

    def _load_imbalance(self) -> np.ndarray:
        """Heavy node vs. idle nodes — tests DRL load-balancing capability."""
        heavy_node = int(self._rng.integers(0, 4))
        cpu = [0.10, 0.10, 0.10, 0.10]
        mem = [0.10, 0.10, 0.10, 0.10]
        cpu[heavy_node] = float(self._rng.uniform(0.75, 0.90))
        mem[heavy_node] = float(self._rng.uniform(0.65, 0.85))
        state = self._base_state(cpu, mem, e2e_ms=160.0, migrations=1.0)
        return self._add_jitter(state)

    # ── Adversarial scenarios — each removes n3 as the trivially optimal node ─

    def _n3_saturated(self) -> np.ndarray:
        """n3 CPU nearly full → MILP cannot fit services there; forces n0/n2.

        With n3 at 93-98% CPU, available_cpu[3] = 4 × (1-0.95) = 0.20 vCPU —
        not enough for even a single service.  The MILP optimal placement shifts
        to n2 (same RAM, 4 vCPU free) and n0 (4 vCPU, less RAM).
        """
        sat = float(self._rng.uniform(0.93, 0.98))
        cpu = [0.20, 0.35, 0.22, sat]
        mem = [0.30, 0.45, 0.25, float(self._rng.uniform(0.90, 0.97))]
        state = self._base_state(cpu, mem, e2e_ms=210.0, migrations=2.0)
        return self._add_jitter(state)

    def _energy_inversion(self) -> np.ndarray:
        """n0 on green/renewable power; n3 on peak-rate diesel.

        e_cpu[n0] ≈ 3 W/core  (solar surplus, scale ≈ 0.15)
        e_cpu[n3] ≈ 50 W/core (diesel peak, scale ≈ 5.0)

        With w_c=0.20, the energy term favours n0 for lightweight services.
        The MILP will co-locate energy-tolerant heavy services on n2 and route
        lightweight services (m0, m1, m5) to cheap n0.
        """
        cpu = [0.15, 0.30, 0.20, 0.20]
        mem = [0.20, 0.40, 0.18, 0.22]
        e_cpu_scale = [
            float(self._rng.uniform(0.12, 0.18)),   # n0: ~2–3.6 W/core (green)
            1.0,                                      # n1: nominal
            float(self._rng.uniform(0.8, 1.2)),      # n2: near nominal
            float(self._rng.uniform(4.5, 5.5)),      # n3: ~45–55 W/core (diesel)
        ]
        state = self._base_state(cpu, mem, e2e_ms=145.0, e_cpu_scale=e_cpu_scale)
        # Boost w_c weight so energy term has visible impact on MILP decision.
        state[41] = float(self._rng.uniform(0.25, 0.40))   # w_c elevated
        state[43] = float(self._rng.uniform(0.45, 0.60))   # w_a reduced
        return self._add_jitter(state)

    def _memory_pressure(self) -> np.ndarray:
        """n3 RAM 94-97% used; n2 has free RAM — heavy services must go to n2.

        n3 mem_available ≈ 7.75 × 0.04 = 0.31 GB → only tiny services fit.
        n2 mem_available ≈ 7.75 × 0.85 = 6.59 GB → absorbs heavy services.
        MILP optimal: gen-ai + detection on n2, lightweight pipeline on n0/n1.
        """
        mem_n3 = float(self._rng.uniform(0.94, 0.97))
        cpu = [0.25, 0.40, 0.18, 0.55]
        mem = [0.30, 0.55, float(self._rng.uniform(0.10, 0.20)), mem_n3]
        state = self._base_state(cpu, mem, e2e_ms=195.0, migrations=1.5)
        return self._add_jitter(state)

    def _pipeline_split(self) -> np.ndarray:
        """Both high-capacity nodes (n2, n3) are 70-82% loaded.

        Neither n2 nor n3 can absorb all 6 services.  The MILP must split the
        pipeline: lightweight services (m0, m1, m5) land on n0; compute-heavy
        services (m2, m3, m4) split across n2 and n3.  Demonstrates the MILP's
        bin-packing capability and the DRL agent's generalisation to split
        placement decisions.
        """
        cpu_n2 = float(self._rng.uniform(0.70, 0.82))
        cpu_n3 = float(self._rng.uniform(0.70, 0.82))
        cpu = [0.12, 0.45, cpu_n2, cpu_n3]
        mem = [
            0.18,
            0.55,
            float(self._rng.uniform(0.55, 0.70)),
            float(self._rng.uniform(0.55, 0.70)),
        ]
        state = self._base_state(cpu, mem, e2e_ms=230.0, migrations=3.0)
        return self._add_jitter(state)

    def _energy_saving(self) -> np.ndarray:
        """Very high energy-cost scenario with strong w_c objective signal.

        n0 on cheap renewable (scale ≈ 0.12–0.18), n3 on expensive diesel (scale ≈ 4–6).
        w_c is pushed to 0.45–0.65 so the optimal policy aggressively avoids n3.
        DRL historically struggles here because training rarely sees w_c > 0.30.
        """
        cpu = [0.18, 0.30, 0.22, 0.28]
        mem = [0.22, 0.40, 0.20, 0.28]
        e_cpu_scale = [
            float(self._rng.uniform(0.12, 0.18)),   # n0: ~2–3.6 W/core (green)
            float(self._rng.uniform(1.0, 1.5)),      # n1: moderate
            float(self._rng.uniform(0.8, 1.2)),      # n2: near nominal
            float(self._rng.uniform(4.0, 6.0)),      # n3: ~40–60 W/core (diesel)
        ]
        state = self._base_state(cpu, mem, e2e_ms=148.0, e_cpu_scale=e_cpu_scale)
        w_c = float(self._rng.uniform(0.45, 0.65))
        w_d = float(self._rng.uniform(0.05, 0.12))
        w_a = max(0.05, 1.0 - w_c - w_d)
        state[41] = w_c
        state[42] = w_d
        state[43] = w_a
        return self._add_jitter(state)

    def _storm_test(self) -> np.ndarray:
        """High-disruption, high-mobility scenario with strong w_d objective signal.

        All nodes are moderately-to-heavily loaded with jitter, and recent
        migrations are elevated to 5–8.  w_d is pushed to 0.30–0.50 to test
        whether DRL avoids unnecessary migrations under disruption cost pressure.

        Placement is randomised across realistic starting positions so the DRL
        must learn to *condition on w_d*: when services are already on good nodes
        (n3/n2) and w_d is high, the optimal policy is to stay put rather than
        migrate for a marginal energy gain.
        """
        cpu = [
            float(self._rng.uniform(0.45, 0.75)),
            float(self._rng.uniform(0.60, 0.88)),
            float(self._rng.uniform(0.38, 0.68)),
            float(self._rng.uniform(0.52, 0.82)),
        ]
        mem = [
            float(self._rng.uniform(0.40, 0.65)),
            float(self._rng.uniform(0.55, 0.82)),
            float(self._rng.uniform(0.32, 0.58)),
            float(self._rng.uniform(0.48, 0.78)),
        ]
        migrations = float(self._rng.uniform(5.0, 8.0))
        state = self._base_state(cpu, mem, e2e_ms=310.0, migrations=migrations)
        w_d = float(self._rng.uniform(0.30, 0.50))
        w_c = float(self._rng.uniform(0.08, 0.20))
        w_a = max(0.10, 1.0 - w_d - w_c)
        state[41] = w_c
        state[42] = w_d
        state[43] = w_a

        # ── Randomise starting placement so DRL sees "already-good" configs ──
        # Without this, training always starts on n0 and DRL never encounters
        # the case where services are already on the energy-efficient n3/n2.
        # Distribution: 35% on n3 (optimal), 25% on n2, 25% on n0, 15% mixed.
        state[15:39] = 0.0
        r = self._rng.random()
        if r < 0.35:
            for svc_idx in range(6):
                state[15 + svc_idx * 4 + 3] = 1.0   # n3: cheapest
        elif r < 0.60:
            for svc_idx in range(6):
                state[15 + svc_idx * 4 + 2] = 1.0   # n2
        elif r < 0.85:
            for svc_idx in range(6):
                state[15 + svc_idx * 4 + 0] = 1.0   # n0: default
        else:
            # Mixed: each service independently on n0/n2/n3
            for svc_idx in range(6):
                node = int(self._rng.choice([0, 2, 3]))
                state[15 + svc_idx * 4 + node] = 1.0

        return self._add_jitter(state, std=0.04)

    def _idle_cluster(self) -> np.ndarray:
        """KWOK-like idle cluster: near-zero utilization, high accuracy weight.

        Matches the live cluster state: CPU/mem ≈ 0 on all nodes, n1 energy
        naturally 2× others (41 W/core vs 20 W/core, no scaling needed), and
        w_a = 0.65–0.82 (accuracy-first optimisation, the default config).
        DRL historically fails here because all training scenarios have
        non-zero utilisation, so the policy has never seen a fully idle cluster.
        """
        cpu = [float(self._rng.uniform(0.0, 0.03)) for _ in range(4)]
        mem = [float(self._rng.uniform(0.0, 0.02)) for _ in range(4)]
        # Default energy (no scale): n1=41 W/core is naturally 2× others
        state = self._base_state(cpu, mem, e2e_ms=0.0, migrations=0.0)

        # ── Randomise placement one-hot [15:39] across realistic starting nodes ─
        # n3 (edge-nodes-4, 10W/core) and n2 (edge-nodes-3, 19.8W/core) are the
        # cheapest nodes and where MILP converges for idle workloads.  The DRL
        # must learn the correct steady-state policy from ANY starting node, not
        # just n0.  Without this, DRL always starts from n0 at training time and
        # never encounters the live-system state (services already on n2/n3).
        #
        # Distribution: 40% already-optimal (n3), 30% second-best (n2),
        #               20% default n0, 10% mixed per-service.
        state[15:39] = 0.0  # clear default n0 placement
        r = self._rng.random()
        if r < 0.40:
            # Already on n3 — teach DRL to stay
            for svc_idx in range(6):
                state[15 + svc_idx * 4 + 3] = 1.0   # n3
        elif r < 0.70:
            # On n2 — teach DRL to recognise cheap node
            for svc_idx in range(6):
                state[15 + svc_idx * 4 + 2] = 1.0   # n2
        elif r < 0.90:
            # Default n0 — teach DRL to migrate away
            for svc_idx in range(6):
                state[15 + svc_idx * 4 + 0] = 1.0   # n0
        else:
            # Mixed: each service independently on n0/n2/n3
            for svc_idx in range(6):
                node = self._rng.choice([0, 2, 3])
                state[15 + svc_idx * 4 + node] = 1.0

        # ── Randomise detection variant one-hot [12:15] ──────────────────────
        # With w_a dominant (0.65-0.82), DRL should prefer medium/small over
        # nano.  Always training with nano [12]=1 prevents learning this.
        state[12:15] = 0.0
        w_a_sample = float(self._rng.uniform(0.65, 0.82))
        if w_a_sample > 0.75:
            # High accuracy weight: bias toward medium variant
            det_var = int(self._rng.choice([0, 1, 2], p=[0.2, 0.4, 0.4]))
        else:
            det_var = int(self._rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
        state[12 + det_var] = 1.0

        # ── Randomise objective weights around live defaults (w_a dominant) ──
        w_a = w_a_sample
        w_c = float(self._rng.uniform(0.08, 0.20))
        w_d = max(0.03, 1.0 - w_a - w_c)
        state[41] = w_c
        state[42] = w_d
        state[43] = w_a
        return self._add_jitter(state, std=0.01)  # tiny jitter for near-zero values

    def _cascade_failure(self) -> np.ndarray:
        """n1 AND n3 both saturated — only n0 + n2 are viable.

        Simulates a partial rack failure or power-zone outage affecting two nodes.
        With n0 (4 vCPU, 3.82 GB) and n2 (4 vCPU, 7.75 GB) as the only options,
        the MILP must pack all 6 services into 8 vCPU / 11.57 GB total.
        Proves MILP correctness under hard constraints AND tests DRL's ability to
        recognise and exploit the constrained topology.
        """
        cpu = [
            float(self._rng.uniform(0.20, 0.35)),   # n0: available
            float(self._rng.uniform(0.94, 0.99)),   # n1: saturated
            float(self._rng.uniform(0.15, 0.30)),   # n2: available
            float(self._rng.uniform(0.93, 0.99)),   # n3: saturated
        ]
        mem = [
            float(self._rng.uniform(0.25, 0.45)),
            float(self._rng.uniform(0.88, 0.97)),
            float(self._rng.uniform(0.20, 0.35)),
            float(self._rng.uniform(0.88, 0.97)),
        ]
        state = self._base_state(cpu, mem, e2e_ms=280.0, migrations=4.0)
        return self._add_jitter(state)
