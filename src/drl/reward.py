"""MILP-aligned reward utilities for DRL.

This module mirrors the normalized objective in src/solver/milp_model.py:
J = w_c * C_norm + w_d * D_norm - w_a * A_norm
reward = -J
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from src.solver.dataset_generator import MILPDataset
except ImportError:
    from solver.dataset_generator import MILPDataset


PlacementDict = dict[str, tuple[str, str]]


@dataclass
class ObjectiveBreakdown:
    objective_j: float
    reward: float
    cost_energy: float
    cost_disruption: float
    gain_accuracy: float
    norm_cost_energy: float
    norm_cost_disruption: float
    norm_gain_accuracy: float
    c_max: float
    d_max: float
    a_max: float


def _compute_normalizers(dataset: MILPDataset) -> tuple[float, float, float]:
    services = [s.service_id for s in dataset.services]
    k_m = {s.service_id: s.valid_variants for s in dataset.services}
    e_cpu = {n.node_id: n.energy_cost for n in dataset.nodes}
    e_mem = {n.node_id: n.e_mem_unit for n in dataset.nodes}
    t_mig = {s.service_id: s.migration_cost for s in dataset.services}

    max_e_cpu = max(e_cpu.values()) if e_cpu else 1.0
    max_e_mem = max(e_mem.values()) if e_mem else 0.0

    c_max = (
        sum(max_e_cpu * max(dataset.r_req[(svc, var)] for var in k_m[svc]) for svc in services)
        + sum(max_e_mem * max(dataset.r_mem[(svc, var)] for var in k_m[svc]) for svc in services)
    )
    d_max = sum(t_mig.values())
    a_max = float(len(services))

    c_max = c_max if c_max > 0 else 1.0
    d_max = d_max if d_max > 0 else 1.0
    a_max = a_max if a_max > 0 else 1.0
    return float(c_max), float(d_max), float(a_max)


def _infer_migrations(dataset: MILPDataset, placement: PlacementDict) -> dict[str, bool]:
    migrations: dict[str, bool] = {}
    for svc, (variant, node) in placement.items():
        prev = dataset.x_prev.get((svc, variant, node), 0.0)
        migrations[svc] = bool(prev < 0.5)
    return migrations


def evaluate_objective_for_placement(
    dataset: MILPDataset,
    placement: PlacementDict,
    migrations: dict[str, bool] | None = None,
) -> ObjectiveBreakdown:
    """Evaluate normalized objective terms for a concrete placement."""
    e_cpu = {n.node_id: n.energy_cost for n in dataset.nodes}
    e_mem = {n.node_id: n.e_mem_unit for n in dataset.nodes}
    t_mig = {s.service_id: s.migration_cost for s in dataset.services}

    if migrations is None:
        migrations = _infer_migrations(dataset, placement)

    cost_energy = 0.0
    gain_accuracy = 0.0
    for svc, (variant, node) in placement.items():
        cpu = float(dataset.r_req[(svc, variant)])
        mem = float(dataset.r_mem[(svc, variant)])
        q = float(dataset.acc[(svc, variant)])
        cost_energy += float(e_cpu[node] * cpu + e_mem[node] * mem)
        gain_accuracy += q

    cost_disruption = 0.0
    for svc, did_migrate in migrations.items():
        if did_migrate:
            cost_disruption += float(t_mig[svc])

    c_max, d_max, a_max = _compute_normalizers(dataset)
    norm_c = cost_energy / c_max
    norm_d = cost_disruption / d_max
    norm_a = gain_accuracy / a_max

    objective_j = float(dataset.w_c * norm_c + dataset.w_d * norm_d - dataset.w_a * norm_a)
    reward = -objective_j

    return ObjectiveBreakdown(
        objective_j=objective_j,
        reward=reward,
        cost_energy=cost_energy,
        cost_disruption=cost_disruption,
        gain_accuracy=gain_accuracy,
        norm_cost_energy=norm_c,
        norm_cost_disruption=norm_d,
        norm_gain_accuracy=norm_a,
        c_max=c_max,
        d_max=d_max,
        a_max=a_max,
    )


def compute_reward_for_placement(
    dataset: MILPDataset,
    placement: PlacementDict,
    migrations: dict[str, bool] | None = None,
) -> float:
    """Return reward = -J for a given placement."""
    return evaluate_objective_for_placement(dataset, placement, migrations).reward
