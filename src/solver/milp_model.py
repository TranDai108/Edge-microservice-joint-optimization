# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportOptionalOperand=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportReturnType=false
"""
milp_model.py

MILP solver for joint AI model-variant selection and microservice placement
on edge/cloud nodes.

Solver choice: Pyomo + HiGHS
  - Pyomo: expressive algebraic modeling language (maps directly to math notation)
  - HiGHS: state-of-the-art open-source MILP solver (Beale-Orchard-Hays Prize 2022)
  - Together they provide academic-grade, reproducible, license-free baselines.

Problem formulation:
  Minimize  J = w_c * C_norm(x) + w_d * D_norm(v) - w_a * A_norm(x)
  Subject to C1 – C5

  Where:
    C_norm = C(x) / C_max   — normalized energy cost       ∈ [0, 1]
    D_norm = D(v) / D_max   — normalized disruption cost   ∈ [0, 1]
    A_norm = A(x) / A_max   — normalized quality gain      ∈ [0, 1]

  This ensures w_c, w_d, w_a behave as strict percentage weights regardless
  of the absolute scale of energy (Watts) or migration costs (seconds).

Key design decisions vs. old solver/:
  1. Per-service variant pool K_m: each service m has its own valid_variants,
     replacing the old global K applied uniformly to all services.
     This prevents non-AI services from being assigned AI model variants.
  2. Sparse decision variable set Valid_MKN: x[m,k,n] is only created for
     valid (service, variant, node) triples, reducing problem size.
  3. Q[m,k] quality metric: non-AI services are locked at Q=1.0 so they
     do not artificially lower the accuracy term in the objective.
  4. Normalized objective: C_max, D_max, A_max denominators ensure the
     objective value J ∈ [0, 1] and weights act as true percentages.
"""

import time
import os
import sys
from dataclasses import dataclass
from typing import Optional

import pyomo.environ as pyo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from solver.dataset_generator import MILPDataset
except ImportError:
    from dataset_generator import MILPDataset


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlacementResult:
    """Solution returned by the MILP solver."""

    status: str                              # "optimal", "feasible", "infeasible", ...
    objective_value: float                   # total J value (normalized, in [0,1])

    # Objective components (raw, pre-normalization — for thesis analysis)
    cost_energy: float                       # C(x)  raw Watts
    cost_disruption: float                   # D(v)  raw migration cost
    gain_accuracy: float                     # A(x)  raw quality sum

    # Normalized components (what actually drive the objective)
    norm_cost_energy: float                  # C_norm = C(x) / C_max
    norm_cost_disruption: float              # D_norm = D(v) / D_max
    norm_gain_accuracy: float                # A_norm = A(x) / A_max

    # Placement: service_id -> (variant_id, node_id)
    placement: dict[str, tuple[str, str]]

    # Migration flags: service_id -> bool
    migrations: dict[str, bool]

    # Types: "Node Migration", "AI Model Redeployment", "Stayed"
    migration_types: dict[str, str]

    # Previous Placement: service_id -> (variant_id, node_id)
    prev_placement: dict[str, tuple[str, str]]

    # Oversubscription ratios: node_id -> float (CPU dimension)
    theta: dict[str, float]

    # Resource usage per node: node_id -> float
    resource_usage: dict[str, float]     # CPU cores used per node
    mem_usage:      dict[str, float]     # RAM GB used per node
    background_cpu: dict[str, float]     # Non-controlled CPU load per node
    background_mem: dict[str, float]     # Non-controlled RAM load per node

    # Solver wall-clock time (seconds)
    solve_time: float

    def summary(self) -> str:
        migrated = sum(1 for v in self.migrations.values() if v)
        return (
            f"Status          : {self.status}\n"
            f"Objective J     : {self.objective_value:.4f}  (normalized, ∈ [0,1])\n"
            f"  w_c·C_norm    : {self.norm_cost_energy:.4f}  (raw C={self.cost_energy:.4f})\n"
            f"  w_d·D_norm    : {self.norm_cost_disruption:.4f}  (raw D={self.cost_disruption:.4f})\n"
            f"  w_a·A_norm    : {self.norm_gain_accuracy:.4f}  (raw A={self.gain_accuracy:.4f})\n"
            f"Migrations      : {migrated}/{len(self.migrations)} services\n"
            f"Solve time      : {self.solve_time:.3f}s\n"
        )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_placement(
    dataset: MILPDataset,
    time_limit: int = 120,
    verbose: bool = False,
) -> Optional[PlacementResult]:
    """
    Solve the MILP placement problem.

    Args:
        dataset   : MILPDataset from dataset_generator.py or metrics_collector.py
        time_limit: solver wall-clock limit in seconds
        verbose   : if True, print Pyomo/HiGHS solver output

    Returns:
        PlacementResult if a feasible solution is found, None otherwise.
    """

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------
    N = [n.node_id for n in dataset.nodes]
    M = [s.service_id for s in dataset.services]

    # Per-service variant pool K_m (replaces old global K)
    K_m: dict[str, list[str]] = {
        s.service_id: s.valid_variants for s in dataset.services
    }

    cap_cpu: dict[str, float] = {n.node_id: n.cap_cpu     for n in dataset.nodes}
    cap_mem: dict[str, float] = {n.node_id: n.cap_mem_gb   for n in dataset.nodes}
    e_cpu:   dict[str, float] = {n.node_id: n.energy_cost  for n in dataset.nodes}  # W/core
    e_mem:   dict[str, float] = {n.node_id: n.e_mem_unit   for n in dataset.nodes}  # W/GB
    T_mig:   dict[str, float] = {s.service_id: s.migration_cost for s in dataset.services}

    r_req  = dataset.r_req    # CPU cores  dict[(m,k)] -> float
    r_mem  = dataset.r_mem    # RAM GB     dict[(m,k)] -> float
    acc    = dataset.acc      # Q[m,k]     dict[(m,k)] -> float
    x_prev = dataset.x_prev   # dict[(m,k,n)] -> float  (0.0 or 1.0)

    w_c = dataset.w_c
    w_d = dataset.w_d
    w_a = dataset.w_a
    theta_max   = dataset.theta_max
    v_storm_max = dataset.v_storm_max
    background_cpu: dict[str, float] = {
        n: max(0.0, float(dataset.background_cpu.get(n, 0.0))) for n in N
    }
    background_mem: dict[str, float] = {
        n: max(0.0, float(dataset.background_mem.get(n, 0.0))) for n in N
    }

    # Sparse index: only valid (service, variant, node) triples
    valid_mkn = [
        (svc, var, n)
        for svc in M
        for var in K_m[svc]
        for n in N
    ]

    # ------------------------------------------------------------------
    # Normalization scalars (computed from parameters, not variables)
    # These make J dimensionless and ensure w_c/w_d/w_a act as % weights.
    # ------------------------------------------------------------------

    # C_max: worst-case energy cost — evaluated independently per resource dimension.
    # Formula (from thesis eq. C_max):
    #   max_n(E_cpu_unit) * sum_m max_k(R_cpu[m,k])
    #   + max_n(E_mem_unit) * sum_m max_k(R_mem[m,k])
    # This preserves C_norm ∈ [0, 1] with the dual-energy objective.
    max_e_cpu = max(e_cpu.values()) if e_cpu else 1.0
    max_e_mem = max(e_mem.values()) if e_mem else 0.0

    c_max = (
        # CPU energy dimension
        sum(max_e_cpu * max(r_req[(svc, var)] for var in K_m[svc]) for svc in M)
        +
        # DRAM energy dimension
        sum(max_e_mem * max(r_mem[(svc, var)] for var in K_m[svc]) for svc in M)
    )

    # D_max: worst-case disruption = every service migrates simultaneously
    d_max = sum(T_mig.values())

    # A_max: best-case quality = every service achieves Q=1.0
    a_max = float(len(M))

    # Guard against degenerate edge cases
    c_max = c_max if c_max > 0 else 1.0
    d_max = d_max if d_max > 0 else 1.0
    a_max = a_max if a_max > 0 else 1.0

    # ------------------------------------------------------------------
    # Build Pyomo ConcreteModel
    # ------------------------------------------------------------------
    model = pyo.ConcreteModel(name="EdgeMILP")

    # Sets
    model.N         = pyo.Set(initialize=N)
    model.M         = pyo.Set(initialize=M)
    model.Valid_MKN = pyo.Set(initialize=valid_mkn)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------

    # x[m,k,n] in {0,1} — service m uses variant k on node n
    # Only defined over Valid_MKN (sparse) — eliminates impossible combos
    model.x = pyo.Var(model.Valid_MKN, domain=pyo.Binary)

    # v[m] in {0,1} — service m migrates this time step
    model.v = pyo.Var(model.M, domain=pyo.Binary)

    # theta[n] in R+ — CPU oversubscription ratio at node n
    # CPU is compressible: throttling slows containers but does not crash them.
    # theta ∈ [1.0, theta_max] allows controlled oversubscription.
    model.theta = pyo.Var(model.N, domain=pyo.NonNegativeReals,
                          bounds=(1.0, theta_max))

    # NOTE: No theta_mem variable.
    # RAM is incompressible: oversubscription triggers OOM kills (pod crashes).
    # C2_mem is therefore a HARD constraint with no slack — Cap_mem[n] is a
    # strict upper bound. This mirrors how Kubernetes enforces memory limits
    # vs. CPU limits (throttle vs. kill).

    # ------------------------------------------------------------------
    # Objective: minimize J = w_c*C_norm + w_d*D_norm - w_a*A_norm
    # All three terms are in [0, 1] — weights truly act as percentages.
    # ------------------------------------------------------------------

    def objective_rule(m: pyo.ConcreteModel) -> pyo.Expression:
        # C(x) = sum_{n,m,k} ( E_cpu[n]*R_cpu[m,k] + E_mem[n]*R_mem[m,k] ) * x[m,k,n]
        energy = sum(
            (e_cpu[n] * r_req[(svc, var)] + e_mem[n] * r_mem[(svc, var)])
            * m.x[svc, var, n]
            for svc in M for var in K_m[svc] for n in N
        )
        # D(v) = sum_m v[m] * T[m]
        disruption = sum(m.v[svc] * T_mig[svc] for svc in M)
        # A(x) = sum_{m,k} Q[m,k] * sum_n x[m,k,n]
        accuracy = sum(
            acc[(svc, var)] * sum(m.x[svc, var, n] for n in N)
            for svc in M for var in K_m[svc]
        )

        norm_c = energy     / c_max
        norm_d = disruption / d_max
        norm_a = accuracy   / a_max

        return w_c * norm_c + w_d * norm_d - w_a * norm_a

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # C1: Unique placement — each service deployed exactly once
    # sum_{k∈K_m, n∈N} x[m,k,n] = 1  for all m
    def c1_unique_placement(m: pyo.ConcreteModel, svc: str) -> pyo.Constraint:
        return sum(m.x[svc, var, n] for var in K_m[svc] for n in N) == 1

    model.c1 = pyo.Constraint(model.M, rule=c1_unique_placement)

    # C2_cpu: CPU capacity — sum R_cpu[m,k]*x[m,k,n] <= Cap_cpu[n] * theta_cpu[n]
    def c2_cpu(m: pyo.ConcreteModel, n: str) -> pyo.Constraint:
        lhs = sum(r_req[(svc, var)] * m.x[svc, var, n]
                  for svc in M for var in K_m[svc])
        return lhs + background_cpu[n] <= cap_cpu[n] * m.theta[n]  # type: ignore[return-value]

    model.c2_cpu = pyo.Constraint(model.N, rule=c2_cpu)

    # C2_mem: RAM hard limit — no oversubscription (theta_mem = 1.0 fixed).
    # RAM is incompressible: exceeding Cap_mem[n] causes OOM kills.
    # n0/n1 (3.82 GB) vs n2/n3 (7.75 GB) creates genuine asymmetry —
    # the solver will prefer n2/n3 for RAM-heavy services like
    # detection:yolo26-medium (1.65 GB) or gen-ai:llama-3b-small (1.0 GB).
    def c2_mem(m: pyo.ConcreteModel, n: str) -> pyo.Constraint:
        lhs = sum(r_mem[(svc, var)] * m.x[svc, var, n]
                  for svc in M for var in K_m[svc])
        return lhs + background_mem[n] <= cap_mem[n]  # type: ignore[return-value]

    model.c2_mem = pyo.Constraint(model.N, rule=c2_mem)

    # C3: Oversubscription safety — encoded in theta bounds (1.0, theta_max)
    # Pyomo Var bounds handle this — no separate constraint needed.

    # C4: Migration definition — v[m] >= x[m,k,n](t) - x[m,k,n](t-1)
    # One constraint per valid (m, k, n) triple.
    def c4_migration_def(
        m: pyo.ConcreteModel, svc: str, var: str, n: str
    ) -> pyo.Constraint:
        delta = x_prev.get((svc, var, n), 0.0)
        return m.v[svc] >= m.x[svc, var, n] - delta  # type: ignore[return-value]

    model.c4 = pyo.Constraint(model.Valid_MKN, rule=c4_migration_def)

    # C5: Migration storm bound — sum_m v[m] <= V_storm_max
    model.c5 = pyo.Constraint(
        expr=sum(model.v[svc] for svc in M) <= v_storm_max
    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available():
        solver = pyo.SolverFactory("cbc")
        if not solver.available():
            raise RuntimeError(
                "No suitable MILP solver found. "
                "Install HiGHS with: pip install highspy\n"
                "or CBC with: brew install cbc"
            )

    options: dict[str, object] = {"time_limit": float(time_limit)}
    if not verbose:
        options["output_flag"] = False

    t0 = time.perf_counter()
    result = solver.solve(model, tee=verbose, options=options)
    solve_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Check termination
    # ------------------------------------------------------------------
    condition = result.solver.termination_condition
    feasible_conditions = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    }
    if condition not in feasible_conditions:
        print(f"[MILP] Solver terminated with: {condition}")
        return None

    status = str(condition)

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------

    # Placement: for each service, find (variant, node) where x == 1
    placement: dict[str, tuple[str, str]] = {}
    for svc in M:
        for var in K_m[svc]:
            for n in N:
                if pyo.value(model.x[svc, var, n]) > 0.5:
                    placement[svc] = (var, n)

    # Previous placement extraction
    prev_placement: dict[str, tuple[str, str]] = {}
    for (svc, var, n), val in dataset.x_prev.items():
        if val > 0.5:
            prev_placement[svc] = (var, n)

    # Migration decisions
    migrations: dict[str, bool] = {
        svc: bool(pyo.value(model.v[svc]) > 0.5) for svc in M
    }

    # Migration types — four cases based on what ACTUALLY changed
    # Note: when w_d=0 the solver may set v[m]=1 even if nothing moved
    # (v is free since it doesn't affect the objective). We detect this
    # by comparing prev/curr placement and override the flag if unchanged.
    migration_types: dict[str, str] = {}
    for svc in M:
        prev_var, prev_node = prev_placement.get(svc, ("-", "-"))
        curr_var, curr_node = placement.get(svc, ("-", "-"))
        node_changed = prev_node != curr_node
        var_changed  = prev_var  != curr_var

        if not migrations[svc] or (not node_changed and not var_changed):
            # Truly stayed — or v[m] was set to 1 spuriously (w_d=0 free variable).
            # Override migrations flag so downstream display is consistent.
            migration_types[svc] = "Stayed"
            migrations[svc] = False
        elif node_changed and var_changed:
            migration_types[svc] = "Node Migration + AI Redeployment"
        elif node_changed:
            migration_types[svc] = "Node Migration"
        else:
            migration_types[svc] = "AI Model Redeployment"

    # Theta values
    theta_vals: dict[str, float] = {
        n: float(pyo.value(model.theta[n])) for n in N
    }

    # Resource usage per node (CPU cores)
    resource_usage: dict[str, float] = {}
    for n in N:
        usage = sum(
            r_req[(svc, var)] * pyo.value(model.x[svc, var, n])
            for svc in M for var in K_m[svc]
        )
        resource_usage[n] = float(usage) + background_cpu[n]

    # RAM usage per node (GB)
    mem_usage: dict[str, float] = {}
    for n in N:
        usage = sum(
            r_mem[(svc, var)] * pyo.value(model.x[svc, var, n])
            for svc in M for var in K_m[svc]
        )
        mem_usage[n] = float(usage) + background_mem[n]

    # Raw objective components (pre-normalization, for thesis analysis)
    # C(x) uses dual-energy: CPU watts + DRAM watts
    cost_energy = float(sum(
        (e_cpu[n] * r_req[(svc, var)] + e_mem[n] * r_mem[(svc, var)])
        * pyo.value(model.x[svc, var, n])
        for svc in M for var in K_m[svc] for n in N
    ))
    cost_disruption = float(sum(
        pyo.value(model.v[svc]) * T_mig[svc] for svc in M
    ))
    gain_accuracy = float(sum(
        acc[(svc, var)] * sum(pyo.value(model.x[svc, var, n]) for n in N)
        for svc in M for var in K_m[svc]
    ))

    # Normalized components
    norm_cost_energy      = cost_energy     / c_max
    norm_cost_disruption  = cost_disruption / d_max
    norm_gain_accuracy    = gain_accuracy   / a_max

    obj_val = float(pyo.value(model.objective))

    return PlacementResult(
        status=status,
        objective_value=obj_val,
        cost_energy=cost_energy,
        cost_disruption=cost_disruption,
        gain_accuracy=gain_accuracy,
        norm_cost_energy=norm_cost_energy,
        norm_cost_disruption=norm_cost_disruption,
        norm_gain_accuracy=norm_gain_accuracy,
        placement=placement,
        migrations=migrations,
        migration_types=migration_types,
        prev_placement=prev_placement,
        theta=theta_vals,
        resource_usage=resource_usage,
        mem_usage=mem_usage,
        background_cpu=background_cpu,
        background_mem=background_mem,
        solve_time=solve_time,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataset_generator import generate_dataset

    print("Generating dataset...")
    ds = generate_dataset(num_nodes=3, num_services=5, seed=42)
    print(ds.summary())
    print("\nServices:")
    for s in ds.services:
        print(f"  {s.service_id} ({s.service_type}): variants={s.valid_variants}")

    print("\nSolving MILP...")
    result = solve_placement(ds, verbose=False)

    if result is None:
        print("No feasible solution found.")
    else:
        print("\n=== Solution ===")
        print(result.summary())

        print("--- Placement ---")
        for svc, (var, node) in sorted(result.placement.items()):
            migrated = "⬆ migrated" if result.migrations[svc] else ""
            print(f"  {svc}: variant={var}, node={node}  {migrated}")

        print("\n--- Node resource usage ---")
        for node in sorted(result.resource_usage):
            n_spec = next(n for n in ds.nodes if n.node_id == node)
            cpu_used = result.resource_usage[node]
            mem_used = result.mem_usage[node]
            theta_v  = result.theta[node]
            print(f"  {node}: CPU={cpu_used:.2f}/{n_spec.cap_cpu:.1f}c  "
                  f"RAM={mem_used:.3f}/{n_spec.cap_mem_gb:.1f}GB  "
                  f"theta={theta_v:.3f}")

        assert -1.0 <= result.objective_value <= 1.0, \
            f"Objective J={result.objective_value:.4f} out of expected range [-1, 1]!"
        print(f"\n\u2713 Objective J={result.objective_value:.4f} is normalized correctly.")
