"""
dataset_generator.py

Generates synthetic datasets for the MILP-based edge AI placement problem.

Problem context:
  - N nodes (edge/cloud)
  - M microservices
  - K_m variants per service (heterogeneous: each service has its own valid set)

Each dataset encodes all parameters required by the MILP formulation:
  - R_req[m,k]   : resource requirement of service m with variant k
  - Cap[n]       : physical resource capacity of node n
  - E_unit[n]    : energy cost per resource unit at node n
  - Q[m,k]       : quality (accuracy) score — 1.0 for non-AI services
  - T[m]         : migration disruption cost for service m
  - x_prev[m,k,n]: previous placement (for migration constraint C4)
  - Theta_max    : max safe oversubscription ratio
  - V_storm_max  : max simultaneous migrations allowed
  - w_c, w_d, w_a: objective weights (behave as % after normalization)
"""

import random
import json
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)
from config import MILP_W_C, MILP_W_D, MILP_W_A
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rf(x: float, ndigits: int) -> float:
    """Round float to ndigits decimal places with an explicit float return type."""
    return float(round(x, ndigits))  # type: ignore[call-overload]


@dataclass
class NodeSpec:
    node_id: str
    cap_cpu: float           # Cap_cpu[n] — allocatable CPU cores
    cap_mem_gb: float        # Cap_mem[n] — allocatable RAM in GB
    energy_cost: float       # E_cpu_unit[n] — Watts per CPU core (from Kepler package_joules)
    e_mem_unit: float = 0.0  # E_mem_unit[n] — Watts per RAM GB  (from Kepler dram_joules)

    # Keep capacity as alias for backward compatibility with any existing code
    @property
    def capacity(self) -> float:
        return self.cap_cpu


@dataclass
class ServiceSpec:
    service_id: str
    service_type: str        # e.g. "detection", "gateway", "ingest"
    migration_cost: float    # T[m]
    valid_variants: list[str]  # K_m — only these variants are valid for this service


@dataclass
class MILPDataset:
    # Sets
    nodes: list[NodeSpec]
    services: list[ServiceSpec]

    # Parameters indexed by (service_id, variant_id)
    r_req: dict[tuple[str, str], float]   # CPU requirement R_cpu[m,k] in cores
    r_mem: dict[tuple[str, str], float]   # RAM requirement R_mem[m,k] in GB
    acc: dict[tuple[str, str], float]     # quality score Q[m,k] (1.0 for non-AI)

    # Previous placement x_prev[m,k,n]
    x_prev: dict[tuple[str, str, str], float]

    # Global parameters
    theta_max: float         # max safe oversubscription ratio (>= 1.0)
    v_storm_max: int         # max simultaneous migrations allowed
    w_c: float               # weight for energy cost   (w_c + w_d + w_a should = 1)
    w_d: float               # weight for disruption
    w_a: float               # weight for accuracy/quality
    background_cpu: dict[str, float] = field(default_factory=dict)
    background_mem: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        total_vars = sum(len(s.valid_variants) for s in self.services)
        total_x    = sum(len(s.valid_variants) * len(self.nodes) for s in self.services)
        cpu_caps   = [f"{n.cap_cpu}c" for n in self.nodes]
        mem_caps   = [f"{n.cap_mem_gb:.1f}GB" for n in self.nodes]
        return (
            f"Nodes: {len(self.nodes)} ({', '.join(cpu_caps)} CPU | {', '.join(mem_caps)} RAM), "
            f"Services: {len(self.services)}, "
            f"Total variant options: {total_vars}, "
            f"Decision variables x: {total_x}"
        )


# ---------------------------------------------------------------------------
# Service type → valid variants pool
# ---------------------------------------------------------------------------

DEFAULT_SERVICE_TYPE_POOL: dict[str, list[str]] = {
    # Non-AI pipeline services — single "standard" variant, Q=1.0
    "gateway":    ["standard"],
    "ingest":     ["standard"],
    "preprocess": ["standard"],
    "postprocess":["standard"],
    # AI inference services — multiple variants with different accuracy/cost tradeoffs
    "detection":  ["yolo26-nano", "yolo26-small", "yolo26-medium"],
    "gen_ai":     ["light", "heavy"],
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_dataset(
    num_nodes: int = 5,
    num_services: int = 8,
    service_type_pool: Optional[dict[str, list[str]]] = None,
    cap_cpu_range: tuple[float, float] = (2.0, 8.0),
    cap_mem_range: tuple[float, float] = (2.0, 16.0),
    energy_cost_range: tuple[float, float] = (0.5, 2.0),
    r_req_range: tuple[float, float] = (0.1, 2.0),
    r_mem_range: tuple[float, float] = (0.05, 2.0),
    acc_range: tuple[float, float] = (0.6, 0.99),
    migration_cost_range: tuple[float, float] = (1.0, 5.0),
    theta_max: float = 1.3,
    v_storm_max: int = 3,
    w_c: float = MILP_W_C,
    w_d: float = MILP_W_D,
    w_a: float = MILP_W_A,
    seed: int = 42,
    with_prev_placement: bool = True,
) -> MILPDataset:
    """
    Generate a random synthetic dataset for the MILP edge placement problem.

    Each service is randomly assigned a type from service_type_pool, which
    determines its valid_variants. Non-AI services get Q=1.0 (locked quality),
    AI services get graduated accuracy values across their variants.
    Both CPU (cores) and RAM (GB) are modelled as independent resource dimensions.

    Args:
        num_nodes           : number of edge/cloud nodes
        num_services        : number of microservices to place
        service_type_pool   : dict mapping service_type -> [valid_variants].
                              Defaults to DEFAULT_SERVICE_TYPE_POOL.
        cap_cpu_range       : (min, max) allocatable CPU cores per node
        cap_mem_range       : (min, max) allocatable RAM (GB) per node
        energy_cost_range   : (min, max) energy cost per CPU core at a node (W/core)
        r_req_range         : (min, max) CPU requirement per (service, variant) in cores
        r_mem_range         : (min, max) RAM requirement per (service, variant) in GB
        acc_range           : (min, max) quality score for AI variants (heavier = higher)
        migration_cost_range: (min, max) disruption cost per service migration
        theta_max           : max oversubscription ratio (>= 1.0)
        v_storm_max         : max simultaneous migrations allowed
        w_c, w_d, w_a       : objective weights (normalized in MILP, so scale doesn't matter)
        seed                : random seed for reproducibility
        with_prev_placement : if True, generate a random previous placement x_prev

    Returns:
        MILPDataset instance with all parameters populated.
    """
    rng = random.Random(seed)

    if service_type_pool is None:
        service_type_pool = DEFAULT_SERVICE_TYPE_POOL

    # --- Nodes: heterogeneous CPU + RAM capacity ---
    cpu_lo, cpu_hi = cap_cpu_range
    mem_lo, mem_hi = cap_mem_range
    e_lo,   e_hi   = energy_cost_range
    nodes = [
        NodeSpec(
            node_id=f"n{i}",
            cap_cpu=_rf(rng.uniform(cpu_lo, cpu_hi), 1),
            cap_mem_gb=_rf(rng.uniform(mem_lo, mem_hi), 1),
            energy_cost=_rf(rng.uniform(e_lo, e_hi), 3),
            # E_mem_unit: DRAM draws ~18.7W (Raspberry Pi 4 baseline) divided
            # by the node's allocatable RAM — smaller nodes pay more per GB.
            # Add small noise to simulate hardware variance.
            e_mem_unit=_rf(max(0.1, 18.7 / _rf(rng.uniform(mem_lo, mem_hi), 1)
                              + rng.uniform(-0.5, 0.5)), 4),
        )
        for i in range(num_nodes)
    ]

    # --- Services ---
    mig_lo, mig_hi = migration_cost_range
    pool_keys = list(service_type_pool.keys())
    services = []
    for j in range(num_services):
        chosen_type = rng.choice(pool_keys)
        services.append(ServiceSpec(
            service_id=f"m{j}",
            service_type=chosen_type,
            migration_cost=_rf(rng.uniform(mig_lo, mig_hi), 2),
            valid_variants=service_type_pool[chosen_type],
        ))

    # --- R_req (CPU) + R_mem (RAM) + Q[m,k] indexed by (service_id, variant_id) ---
    r_req: dict[tuple[str, str], float] = {}  # CPU cores
    r_mem: dict[tuple[str, str], float] = {}  # RAM GB
    acc:   dict[tuple[str, str], float] = {}

    r_req_min, r_req_max = r_req_range
    r_mem_min, r_mem_max = r_mem_range
    acc_min,   acc_max   = acc_range

    for svc in services:
        n_vars = len(svc.valid_variants)
        for idx, var in enumerate(svc.valid_variants):
            if n_vars == 1:
                # Standard (non-AI) service: lighter resource use, Q fixed at 1.0
                r_req[(svc.service_id, var)] = _rf(rng.uniform(r_req_min, r_req_max / 2), 2)
                r_mem[(svc.service_id, var)] = _rf(rng.uniform(r_mem_min, r_mem_max / 4), 3)
                acc[(svc.service_id, var)]   = 1.0
            else:
                # AI service: heavier variant → more CPU + RAM, higher quality
                weight = idx / max(n_vars - 1, 1)   # 0.0 (lightest) → 1.0 (heaviest)
                r_req[(svc.service_id, var)] = _rf(
                    r_req_min + weight * (r_req_max - r_req_min)
                    + rng.uniform(-0.05, 0.05), 2
                )
                r_mem[(svc.service_id, var)] = _rf(
                    max(0.01, r_mem_min + weight * (r_mem_max - r_mem_min)
                    + rng.uniform(-0.1, 0.1)), 3
                )
                acc[(svc.service_id, var)] = _rf(
                    min(acc_max, max(acc_min,
                        acc_min + weight * (acc_max - acc_min)
                        + rng.uniform(-0.05, 0.05)
                    )), 4
                )

    # --- Previous placement x_prev ---
    # Sparse dict: only valid (m, k, n) combinations initialized
    x_prev: dict[tuple[str, str, str], float] = {}

    for svc in services:
        for var in svc.valid_variants:
            for node in nodes:
                x_prev[(svc.service_id, var, node.node_id)] = 0.0

    if with_prev_placement:
        for svc in services:
            chosen_var  = rng.choice(svc.valid_variants)
            chosen_node = rng.choice(nodes)
            x_prev[(svc.service_id, chosen_var, chosen_node.node_id)] = 1.0

    return MILPDataset(
        nodes=nodes,
        services=services,
        r_req=r_req,
        r_mem=r_mem,
        acc=acc,
        x_prev=x_prev,
        theta_max=theta_max,
        v_storm_max=v_storm_max,
        w_c=w_c,
        w_d=w_d,
        w_a=w_a,
    )


# ---------------------------------------------------------------------------
# Serialization helpers  (save/load for experiment reproducibility)
# ---------------------------------------------------------------------------

def dataset_to_dict(ds: MILPDataset) -> dict[str, object]:
    """Convert a MILPDataset to a JSON-serializable dict."""
    return {
        "nodes": [
            {
                "node_id":    n.node_id,
                "cap_cpu":    n.cap_cpu,
                "cap_mem_gb": n.cap_mem_gb,
                "energy_cost":n.energy_cost,
                "e_mem_unit": n.e_mem_unit,
            }
            for n in ds.nodes
        ],
        "services": [
            {
                "service_id":    s.service_id,
                "service_type":  s.service_type,
                "migration_cost":s.migration_cost,
                "valid_variants":s.valid_variants,
            }
            for s in ds.services
        ],
        "r_req":  {f"{m},{k}": v for (m, k), v in ds.r_req.items()},
        "r_mem":  {f"{m},{k}": v for (m, k), v in ds.r_mem.items()},
        "acc":    {f"{m},{k}": v for (m, k), v in ds.acc.items()},
        "x_prev": {f"{m},{k},{n}": v for (m, k, n), v in ds.x_prev.items()},
        "background_cpu": dict(ds.background_cpu),
        "background_mem": dict(ds.background_mem),
        "theta_max":   ds.theta_max,
        "v_storm_max": ds.v_storm_max,
        "w_c": ds.w_c,
        "w_d": ds.w_d,
        "w_a": ds.w_a,
    }


def dataset_from_dict(d: dict) -> MILPDataset:
    """Reconstruct a MILPDataset from a JSON-loaded dict (backward-compatible)."""
    nodes = [
        NodeSpec(
            node_id=str(raw["node_id"]),
            # Support both old format ("capacity") and new format ("cap_cpu")
            cap_cpu=float(raw.get("cap_cpu", raw.get("capacity", 2.0))),
            cap_mem_gb=float(raw.get("cap_mem_gb", 2.0)),   # default 2GB if old file
            energy_cost=float(raw["energy_cost"]),
            e_mem_unit=float(raw.get("e_mem_unit", 0.0)),   # default 0.0 for old files
        )
        for raw in d["nodes"]
    ]
    services = [
        ServiceSpec(
            service_id=str(raw["service_id"]),
            service_type=str(raw.get("service_type", "unknown")),
            migration_cost=float(raw["migration_cost"]),
            valid_variants=list(raw.get("valid_variants", ["standard"])),
        )
        for raw in d["services"]
    ]

    r_req: dict[tuple[str, str], float] = {
        (parts[0], parts[1]): float(v)
        for k, v in d["r_req"].items()
        for parts in [k.split(",")]
    }
    # r_mem: backward-compatible — old datasets don't have it, default to 0.1 GB
    r_mem: dict[tuple[str, str], float] = {
        (parts[0], parts[1]): float(v)
        for k, v in d.get("r_mem", {}).items()
        for parts in [k.split(",")]
    } if "r_mem" in d else {k: 0.1 for k in r_req}
    acc: dict[tuple[str, str], float] = {
        (parts[0], parts[1]): float(v)
        for k, v in d["acc"].items()
        for parts in [k.split(",")]
    }
    x_prev: dict[tuple[str, str, str], float] = {
        (parts[0], parts[1], parts[2]): float(v)
        for k, v in d["x_prev"].items()
        for parts in [k.split(",")]
    }
    background_cpu = {
        str(k): float(v)
        for k, v in d.get("background_cpu", {}).items()
    }
    background_mem = {
        str(k): float(v)
        for k, v in d.get("background_mem", {}).items()
    }

    return MILPDataset(
        nodes=nodes,
        services=services,
        r_req=r_req,
        r_mem=r_mem,
        acc=acc,
        x_prev=x_prev,
        theta_max=float(d["theta_max"]),
        v_storm_max=int(d["v_storm_max"]),
        w_c=float(d["w_c"]),
        w_d=float(d["w_d"]),
        w_a=float(d["w_a"]),
        background_cpu=background_cpu,
        background_mem=background_mem,
    )


def save_dataset(ds: MILPDataset, path: str) -> None:
    """Save dataset as a JSON file for reproducibility."""
    with open(path, "w") as f:
        json.dump(dataset_to_dict(ds), f, indent=2)
    print(f"Dataset saved to {path}")


def load_dataset(path: str) -> MILPDataset:
    """Load a dataset from a JSON file."""
    with open(path, "r") as f:
        d = json.load(f)
    return dataset_from_dict(d)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = generate_dataset(num_nodes=3, num_services=5, seed=42)

    print("=== Generated Dataset ===")
    print(ds.summary())

    print("\n--- Services & variants ---")
    for s in ds.services:
        print(f"  {s.service_id} ({s.service_type}): variants={s.valid_variants}")
        for var in s.valid_variants:
            print(f"    {var}: R_req={ds.r_req[(s.service_id, var)]:.2f}  "
                  f"Q={ds.acc[(s.service_id, var)]:.4f}")

    print("\n--- Previous placement (non-zero entries) ---")
    for (m, k, n), v in ds.x_prev.items():
        if v > 0:
            print(f"  service={m}, variant={k}, node={n}")

    # Round-trip serialization test
    save_dataset(ds, "/tmp/milp_dataset_test.json")
    ds2 = load_dataset("/tmp/milp_dataset_test.json")
    assert ds.r_req == ds2.r_req, "Serialization round-trip failed!"
    print("\nSerialization round-trip: OK ✓")
