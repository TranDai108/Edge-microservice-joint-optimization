"""Storm-safe constrained decoder for legacy 44-dim MultiDiscrete policy heads."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable, Sequence

import numpy as np

DEFAULT_ACTION_DIMS: tuple[int, ...] = (4, 4, 4, 12, 12, 4)
DEFAULT_TOPK_PER_HEAD: tuple[int, ...] = (2, 2, 2, 4, 4, 2)
_EPS = 1e-9
_FULL_ACTION_MATRIX_CACHE: dict[tuple[int, ...], np.ndarray] = {}


@dataclass
class Legacy44DecodeOutcome:
    action: np.ndarray
    placement: dict[str, tuple[str, str]]
    objective_j: float
    migration_count: int
    storm_max: int
    feasible: bool
    decoder_mode: str
    fallback_full_enumeration: bool
    candidates_evaluated: int


@dataclass
class _FastObjectiveContext:
    caps_cpu: dict[str, float]
    caps_mem: dict[str, float]
    r_req: dict[tuple[str, str], float]
    r_mem: dict[tuple[str, str], float]
    acc: dict[tuple[str, str], float]
    e_cpu: dict[str, float]
    e_mem: dict[str, float]
    mig_cost: dict[str, float]
    x_prev: dict[tuple[str, str, str], float]
    w_c: float
    w_d: float
    w_a: float
    c_max: float
    d_max: float
    a_max: float


@dataclass
class _VectorizedDecodeContext:
    service_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    caps_cpu: np.ndarray
    caps_mem: np.ndarray
    option_node_idx: tuple[np.ndarray, ...]
    option_cpu: tuple[np.ndarray, ...]
    option_mem: tuple[np.ndarray, ...]
    option_acc: tuple[np.ndarray, ...]
    option_energy: tuple[np.ndarray, ...]
    option_disruption: tuple[np.ndarray, ...]
    option_migration: tuple[np.ndarray, ...]
    option_valid: tuple[np.ndarray, ...]


def split_head_logits(
    flat_logits: np.ndarray,
    action_dims: Sequence[int] = DEFAULT_ACTION_DIMS,
) -> list[np.ndarray]:
    logits = np.asarray(flat_logits, dtype=np.float64).reshape(-1)
    if int(np.sum(action_dims)) != int(logits.shape[0]):
        raise ValueError(
            f"Flat logits size mismatch: got {logits.shape[0]}, expected {int(np.sum(action_dims))}"
        )
    out: list[np.ndarray] = []
    cursor = 0
    for d in action_dims:
        out.append(logits[cursor: cursor + int(d)].copy())
        cursor += int(d)
    return out


def apply_flat_mask_to_head_logits(
    head_logits: Sequence[np.ndarray],
    flat_mask: np.ndarray | Sequence[bool] | Sequence[float] | None,
    action_dims: Sequence[int] = DEFAULT_ACTION_DIMS,
) -> list[np.ndarray]:
    masked = [np.asarray(h, dtype=np.float64).copy() for h in head_logits]
    if flat_mask is None:
        return masked
    mask = np.asarray(flat_mask).reshape(-1)
    if int(mask.shape[0]) != int(np.sum(action_dims)):
        return masked
    cursor = 0
    for i, d in enumerate(action_dims):
        head_mask = mask[cursor: cursor + int(d)]
        cursor += int(d)
        invalid = head_mask <= 0.0
        if np.any(invalid):
            masked[i][invalid] = -np.inf
    return masked


def decode_legacy44_action(
    action: np.ndarray | Sequence[int],
    *,
    node_ids: Sequence[str],
    detection_variants: Sequence[str],
    gen_ai_variants: Sequence[str],
) -> dict[str, tuple[str, str]]:
    a = np.asarray(action, dtype=np.int64).tolist()
    nodes = list(node_ids)
    det_vars = list(detection_variants)
    gen_vars = list(gen_ai_variants)

    def _vn(value: int, variants: list[str]) -> tuple[str, str]:
        n_nodes = len(nodes)
        vi = min(max(int(value) // n_nodes, 0), len(variants) - 1)
        ni = min(max(int(value) % n_nodes, 0), n_nodes - 1)
        return variants[vi], nodes[ni]

    det_var, det_node = _vn(int(a[3]), det_vars)
    gen_var, gen_node = _vn(int(a[4]), gen_vars)
    return {
        "m0": ("standard", nodes[min(int(a[0]), len(nodes) - 1)]),
        "m1": ("standard", nodes[min(int(a[1]), len(nodes) - 1)]),
        "m2": ("standard", nodes[min(int(a[2]), len(nodes) - 1)]),
        "m3": (det_var, det_node),
        "m4": (gen_var, gen_node),
        "m5": ("standard", nodes[min(int(a[5]), len(nodes) - 1)]),
    }


def _topk_indices_from_logits(head_logits: np.ndarray, k: int) -> list[int]:
    logits = np.asarray(head_logits, dtype=np.float64).reshape(-1)
    finite_idx = np.where(np.isfinite(logits))[0]
    if len(finite_idx) <= 0:
        finite_idx = np.arange(len(logits), dtype=np.int64)
    if len(finite_idx) <= 0:
        return [0]
    kk = max(1, min(int(k), int(len(finite_idx))))
    finite_logits = logits[finite_idx]
    if kk >= len(finite_idx):
        ordered = finite_idx[np.argsort(-finite_logits)]
    else:
        part = np.argpartition(-finite_logits, kk - 1)[:kk]
        top = finite_idx[part]
        ordered = top[np.argsort(-logits[top])]
    return [int(i) for i in ordered.tolist()]


def _iter_topk_candidates(
    head_logits: Sequence[np.ndarray],
    topk_per_head: Sequence[int],
) -> Iterable[np.ndarray]:
    choices: list[list[int]] = []
    for i, logits in enumerate(head_logits):
        k = int(topk_per_head[i]) if i < len(topk_per_head) else len(logits)
        choices.append(_topk_indices_from_logits(np.asarray(logits, dtype=np.float64), k))
    for combo in product(*choices):
        yield np.asarray(combo, dtype=np.int64)


def _iter_full_candidates(action_dims: Sequence[int]) -> Iterable[np.ndarray]:
    dims = [range(int(d)) for d in action_dims]
    for combo in product(*dims):
        yield np.asarray(combo, dtype=np.int64)


def _candidate_matrix_from_choices(choices: Sequence[Sequence[int]]) -> np.ndarray:
    if not choices:
        return np.empty((0, 0), dtype=np.int64)
    arrays = [np.asarray(c, dtype=np.int64).reshape(-1) for c in choices]
    if any(arr.size == 0 for arr in arrays):
        return np.empty((0, len(arrays)), dtype=np.int64)
    grids = np.meshgrid(*arrays, indexing="ij")
    return np.stack(grids, axis=-1).reshape(-1, len(arrays))


def _candidate_matrix_full(action_dims: Sequence[int]) -> np.ndarray:
    key = tuple(int(d) for d in action_dims)
    cached = _FULL_ACTION_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    choices = [np.arange(int(d), dtype=np.int64) for d in key]
    matrix = _candidate_matrix_from_choices(choices)
    _FULL_ACTION_MATRIX_CACHE[key] = matrix
    return matrix


def _migration_count(ds: Any, placement: dict[str, tuple[str, str]]) -> int:
    x_prev = getattr(ds, "x_prev", {})
    return int(
        sum(
            1
            for svc, (var, node) in placement.items()
            if float(x_prev.get((svc, var, node), 0.0)) < 0.5
        )
    )


def _build_fast_objective_context(dataset: Any) -> _FastObjectiveContext:
    nodes = list(dataset.nodes)
    services = list(dataset.services)
    caps_cpu = {str(n.node_id): float(n.cap_cpu) for n in nodes}
    caps_mem = {str(n.node_id): float(n.cap_mem_gb) for n in nodes}
    r_req = {(str(svc), str(var)): float(v) for (svc, var), v in dict(dataset.r_req).items()}
    r_mem = {(str(svc), str(var)): float(v) for (svc, var), v in dict(dataset.r_mem).items()}
    acc = {(str(svc), str(var)): float(v) for (svc, var), v in dict(dataset.acc).items()}
    e_cpu = {str(n.node_id): float(n.energy_cost) for n in nodes}
    e_mem = {str(n.node_id): float(n.e_mem_unit) for n in nodes}
    mig_cost = {str(s.service_id): float(s.migration_cost) for s in services}
    x_prev = {(str(s), str(v), str(n)): float(val) for (s, v, n), val in dict(dataset.x_prev).items()}
    w_c = float(getattr(dataset, "w_c", 0.15))
    w_d = float(getattr(dataset, "w_d", 0.10))
    w_a = float(getattr(dataset, "w_a", 0.75))

    svc_ids = [str(s.service_id) for s in services]
    svc_vars = {str(s.service_id): [str(v) for v in s.valid_variants] for s in services}
    max_e_cpu = max(e_cpu.values()) if e_cpu else 1.0
    max_e_mem = max(e_mem.values()) if e_mem else 0.0
    c_max = 0.0
    for svc in svc_ids:
        vars_ = svc_vars.get(svc, [])
        if not vars_:
            continue
        c_max += max_e_cpu * max(r_req.get((svc, var), 0.0) for var in vars_)
        c_max += max_e_mem * max(r_mem.get((svc, var), 0.0) for var in vars_)
    d_max = float(sum(mig_cost.values()))
    a_max = float(len(svc_ids))
    if c_max <= 0.0:
        c_max = 1.0
    if d_max <= 0.0:
        d_max = 1.0
    if a_max <= 0.0:
        a_max = 1.0

    return _FastObjectiveContext(
        caps_cpu=caps_cpu,
        caps_mem=caps_mem,
        r_req=r_req,
        r_mem=r_mem,
        acc=acc,
        e_cpu=e_cpu,
        e_mem=e_mem,
        mig_cost=mig_cost,
        x_prev=x_prev,
        w_c=w_c,
        w_d=w_d,
        w_a=w_a,
        c_max=float(c_max),
        d_max=float(d_max),
        a_max=float(a_max),
    )


def _evaluate_fast(
    ctx: _FastObjectiveContext,
    placement: dict[str, tuple[str, str]],
    *,
    storm_max: int,
) -> tuple[bool, int, float] | None:
    used_cpu = {nid: 0.0 for nid in ctx.caps_cpu}
    used_mem = {nid: 0.0 for nid in ctx.caps_mem}
    migration_count = 0
    cost_energy = 0.0
    cost_disruption = 0.0
    gain_accuracy = 0.0

    for svc, (var, node) in placement.items():
        svc_id = str(svc)
        var_id = str(var)
        node_id = str(node)
        key = (svc_id, var_id)
        if node_id not in used_cpu:
            return None
        cpu = ctx.r_req.get(key)
        mem = ctx.r_mem.get(key)
        qa = ctx.acc.get(key)
        if cpu is None or mem is None or qa is None:
            return None
        used_cpu[node_id] += cpu
        used_mem[node_id] += mem
        if used_cpu[node_id] > ctx.caps_cpu[node_id] + _EPS:
            return None
        if used_mem[node_id] > ctx.caps_mem[node_id] + _EPS:
            return None

        migrated = ctx.x_prev.get((svc_id, var_id, node_id), 0.0) < 0.5
        if migrated:
            migration_count += 1
            cost_disruption += ctx.mig_cost.get(svc_id, 0.0)
            if migration_count > storm_max:
                return None

        cost_energy += ctx.e_cpu[node_id] * cpu + ctx.e_mem[node_id] * mem
        gain_accuracy += qa

    norm_c = cost_energy / ctx.c_max
    norm_d = cost_disruption / ctx.d_max
    norm_a = gain_accuracy / ctx.a_max
    objective_j = float(ctx.w_c * norm_c + ctx.w_d * norm_d - ctx.w_a * norm_a)
    return True, migration_count, objective_j


def _build_vectorized_decode_context(
    *,
    dataset: Any,
    decode_action_fn: Callable[[np.ndarray], dict[str, tuple[str, str]]],
    action_dims: Sequence[int],
    fast_ctx: _FastObjectiveContext,
) -> _VectorizedDecodeContext | None:
    n_heads = len(action_dims)
    if n_heads == 0:
        return None

    try:
        base_action = np.zeros((n_heads,), dtype=np.int64)
        base_placement = decode_action_fn(base_action)
    except Exception:
        return None

    service_ids = tuple(str(s) for s in base_placement.keys())
    if len(service_ids) != n_heads:
        return None

    node_ids = tuple(str(n.node_id) for n in getattr(dataset, "nodes", []))
    if not node_ids:
        return None
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    caps_cpu = np.asarray([fast_ctx.caps_cpu[nid] for nid in node_ids], dtype=np.float64)
    caps_mem = np.asarray([fast_ctx.caps_mem[nid] for nid in node_ids], dtype=np.float64)

    option_node_idx: list[np.ndarray] = []
    option_cpu: list[np.ndarray] = []
    option_mem: list[np.ndarray] = []
    option_acc: list[np.ndarray] = []
    option_energy: list[np.ndarray] = []
    option_disruption: list[np.ndarray] = []
    option_migration: list[np.ndarray] = []
    option_valid: list[np.ndarray] = []

    for head_idx, dim in enumerate(action_dims):
        d = int(dim)
        node_idx = np.full((d,), -1, dtype=np.int16)
        cpu = np.zeros((d,), dtype=np.float64)
        mem = np.zeros((d,), dtype=np.float64)
        acc = np.zeros((d,), dtype=np.float64)
        energy = np.zeros((d,), dtype=np.float64)
        disruption = np.zeros((d,), dtype=np.float64)
        migration = np.zeros((d,), dtype=np.int16)
        valid = np.zeros((d,), dtype=bool)
        svc_id = service_ids[head_idx]

        for val in range(d):
            action = np.zeros((n_heads,), dtype=np.int64)
            action[head_idx] = int(val)
            try:
                placement = decode_action_fn(action)
            except Exception:
                continue
            if svc_id not in placement:
                continue

            var, node = placement[svc_id]
            var_id = str(var)
            node_id = str(node)
            node_pos = node_to_idx.get(node_id)
            if node_pos is None:
                continue

            key = (svc_id, var_id)
            req_cpu = fast_ctx.r_req.get(key)
            req_mem = fast_ctx.r_mem.get(key)
            qa = fast_ctx.acc.get(key)
            if req_cpu is None or req_mem is None or qa is None:
                continue

            mig = 1 if fast_ctx.x_prev.get((svc_id, var_id, node_id), 0.0) < 0.5 else 0
            node_idx[val] = np.int16(node_pos)
            cpu[val] = float(req_cpu)
            mem[val] = float(req_mem)
            acc[val] = float(qa)
            energy[val] = float(fast_ctx.e_cpu[node_id] * req_cpu + fast_ctx.e_mem[node_id] * req_mem)
            disruption[val] = float(fast_ctx.mig_cost.get(svc_id, 0.0) * mig)
            migration[val] = np.int16(mig)
            valid[val] = True

        option_node_idx.append(node_idx)
        option_cpu.append(cpu)
        option_mem.append(mem)
        option_acc.append(acc)
        option_energy.append(energy)
        option_disruption.append(disruption)
        option_migration.append(migration)
        option_valid.append(valid)

    return _VectorizedDecodeContext(
        service_ids=service_ids,
        node_ids=node_ids,
        caps_cpu=caps_cpu,
        caps_mem=caps_mem,
        option_node_idx=tuple(option_node_idx),
        option_cpu=tuple(option_cpu),
        option_mem=tuple(option_mem),
        option_acc=tuple(option_acc),
        option_energy=tuple(option_energy),
        option_disruption=tuple(option_disruption),
        option_migration=tuple(option_migration),
        option_valid=tuple(option_valid),
    )


def _best_candidate_vectorized(
    *,
    vec_ctx: _VectorizedDecodeContext,
    head_logits: Sequence[np.ndarray],
    action_matrix: np.ndarray,
    storm_max: int,
    fast_ctx: _FastObjectiveContext,
) -> tuple[np.ndarray, int, float] | None:
    actions = np.asarray(action_matrix, dtype=np.int64)
    if actions.ndim != 2 or actions.shape[0] <= 0:
        return None

    n_candidates, n_heads = actions.shape
    valid = np.ones((n_candidates,), dtype=bool)
    migration_count = np.zeros((n_candidates,), dtype=np.int16)
    cost_energy = np.zeros((n_candidates,), dtype=np.float64)
    cost_disruption = np.zeros((n_candidates,), dtype=np.float64)
    gain_accuracy = np.zeros((n_candidates,), dtype=np.float64)
    logit_sum = np.zeros((n_candidates,), dtype=np.float64)

    nodes_per_head: list[np.ndarray] = []
    cpu_per_head: list[np.ndarray] = []
    mem_per_head: list[np.ndarray] = []

    for head_idx in range(n_heads):
        idx = actions[:, head_idx]
        head_valid = vec_ctx.option_valid[head_idx][idx]
        valid &= head_valid
        nodes = vec_ctx.option_node_idx[head_idx][idx]
        cpus = vec_ctx.option_cpu[head_idx][idx]
        mems = vec_ctx.option_mem[head_idx][idx]
        nodes_per_head.append(nodes)
        cpu_per_head.append(cpus)
        mem_per_head.append(mems)
        migration_count += vec_ctx.option_migration[head_idx][idx]
        cost_energy += vec_ctx.option_energy[head_idx][idx]
        cost_disruption += vec_ctx.option_disruption[head_idx][idx]
        gain_accuracy += vec_ctx.option_acc[head_idx][idx]
        logit_sum += np.take(np.asarray(head_logits[head_idx], dtype=np.float64), idx)

    if not np.any(valid):
        return None

    valid &= migration_count <= int(storm_max)
    if not np.any(valid):
        return None

    nodes_stack = np.stack(nodes_per_head, axis=1)
    cpu_stack = np.stack(cpu_per_head, axis=1)
    mem_stack = np.stack(mem_per_head, axis=1)
    for node_idx in range(len(vec_ctx.node_ids)):
        cpu_total = np.where(nodes_stack == node_idx, cpu_stack, 0.0).sum(axis=1)
        mem_total = np.where(nodes_stack == node_idx, mem_stack, 0.0).sum(axis=1)
        valid &= cpu_total <= vec_ctx.caps_cpu[node_idx] + _EPS
        valid &= mem_total <= vec_ctx.caps_mem[node_idx] + _EPS
        if not np.any(valid):
            return None

    objective_j = (
        fast_ctx.w_c * (cost_energy / fast_ctx.c_max)
        + fast_ctx.w_d * (cost_disruption / fast_ctx.d_max)
        - fast_ctx.w_a * (gain_accuracy / fast_ctx.a_max)
    )

    feasible_idx = np.flatnonzero(valid)
    if feasible_idx.size <= 0:
        return None
    rank = np.lexsort((-logit_sum[feasible_idx], objective_j[feasible_idx]))
    best_idx = int(feasible_idx[int(rank[0])])
    return (
        actions[best_idx].astype(np.int64, copy=False),
        int(migration_count[best_idx]),
        float(objective_j[best_idx]),
    )


def _best_candidate_python(
    *,
    dataset: Any,
    head_logits: Sequence[np.ndarray],
    candidates: Iterable[np.ndarray],
    decode_action_fn: Callable[[np.ndarray], dict[str, tuple[str, str]]],
    action_dims: Sequence[int],
    storm_max: int,
    fast_ctx: _FastObjectiveContext,
) -> tuple[Legacy44DecodeOutcome | None, int]:
    best: Legacy44DecodeOutcome | None = None
    best_key: tuple[float, float] | None = None
    checked = 0
    for action in candidates:
        checked += 1
        placement = decode_action_fn(action)
        fast_eval = _evaluate_fast(fast_ctx, placement, storm_max=storm_max)
        if fast_eval is None:
            continue
        _, mig, objective_j = fast_eval
        logit = float(sum(float(head_logits[i][int(action[i])]) for i in range(len(action_dims))))
        key = (objective_j, -logit)
        if best is None or best_key is None or key < best_key:
            best = Legacy44DecodeOutcome(
                action=np.asarray(action, dtype=np.int64),
                placement=placement,
                objective_j=objective_j,
                migration_count=mig,
                storm_max=storm_max,
                feasible=True,
                decoder_mode="legacy44_topk_stormsafe",
                fallback_full_enumeration=False,
                candidates_evaluated=checked,
            )
            best_key = key
    return best, checked


def _best_candidate_fullenum_pruned(
    *,
    vec_ctx: _VectorizedDecodeContext,
    head_logits: Sequence[np.ndarray],
    action_dims: Sequence[int],
    storm_max: int,
    fast_ctx: _FastObjectiveContext,
) -> tuple[np.ndarray, int, float] | None:
    n_heads = len(action_dims)
    n_nodes = len(vec_ctx.node_ids)
    if n_heads <= 0 or n_nodes <= 0:
        return None

    caps_cpu = vec_ctx.caps_cpu
    caps_mem = vec_ctx.caps_mem
    used_cpu = np.zeros((n_nodes,), dtype=np.float64)
    used_mem = np.zeros((n_nodes,), dtype=np.float64)
    action = np.zeros((n_heads,), dtype=np.int64)
    logits = [np.asarray(h, dtype=np.float64) for h in head_logits]

    best_key: tuple[float, float] | None = None
    best_action: np.ndarray | None = None
    best_mig = 0
    best_obj = float("inf")

    option_valid = vec_ctx.option_valid
    option_nodes = vec_ctx.option_node_idx
    option_cpu = vec_ctx.option_cpu
    option_mem = vec_ctx.option_mem
    option_acc = vec_ctx.option_acc
    option_energy = vec_ctx.option_energy
    option_disruption = vec_ctx.option_disruption
    option_migration = vec_ctx.option_migration

    def _dfs(
        depth: int,
        migration_count: int,
        cost_energy: float,
        cost_disruption: float,
        gain_accuracy: float,
        logit_sum: float,
    ) -> None:
        nonlocal best_action, best_key, best_mig, best_obj
        if depth >= n_heads:
            obj = (
                fast_ctx.w_c * (cost_energy / fast_ctx.c_max)
                + fast_ctx.w_d * (cost_disruption / fast_ctx.d_max)
                - fast_ctx.w_a * (gain_accuracy / fast_ctx.a_max)
            )
            key = (float(obj), -float(logit_sum))
            if best_key is None or key < best_key:
                best_key = key
                best_obj = float(obj)
                best_mig = int(migration_count)
                best_action = action.copy()
            return

        valid = option_valid[depth]
        nodes = option_nodes[depth]
        cpus = option_cpu[depth]
        mems = option_mem[depth]
        accs = option_acc[depth]
        energies = option_energy[depth]
        disruptions = option_disruption[depth]
        migrations = option_migration[depth]
        h_logits = logits[depth]
        dim = int(action_dims[depth])

        for opt in range(dim):
            if not bool(valid[opt]):
                continue
            node_idx = int(nodes[opt])
            if node_idx < 0 or node_idx >= n_nodes:
                continue
            cpu = float(cpus[opt])
            mem = float(mems[opt])
            if used_cpu[node_idx] + cpu > caps_cpu[node_idx] + _EPS:
                continue
            if used_mem[node_idx] + mem > caps_mem[node_idx] + _EPS:
                continue

            next_mig = int(migration_count) + int(migrations[opt])
            if next_mig > int(storm_max):
                continue

            action[depth] = int(opt)
            used_cpu[node_idx] += cpu
            used_mem[node_idx] += mem
            _dfs(
                depth + 1,
                next_mig,
                cost_energy + float(energies[opt]),
                cost_disruption + float(disruptions[opt]),
                gain_accuracy + float(accs[opt]),
                logit_sum + float(h_logits[opt]),
            )
            used_cpu[node_idx] -= cpu
            used_mem[node_idx] -= mem

    _dfs(0, 0, 0.0, 0.0, 0.0, 0.0)
    if best_action is None:
        return None
    return best_action, best_mig, best_obj


def _one_flip_neighborhood(action: np.ndarray, action_dims: Sequence[int]) -> np.ndarray:
    base = np.asarray(action, dtype=np.int64).reshape(-1)
    candidates = [base]
    for i, dim in enumerate(action_dims):
        d = int(dim)
        cur = int(base[i])
        for opt in range(d):
            if opt == cur:
                continue
            cand = base.copy()
            cand[i] = int(opt)
            candidates.append(cand)
    return np.asarray(candidates, dtype=np.int64)


def select_stormsafe_action(
    *,
    dataset: Any,
    head_logits: Sequence[np.ndarray],
    decode_action_fn: Callable[[np.ndarray], dict[str, tuple[str, str]]],
    objective_fn: Callable[[Any, dict[str, tuple[str, str]]], Any],
    action_dims: Sequence[int] = DEFAULT_ACTION_DIMS,
    topk_per_head: Sequence[int] = DEFAULT_TOPK_PER_HEAD,
) -> Legacy44DecodeOutcome:
    storm_max = int(getattr(dataset, "v_storm_max", 0))
    fast_ctx = _build_fast_objective_context(dataset)
    vec_ctx = _build_vectorized_decode_context(
        dataset=dataset,
        decode_action_fn=decode_action_fn,
        action_dims=action_dims,
        fast_ctx=fast_ctx,
    )

    topk_checked = 0
    full_checked = 0

    topk_choices: list[list[int]] = []
    for i, logits in enumerate(head_logits):
        k = int(topk_per_head[i]) if i < len(topk_per_head) else len(logits)
        topk_choices.append(_topk_indices_from_logits(np.asarray(logits, dtype=np.float64), k))

    if vec_ctx is not None:
        topk_actions = _candidate_matrix_from_choices(topk_choices)
        topk_checked = int(topk_actions.shape[0])
        best_vec = _best_candidate_vectorized(
            vec_ctx=vec_ctx,
            head_logits=head_logits,
            action_matrix=topk_actions,
            storm_max=storm_max,
            fast_ctx=fast_ctx,
        )
        if best_vec is not None:
            action, mig, objective_j = best_vec
            refine_actions = _one_flip_neighborhood(action, action_dims)
            refine_best = _best_candidate_vectorized(
                vec_ctx=vec_ctx,
                head_logits=head_logits,
                action_matrix=refine_actions,
                storm_max=storm_max,
                fast_ctx=fast_ctx,
            )
            if refine_best is not None:
                action, mig, objective_j = refine_best
                decoder_mode = "legacy44_topk_stormsafe_refine1"
                topk_checked += int(refine_actions.shape[0])
            else:
                decoder_mode = "legacy44_topk_stormsafe"
            return Legacy44DecodeOutcome(
                action=action,
                placement=decode_action_fn(action),
                objective_j=objective_j,
                migration_count=mig,
                storm_max=storm_max,
                feasible=True,
                decoder_mode=decoder_mode,
                fallback_full_enumeration=False,
                candidates_evaluated=topk_checked,
            )

        full_checked = int(np.prod(np.asarray(action_dims, dtype=np.int64)))
        best_full_vec = _best_candidate_fullenum_pruned(
            vec_ctx=vec_ctx,
            head_logits=head_logits,
            action_dims=action_dims,
            storm_max=storm_max,
            fast_ctx=fast_ctx,
        )
        if best_full_vec is not None:
            action, mig, objective_j = best_full_vec
            return Legacy44DecodeOutcome(
                action=action,
                placement=decode_action_fn(action),
                objective_j=objective_j,
                migration_count=mig,
                storm_max=storm_max,
                feasible=True,
                decoder_mode="legacy44_fullenum_stormsafe",
                fallback_full_enumeration=True,
                candidates_evaluated=full_checked,
            )
    else:
        topk_best, topk_checked = _best_candidate_python(
            dataset=dataset,
            head_logits=head_logits,
            candidates=_iter_topk_candidates(head_logits, topk_per_head),
            decode_action_fn=decode_action_fn,
            action_dims=action_dims,
            storm_max=storm_max,
            fast_ctx=fast_ctx,
        )
        if topk_best is not None:
            topk_best.candidates_evaluated = topk_checked
            return topk_best

        full_best, full_checked = _best_candidate_python(
            dataset=dataset,
            head_logits=head_logits,
            candidates=_iter_full_candidates(action_dims),
            decode_action_fn=decode_action_fn,
            action_dims=action_dims,
            storm_max=storm_max,
            fast_ctx=fast_ctx,
        )
        if full_best is not None:
            full_best.decoder_mode = "legacy44_fullenum_stormsafe"
            full_best.fallback_full_enumeration = True
            full_best.candidates_evaluated = full_checked
            return full_best

    greedy = np.asarray([int(np.argmax(h)) for h in head_logits], dtype=np.int64)
    placement = decode_action_fn(greedy)
    mig = _migration_count(dataset, placement)
    obj = float("inf")
    try:
        fast_eval = _evaluate_fast(fast_ctx, placement, storm_max=storm_max)
        if fast_eval is not None:
            _, mig, obj = fast_eval
    except Exception:
        pass
    return Legacy44DecodeOutcome(
        action=greedy,
        placement=placement,
        objective_j=obj,
        migration_count=mig,
        storm_max=storm_max,
        feasible=False,
        decoder_mode="legacy44_greedy_fallback",
        fallback_full_enumeration=True,
        candidates_evaluated=topk_checked + full_checked,
    )
