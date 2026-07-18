"""Dynamic DRL v2 offline pipeline: MILP-oracle data, BC bootstrap, and benchmark.

This module provides a size-agnostic (within configured envelope) sequential policy
that assigns each service to (variant, node) with hard action masking for:
- node CPU/RAM capacity
- migration storm budget

Workflow:
1) generate: create MILP-oracle episodes in JSONL
2) train:    behavioral cloning from oracle actions
3) benchmark:compare policy vs MILP on fresh random datasets
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.solver.dataset_generator import MILPDataset, generate_dataset
from src.solver.milp_model import solve_placement
from src.drl.reward import evaluate_objective_for_placement


@dataclass
class NodeSnapshot:
    node_id: str
    cap_cpu: float
    cap_mem_gb: float
    energy_cost: float
    e_mem_unit: float


@dataclass
class VariantSnapshot:
    variant_id: str
    cpu: float
    mem_gb: float
    acc: float


@dataclass
class ServiceSnapshot:
    service_id: str
    service_type: str
    migration_cost: float
    variants: list[VariantSnapshot]
    prev_variant: str
    prev_node: str


@dataclass
class EpisodeRecord:
    episode_id: str
    storm_max: int
    w_c: float
    w_d: float
    w_a: float
    nodes: list[NodeSnapshot]
    services: list[ServiceSnapshot]
    oracle_objective_j: float
    oracle_placement: dict[str, dict[str, str]]


def _normalize_weights(w_c: float, w_d: float, w_a: float) -> tuple[float, float, float]:
    s = max(w_c + w_d + w_a, 1e-9)
    return w_c / s, w_d / s, w_a / s


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


def _build_episode(ds: MILPDataset, objective_j: float, placement: dict[str, tuple[str, str]], episode_id: str) -> EpisodeRecord:
    prev = _extract_prev_placement(ds)
    nodes = [
        NodeSnapshot(
            node_id=n.node_id,
            cap_cpu=float(n.cap_cpu),
            cap_mem_gb=float(n.cap_mem_gb),
            energy_cost=float(n.energy_cost),
            e_mem_unit=float(n.e_mem_unit),
        )
        for n in ds.nodes
    ]
    services: list[ServiceSnapshot] = []
    for svc in ds.services:
        variants = [
            VariantSnapshot(
                variant_id=v,
                cpu=float(ds.r_req[(svc.service_id, v)]),
                mem_gb=float(ds.r_mem[(svc.service_id, v)]),
                acc=float(ds.acc[(svc.service_id, v)]),
            )
            for v in svc.valid_variants
        ]
        pvar, pnode = prev[svc.service_id]
        services.append(
            ServiceSnapshot(
                service_id=svc.service_id,
                service_type=svc.service_type,
                migration_cost=float(svc.migration_cost),
                variants=variants,
                prev_variant=pvar,
                prev_node=pnode,
            )
        )

    oracle_pl = {
        svc: {"variant": var, "node": node}
        for svc, (var, node) in placement.items()
    }

    return EpisodeRecord(
        episode_id=episode_id,
        storm_max=int(ds.v_storm_max),
        w_c=float(ds.w_c),
        w_d=float(ds.w_d),
        w_a=float(ds.w_a),
        nodes=nodes,
        services=services,
        oracle_objective_j=float(objective_j),
        oracle_placement=oracle_pl,
    )


def _episode_to_dict(ep: EpisodeRecord) -> dict[str, Any]:
    return {
        "episode_id": ep.episode_id,
        "storm_max": ep.storm_max,
        "weights": {"w_c": ep.w_c, "w_d": ep.w_d, "w_a": ep.w_a},
        "nodes": [
            {
                "node_id": n.node_id,
                "cap_cpu": n.cap_cpu,
                "cap_mem_gb": n.cap_mem_gb,
                "energy_cost": n.energy_cost,
                "e_mem_unit": n.e_mem_unit,
            }
            for n in ep.nodes
        ],
        "services": [
            {
                "service_id": s.service_id,
                "service_type": s.service_type,
                "migration_cost": s.migration_cost,
                "prev_variant": s.prev_variant,
                "prev_node": s.prev_node,
                "variants": [
                    {
                        "variant_id": v.variant_id,
                        "cpu": v.cpu,
                        "mem_gb": v.mem_gb,
                        "acc": v.acc,
                    }
                    for v in s.variants
                ],
            }
            for s in ep.services
        ],
        "oracle": {
            "objective_j": ep.oracle_objective_j,
            "placement": ep.oracle_placement,
        },
    }


def _dict_to_episode(d: dict[str, Any]) -> EpisodeRecord:
    w = d.get("weights", {})
    return EpisodeRecord(
        episode_id=str(d["episode_id"]),
        storm_max=int(d["storm_max"]),
        w_c=float(w.get("w_c", 0.15)),
        w_d=float(w.get("w_d", 0.10)),
        w_a=float(w.get("w_a", 0.75)),
        nodes=[
            NodeSnapshot(
                node_id=str(n["node_id"]),
                cap_cpu=float(n["cap_cpu"]),
                cap_mem_gb=float(n["cap_mem_gb"]),
                energy_cost=float(n["energy_cost"]),
                e_mem_unit=float(n.get("e_mem_unit", 0.0)),
            )
            for n in d["nodes"]
        ],
        services=[
            ServiceSnapshot(
                service_id=str(s["service_id"]),
                service_type=str(s.get("service_type", "unknown")),
                migration_cost=float(s["migration_cost"]),
                variants=[
                    VariantSnapshot(
                        variant_id=str(v["variant_id"]),
                        cpu=float(v["cpu"]),
                        mem_gb=float(v["mem_gb"]),
                        acc=float(v["acc"]),
                    )
                    for v in s["variants"]
                ],
                prev_variant=str(s["prev_variant"]),
                prev_node=str(s["prev_node"]),
            )
            for s in d["services"]
        ],
        oracle_objective_j=float(d["oracle"]["objective_j"]),
        oracle_placement={
            str(k): {"variant": str(v["variant"]), "node": str(v["node"])}
            for k, v in d["oracle"]["placement"].items()
        },
    )


def generate_oracle_dataset(
    output_path: Path,
    samples: int,
    min_nodes: int,
    max_nodes: int,
    min_services: int,
    max_services: int,
    seed: int,
    milp_time_limit: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    attempted = 0
    with output_path.open("w", encoding="utf-8") as f:
        while generated < samples and attempted < samples * 12:
            attempted += 1
            n_nodes = rng.randint(min_nodes, max_nodes)
            n_services = rng.randint(min_services, max_services)

            profile = rng.choice(["balanced", "energy", "quality", "storm"])
            if profile == "energy":
                w_c, w_d, w_a = _normalize_weights(rng.uniform(0.45, 0.65), rng.uniform(0.05, 0.15), rng.uniform(0.25, 0.45))
            elif profile == "quality":
                w_c, w_d, w_a = _normalize_weights(rng.uniform(0.05, 0.15), rng.uniform(0.04, 0.12), rng.uniform(0.75, 0.90))
            elif profile == "storm":
                w_c, w_d, w_a = _normalize_weights(rng.uniform(0.10, 0.22), rng.uniform(0.25, 0.50), rng.uniform(0.35, 0.60))
            else:
                w_c, w_d, w_a = _normalize_weights(rng.uniform(0.12, 0.25), rng.uniform(0.08, 0.20), rng.uniform(0.55, 0.80))

            ds = generate_dataset(
                num_nodes=n_nodes,
                num_services=n_services,
                seed=seed + attempted,
                v_storm_max=rng.randint(2, max(2, min(6, n_services))),
                w_c=w_c,
                w_d=w_d,
                w_a=w_a,
                theta_max=rng.uniform(1.15, 1.35),
            )
            try:
                result = solve_placement(ds, time_limit=milp_time_limit, verbose=False)
            except Exception:
                continue
            if result is None or result.status not in {"optimal", "feasible"}:
                continue

            episode_id = f"ep_{generated:06d}"
            ep = _build_episode(ds, result.objective_value, result.placement, episode_id)
            f.write(json.dumps(_episode_to_dict(ep), ensure_ascii=True) + "\n")
            generated += 1

    return {
        "output": str(output_path),
        "requested": samples,
        "generated": generated,
        "attempted": attempted,
    }


@dataclass
class FeatureConfig:
    max_nodes: int
    max_services: int
    max_variants: int
    input_dim: int
    action_dim: int


def _build_feature_config(episodes: list[EpisodeRecord]) -> FeatureConfig:
    max_nodes = max(len(ep.nodes) for ep in episodes)
    max_services = max(len(ep.services) for ep in episodes)
    max_variants = max(max(len(s.variants) for s in ep.services) for ep in episodes)
    # global(7) + nodes(max_nodes*9) + current_service(max_variants*4 + 5)
    input_dim = 7 + max_nodes * 9 + max_variants * 4 + 5
    action_dim = max_nodes * max_variants
    return FeatureConfig(max_nodes=max_nodes, max_services=max_services, max_variants=max_variants, input_dim=input_dim, action_dim=action_dim)


def _placement_dict_from_episode(ep: EpisodeRecord) -> dict[str, tuple[str, str]]:
    return {
        s.service_id: (s.prev_variant, s.prev_node)
        for s in ep.services
    }


def _build_index_maps(ep: EpisodeRecord) -> tuple[dict[str, int], dict[str, int]]:
    node_idx = {n.node_id: i for i, n in enumerate(ep.nodes)}
    svc_idx = {s.service_id: i for i, s in enumerate(ep.services)}
    return node_idx, svc_idx


def _action_index(variant_idx: int, node_idx: int, max_nodes: int) -> int:
    return variant_idx * max_nodes + node_idx


def _decode_action_index(action_idx: int, max_nodes: int) -> tuple[int, int]:
    return action_idx // max_nodes, action_idx % max_nodes


def _build_mask_for_service(
    ep: EpisodeRecord,
    service: ServiceSnapshot,
    residual_cpu: list[float],
    residual_mem: list[float],
    migration_used: int,
    cfg: FeatureConfig,
) -> np.ndarray:
    mask = np.zeros((cfg.action_dim,), dtype=np.float32)
    node_index, _ = _build_index_maps(ep)

    for vi, var in enumerate(service.variants):
        for ni, node in enumerate(ep.nodes):
            delta_mig = 0
            if service.prev_variant != var.variant_id or service.prev_node != node.node_id:
                delta_mig = 1
            if migration_used + delta_mig > ep.storm_max:
                continue
            if residual_cpu[ni] + 1e-9 < var.cpu:
                continue
            if residual_mem[ni] + 1e-9 < var.mem_gb:
                continue
            mask[_action_index(vi, ni, cfg.max_nodes)] = 1.0

    # Fallback safety: allow staying put when available in action space
    if float(mask.sum()) <= 0.0:
        prev_node_idx = node_index.get(service.prev_node, 0)
        prev_variant_idx = 0
        for i, v in enumerate(service.variants):
            if v.variant_id == service.prev_variant:
                prev_variant_idx = i
                break
        mask[_action_index(prev_variant_idx, prev_node_idx, cfg.max_nodes)] = 1.0

    return mask


def _vectorize_state_for_service(
    ep: EpisodeRecord,
    service: ServiceSnapshot,
    step_idx: int,
    residual_cpu: list[float],
    residual_mem: list[float],
    migration_used: int,
    cfg: FeatureConfig,
) -> np.ndarray:
    max_cap_cpu = max(n.cap_cpu for n in ep.nodes)
    max_cap_mem = max(n.cap_mem_gb for n in ep.nodes)
    max_energy = max(n.energy_cost for n in ep.nodes)
    max_mem_energy = max(max(n.e_mem_unit for n in ep.nodes), 1e-6)
    max_mig_cost = max(s.migration_cost for s in ep.services)

    feats: list[float] = []

    feats.extend([
        len(ep.nodes) / max(cfg.max_nodes, 1),
        len(ep.services) / max(cfg.max_services, 1),
        ep.w_c,
        ep.w_d,
        ep.w_a,
        (ep.storm_max - migration_used) / max(ep.storm_max, 1),
        step_idx / max(len(ep.services) - 1, 1),
    ])

    for i in range(cfg.max_nodes):
        if i < len(ep.nodes):
            n = ep.nodes[i]
            used_cpu = max(0.0, n.cap_cpu - residual_cpu[i])
            used_mem = max(0.0, n.cap_mem_gb - residual_mem[i])
            feats.extend([
                n.cap_cpu / max(max_cap_cpu, 1e-6),
                n.cap_mem_gb / max(max_cap_mem, 1e-6),
                n.energy_cost / max(max_energy, 1e-6),
                n.e_mem_unit / max(max_mem_energy, 1e-6),
                residual_cpu[i] / max(n.cap_cpu, 1e-6),
                residual_mem[i] / max(n.cap_mem_gb, 1e-6),
                used_cpu / max(n.cap_cpu, 1e-6),
                used_mem / max(n.cap_mem_gb, 1e-6),
                1.0,
            ])
        else:
            feats.extend([0.0] * 9)

    prev_node_norm = 0.0
    for ni, node in enumerate(ep.nodes):
        if node.node_id == service.prev_node:
            prev_node_norm = ni / max(len(ep.nodes) - 1, 1)
            break

    for i in range(cfg.max_variants):
        if i < len(service.variants):
            v = service.variants[i]
            is_prev = 1.0 if v.variant_id == service.prev_variant else 0.0
            feats.extend([
                v.cpu / max(max_cap_cpu, 1e-6),
                v.mem_gb / max(max_cap_mem, 1e-6),
                v.acc,
                is_prev,
            ])
        else:
            feats.extend([0.0] * 4)

    feats.extend([
        service.migration_cost / max(max_mig_cost, 1e-6),
        1.0 if len(service.variants) > 1 else 0.0,
        prev_node_norm,
        1.0 if service.service_type in {"detection", "gen_ai"} else 0.0,
        len(service.variants) / max(cfg.max_variants, 1),
    ])

    return np.asarray(feats, dtype=np.float32)


def _oracle_action_index(ep: EpisodeRecord, service: ServiceSnapshot, cfg: FeatureConfig) -> int:
    node_map, _ = _build_index_maps(ep)
    target = ep.oracle_placement[service.service_id]
    target_variant = target["variant"]
    target_node = target["node"]

    vi = 0
    for i, v in enumerate(service.variants):
        if v.variant_id == target_variant:
            vi = i
            break
    ni = node_map[target_node]
    return _action_index(vi, ni, cfg.max_nodes)


@dataclass
class StepSample:
    state: np.ndarray
    mask: np.ndarray
    target: int


def _build_step_samples(ep: EpisodeRecord, cfg: FeatureConfig) -> list[StepSample]:
    # Stable order keeps training deterministic and simple.
    services = sorted(ep.services, key=lambda s: s.service_id)

    residual_cpu = [n.cap_cpu for n in ep.nodes]
    residual_mem = [n.cap_mem_gb for n in ep.nodes]
    migration_used = 0

    steps: list[StepSample] = []
    node_map, _ = _build_index_maps(ep)

    for step_idx, svc in enumerate(services):
        mask = _build_mask_for_service(ep, svc, residual_cpu, residual_mem, migration_used, cfg)
        state_vec = _vectorize_state_for_service(ep, svc, step_idx, residual_cpu, residual_mem, migration_used, cfg)
        target_idx = _oracle_action_index(ep, svc, cfg)

        # Ensure training target is always selectable.
        if mask[target_idx] < 0.5:
            mask[target_idx] = 1.0

        steps.append(StepSample(state=state_vec, mask=mask, target=target_idx))

        target = ep.oracle_placement[svc.service_id]
        chosen_variant = target["variant"]
        chosen_node = target["node"]

        chosen_variant_obj = None
        for v in svc.variants:
            if v.variant_id == chosen_variant:
                chosen_variant_obj = v
                break
        if chosen_variant_obj is None:
            chosen_variant_obj = svc.variants[0]

        ni = node_map[chosen_node]
        residual_cpu[ni] -= chosen_variant_obj.cpu
        residual_mem[ni] -= chosen_variant_obj.mem_gb

        if svc.prev_variant != chosen_variant or svc.prev_node != chosen_node:
            migration_used += 1

    return steps


class StepDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, steps: list[StepSample]) -> None:
        self.states = [torch.tensor(s.state, dtype=torch.float32) for s in steps]
        self.masks = [torch.tensor(s.mask, dtype=torch.float32) for s in steps]
        self.targets = [torch.tensor(s.target, dtype=torch.long) for s in steps]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.masks[idx], self.targets[idx]


class DynamicSequentialPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _load_episodes(path: Path) -> list[EpisodeRecord]:
    eps: list[EpisodeRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eps.append(_dict_to_episode(json.loads(line)))
    return eps


def _split_episodes(episodes: list[EpisodeRecord], seed: int, val_ratio: float = 0.2) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def train_bc(
    dataset_path: Path,
    output_model: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    episodes = _load_episodes(dataset_path)
    if len(episodes) < 10:
        raise ValueError("Need at least 10 episodes for training.")

    cfg = _build_feature_config(episodes)
    train_eps, val_eps = _split_episodes(episodes, seed=seed)
    train_steps = [st for ep in train_eps for st in _build_step_samples(ep, cfg)]
    val_steps = [st for ep in val_eps for st in _build_step_samples(ep, cfg)]

    train_loader = DataLoader(StepDataset(train_steps), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StepDataset(val_steps), batch_size=batch_size, shuffle=False)

    model = DynamicSequentialPolicy(cfg.input_dim, cfg.action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses: list[float] = []
        for states, masks, targets in train_loader:
            logits = model(states)
            masked_logits = logits.masked_fill(masks <= 0.5, -1e9)
            loss = criterion(masked_logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        correct = 0
        total = 0
        with torch.no_grad():
            for states, masks, targets in val_loader:
                logits = model(states)
                masked_logits = logits.masked_fill(masks <= 0.5, -1e9)
                loss = criterion(masked_logits, targets)
                val_losses.append(float(loss.detach().cpu().item()))
                pred = torch.argmax(masked_logits, dim=1)
                correct += int((pred == targets).sum().item())
                total += int(targets.numel())

        tr = float(np.mean(tr_losses)) if tr_losses else math.inf
        vl = float(np.mean(val_losses)) if val_losses else math.inf
        acc = (correct / total) if total else 0.0
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": vl, "val_acc": acc})

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": best_state,
        "feature_config": {
            "max_nodes": cfg.max_nodes,
            "max_services": cfg.max_services,
            "max_variants": cfg.max_variants,
            "input_dim": cfg.input_dim,
            "action_dim": cfg.action_dim,
        },
        "history": history,
    }
    torch.save(payload, output_model)

    return {
        "output_model": str(output_model),
        "episodes": len(episodes),
        "train_steps": len(train_steps),
        "val_steps": len(val_steps),
        "best_val_loss": best_val,
        "final_val_acc": history[-1]["val_acc"],
    }


def _load_model_checkpoint(path: Path) -> tuple[DynamicSequentialPolicy, FeatureConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    fc = ckpt["feature_config"]
    cfg = FeatureConfig(
        max_nodes=int(fc["max_nodes"]),
        max_services=int(fc["max_services"]),
        max_variants=int(fc["max_variants"]),
        input_dim=int(fc["input_dim"]),
        action_dim=int(fc["action_dim"]),
    )
    model = DynamicSequentialPolicy(cfg.input_dim, cfg.action_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


class DynamicPlacementEnv(gym.Env):
    """Sequential placement environment for PPO fine-tuning on dynamic episodes."""

    metadata = {"render_modes": []}

    def __init__(self, episodes: list[EpisodeRecord], cfg: FeatureConfig, seed: int = 42) -> None:
        super().__init__()
        self.episodes = episodes
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.action_space = gym.spaces.Discrete(cfg.action_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(cfg.input_dim,),
            dtype=np.float32,
        )
        self._ep: EpisodeRecord | None = None
        self._services: list[ServiceSnapshot] = []
        self._step_idx = 0
        self._residual_cpu: list[float] = []
        self._residual_mem: list[float] = []
        self._migration_used = 0
        self._placement: dict[str, tuple[str, str]] = {}
        self._last_mask = np.ones((cfg.action_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        self._ep = self.rng.choice(self.episodes)
        self._services = sorted(self._ep.services, key=lambda s: s.service_id)
        self._step_idx = 0
        self._residual_cpu = [n.cap_cpu for n in self._ep.nodes]
        self._residual_mem = [n.cap_mem_gb for n in self._ep.nodes]
        self._migration_used = 0
        self._placement = {}
        obs = self._build_obs()
        return obs, {}

    def action_masks(self) -> np.ndarray:
        if self._ep is None or self._step_idx >= len(self._services):
            return np.ones((self.cfg.action_dim,), dtype=np.float32)
        svc = self._services[self._step_idx]
        self._last_mask = _build_mask_for_service(
            self._ep,
            svc,
            self._residual_cpu,
            self._residual_mem,
            self._migration_used,
            self.cfg,
        )
        return self._last_mask

    def _build_obs(self) -> np.ndarray:
        if self._ep is None:
            return np.zeros((self.cfg.input_dim,), dtype=np.float32)
        if self._step_idx >= len(self._services):
            return np.zeros((self.cfg.input_dim,), dtype=np.float32)
        svc = self._services[self._step_idx]
        return _vectorize_state_for_service(
            self._ep,
            svc,
            self._step_idx,
            self._residual_cpu,
            self._residual_mem,
            self._migration_used,
            self.cfg,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._ep is None:
            raise RuntimeError("reset() must be called before step().")

        done = self._step_idx >= len(self._services)
        if done:
            return np.zeros((self.cfg.input_dim,), dtype=np.float32), 0.0, True, False, {}

        reward = 0.0
        mask = self.action_masks()
        if action < 0 or action >= self.cfg.action_dim or mask[action] < 0.5:
            # Keep a small penalty for invalid picks; then snap to best valid action.
            reward -= 0.2
            valid = np.where(mask > 0.5)[0]
            action = int(valid[0]) if len(valid) else 0

        svc = self._services[self._step_idx]
        vi, ni = _decode_action_index(int(action), self.cfg.max_nodes)
        if ni >= len(self._ep.nodes) or vi >= len(svc.variants):
            chosen_variant = svc.prev_variant
            chosen_node = svc.prev_node
        else:
            chosen_variant = svc.variants[vi].variant_id
            chosen_node = self._ep.nodes[ni].node_id

        self._placement[svc.service_id] = (chosen_variant, chosen_node)
        node_map, _ = _build_index_maps(self._ep)
        node_idx = node_map.get(chosen_node, 0)
        var_obj = next((v for v in svc.variants if v.variant_id == chosen_variant), svc.variants[0])
        self._residual_cpu[node_idx] -= var_obj.cpu
        self._residual_mem[node_idx] -= var_obj.mem_gb

        migrated = int(svc.prev_variant != chosen_variant or svc.prev_node != chosen_node)
        self._migration_used += migrated
        reward -= 0.01 * migrated
        oracle_t = self._ep.oracle_placement.get(svc.service_id, {})
        if (
            oracle_t.get("variant") == chosen_variant
            and oracle_t.get("node") == chosen_node
        ):
            reward += 0.05
        else:
            reward -= 0.02
        self._step_idx += 1

        terminated = self._step_idx >= len(self._services)
        info: dict[str, Any] = {}
        if terminated:
            ds = _dataset_from_episode(self._ep)
            final_obj = float(evaluate_objective_for_placement(ds, self._placement).objective_j)
            delta_j = final_obj - self._ep.oracle_objective_j
            reward += max(-5.0, -50.0 * abs(delta_j))
            info = {
                "final_objective_j": final_obj,
                "oracle_objective_j": self._ep.oracle_objective_j,
                "delta_j": delta_j,
                "storm_ok": self._migration_used <= self._ep.storm_max,
                "migrations": self._migration_used,
            }

        obs = self._build_obs()
        return obs, float(reward), terminated, False, info


def _mask_fn(env: DynamicPlacementEnv) -> np.ndarray:
    return env.action_masks()


def _apply_bc_warmstart_to_ppo(ppo_model: MaskablePPO, bc_ckpt_path: Path) -> int:
    """Copy BC backbone/head weights into PPO policy/action nets when shape-compatible."""
    ckpt = torch.load(bc_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    transferred = 0
    policy_net = ppo_model.policy.mlp_extractor.policy_net
    mapping = {
        "backbone.0.weight": (0, "weight"),
        "backbone.0.bias": (0, "bias"),
        "backbone.2.weight": (2, "weight"),
        "backbone.2.bias": (2, "bias"),
    }
    with torch.no_grad():
        for bc_key, (layer_idx, attr) in mapping.items():
            if bc_key not in sd:
                continue
            t = sd[bc_key]
            p = getattr(policy_net[layer_idx], attr)
            if p.shape == t.shape:
                p.copy_(t)
                transferred += 1

        if "head.weight" in sd and hasattr(ppo_model.policy, "action_net"):
            t = sd["head.weight"]
            p = ppo_model.policy.action_net.weight
            if p.shape == t.shape:
                p.copy_(t)
                transferred += 1
        if "head.bias" in sd and hasattr(ppo_model.policy, "action_net"):
            t = sd["head.bias"]
            p = ppo_model.policy.action_net.bias
            if p.shape == t.shape:
                p.copy_(t)
                transferred += 1

        # Also warm-start value net from shared backbone if possible.
        value_net = ppo_model.policy.mlp_extractor.value_net
        for bc_key, (layer_idx, attr) in mapping.items():
            if bc_key not in sd:
                continue
            t = sd[bc_key]
            p = getattr(value_net[layer_idx], attr)
            if p.shape == t.shape:
                p.copy_(t)
    return transferred


def train_ppo_from_bc(
    dataset_path: Path,
    bc_model_path: Path,
    output_model_path: Path,
    timesteps: int,
    learning_rate: float,
    seed: int,
    eval_every: int = 5000,
    eval_cycles: int = 40,
    min_nodes: int = 4,
    max_nodes: int = 8,
    min_services: int = 6,
    max_services: int = 20,
    milp_time_limit: int = 20,
    eval_decode_mode: str = "greedy",
    beam_width: int = 16,
    topk_actions: int = 6,
) -> dict[str, Any]:
    episodes = _load_episodes(dataset_path)
    if len(episodes) < 10:
        raise ValueError("Need at least 10 episodes for PPO training.")
    cfg = _build_feature_config(episodes)

    def env_fn() -> Monitor:
        env = DynamicPlacementEnv(episodes=episodes, cfg=cfg, seed=seed)
        env = ActionMasker(env, _mask_fn)
        return Monitor(env)

    vec_env = DummyVecEnv([env_fn])
    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.0,
        target_kl=0.01,
        verbose=0,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )
    transferred = _apply_bc_warmstart_to_ppo(model, bc_model_path)

    # Evaluate initial warm-start checkpoint before any PPO update.
    best_summary, _ = _benchmark_ppo_instance(
        ppo_model=model,
        cfg=cfg,
        cycles=eval_cycles,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        min_services=min_services,
        max_services=max_services,
        seed=seed + 7000,
        milp_time_limit=milp_time_limit,
        decode_mode=eval_decode_mode,
        beam_width=beam_width,
        topk_actions=topk_actions,
    )
    best_score = float(best_summary["avg_delta_j"])
    best_params = model.get_parameters()
    history: list[dict[str, Any]] = [{"step": 0, **best_summary}]

    trained = 0
    while trained < timesteps:
        chunk = min(eval_every, timesteps - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        trained += chunk
        cur_summary, _ = _benchmark_ppo_instance(
            ppo_model=model,
            cfg=cfg,
            cycles=eval_cycles,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            min_services=min_services,
            max_services=max_services,
            seed=seed + 7000 + trained,
            milp_time_limit=milp_time_limit,
            decode_mode=eval_decode_mode,
            beam_width=beam_width,
            topk_actions=topk_actions,
        )
        history.append({"step": trained, **cur_summary})
        cur_score = float(cur_summary["avg_delta_j"])
        if cur_score < best_score:
            best_score = cur_score
            best_params = model.get_parameters()

    model.set_parameters(best_params)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path))
    return {
        "output_model": str(output_model_path),
        "timesteps": timesteps,
        "episodes": len(episodes),
        "warmstart_tensors": transferred,
        "best_eval_avg_delta_j": best_score,
        "eval_decode_mode": _normalize_decode_mode(eval_decode_mode),
        "eval_history": history,
    }


def _ppo_action_probs(
    ppo_model: MaskablePPO,
    obs: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    obs_t, _ = ppo_model.policy.obs_to_tensor(obs)
    dist = ppo_model.policy.get_distribution(
        obs_t,
        action_masks=mask.reshape(1, -1).astype(bool),
    )
    probs = dist.distribution.probs.squeeze(0).detach().cpu().numpy()
    return probs.astype(np.float64)


def _ppo_predict_action_index(
    ppo_model: MaskablePPO,
    obs: np.ndarray,
    mask: np.ndarray,
) -> int | None:
    valid_idx = np.where(mask > 0.5)[0]
    if len(valid_idx) <= 0:
        return None
    try:
        action, _ = ppo_model.predict(obs, deterministic=True, action_masks=mask.astype(bool))
        action_idx = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
        if mask[action_idx] > 0.5:
            return action_idx
    except Exception:
        pass
    probs = _ppo_action_probs(ppo_model, obs, mask)
    masked_probs = np.where(mask > 0.5, probs, -np.inf)
    return int(np.argmax(masked_probs))


def _normalize_decode_mode(decode_mode: str) -> str:
    mode = str(decode_mode).strip().lower()
    if mode in {"greedy", "beam"}:
        return mode
    return "beam"


def _sequential_infer_placement_ppo(
    ep: EpisodeRecord,
    ppo_model: MaskablePPO,
    cfg: FeatureConfig,
    decode_mode: str = "beam",
    beam_width: int = 16,
    topk_actions: int = 6,
) -> tuple[dict[str, tuple[str, str]], int, dict[str, Any]]:
    services = sorted(ep.services, key=lambda s: s.service_id)
    node_map, _ = _build_index_maps(ep)
    reward_ds = _dataset_from_episode(ep)
    mode = _normalize_decode_mode(decode_mode)
    policy_inference_ms = 0.0
    policy_calls = 0

    if mode == "greedy":
        residual_cpu = [n.cap_cpu for n in ep.nodes]
        residual_mem = [n.cap_mem_gb for n in ep.nodes]
        migration_used = 0
        placement: dict[str, tuple[str, str]] = {}
        for step_idx, svc in enumerate(services):
            mask = _build_mask_for_service(ep, svc, residual_cpu, residual_mem, migration_used, cfg)
            obs = _vectorize_state_for_service(ep, svc, step_idx, residual_cpu, residual_mem, migration_used, cfg)
            t_policy = time.perf_counter()
            action_idx = _ppo_predict_action_index(ppo_model, obs, mask)
            policy_inference_ms += (time.perf_counter() - t_policy) * 1000.0
            policy_calls += 1
            if action_idx is None:
                chosen_variant = svc.prev_variant
                chosen_node = svc.prev_node
            else:
                vi, ni = _decode_action_index(action_idx, cfg.max_nodes)
                if ni >= len(ep.nodes) or vi >= len(svc.variants):
                    chosen_variant = svc.prev_variant
                    chosen_node = svc.prev_node
                else:
                    chosen_var = svc.variants[vi]
                    chosen_variant = chosen_var.variant_id
                    chosen_node = ep.nodes[ni].node_id
            placement[svc.service_id] = (chosen_variant, chosen_node)
            var_obj = next((v for v in svc.variants if v.variant_id == chosen_variant), svc.variants[0])
            node_idx = node_map.get(chosen_node, 0)
            residual_cpu[node_idx] -= var_obj.cpu
            residual_mem[node_idx] -= var_obj.mem_gb
            if svc.prev_variant != chosen_variant or svc.prev_node != chosen_node:
                migration_used += 1
        for svc in services:
            if svc.service_id not in placement:
                placement[svc.service_id] = (svc.prev_variant, svc.prev_node)
        timing = {
            "decode_mode": mode,
            "policy_inference_ms": float(policy_inference_ms),
            "policy_calls": int(policy_calls),
        }
        return placement, migration_used, timing

    @dataclass
    class BeamState:
        placement: dict[str, tuple[str, str]]
        residual_cpu: list[float]
        residual_mem: list[float]
        migration_used: int
        logp: float

    beams = [
        BeamState(
            placement={},
            residual_cpu=[n.cap_cpu for n in ep.nodes],
            residual_mem=[n.cap_mem_gb for n in ep.nodes],
            migration_used=0,
            logp=0.0,
        )
    ]

    for step_idx, svc in enumerate(services):
        expanded: list[BeamState] = []
        for b in beams:
            mask = _build_mask_for_service(ep, svc, b.residual_cpu, b.residual_mem, b.migration_used, cfg)
            obs = _vectorize_state_for_service(ep, svc, step_idx, b.residual_cpu, b.residual_mem, b.migration_used, cfg)
            t_policy = time.perf_counter()
            probs = _ppo_action_probs(ppo_model, obs, mask)
            policy_inference_ms += (time.perf_counter() - t_policy) * 1000.0
            policy_calls += 1
            valid_count = int((mask > 0.5).sum())
            if valid_count <= 0:
                continue
            k = max(1, min(topk_actions, valid_count))
            top_idx = np.argpartition(-probs, k - 1)[:k]
            top_idx = top_idx[np.argsort(-probs[top_idx])]

            for action_idx in top_idx.tolist():
                if mask[action_idx] < 0.5:
                    continue
                vi, ni = _decode_action_index(int(action_idx), cfg.max_nodes)
                if ni >= len(ep.nodes) or vi >= len(svc.variants):
                    continue
                chosen_var = svc.variants[vi]
                chosen_node = ep.nodes[ni].node_id
                nxt = BeamState(
                    placement=dict(b.placement),
                    residual_cpu=list(b.residual_cpu),
                    residual_mem=list(b.residual_mem),
                    migration_used=b.migration_used,
                    logp=b.logp + math.log(max(float(probs[action_idx]), 1e-12)),
                )
                nxt.placement[svc.service_id] = (chosen_var.variant_id, chosen_node)
                nxt.residual_cpu[ni] -= chosen_var.cpu
                nxt.residual_mem[ni] -= chosen_var.mem_gb
                if svc.prev_variant != chosen_var.variant_id or svc.prev_node != chosen_node:
                    nxt.migration_used += 1
                expanded.append(nxt)

        if not expanded:
            break
        expanded.sort(key=lambda x: x.logp, reverse=True)
        beams = expanded[: max(1, beam_width)]

    for b in beams:
        for svc in services:
            if svc.service_id not in b.placement:
                b.placement[svc.service_id] = (svc.prev_variant, svc.prev_node)

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
        placement = {svc.service_id: (svc.prev_variant, svc.prev_node) for svc in services}
        timing = {
            "decode_mode": mode,
            "policy_inference_ms": float(policy_inference_ms),
            "policy_calls": int(policy_calls),
        }
        return placement, 0, timing

    migration_used = 0
    for svc in services:
        var, node = best.placement[svc.service_id]
        if svc.prev_variant != var or svc.prev_node != node:
            migration_used += 1
    timing = {
        "decode_mode": mode,
        "policy_inference_ms": float(policy_inference_ms),
        "policy_calls": int(policy_calls),
    }
    return best.placement, migration_used, timing


def _benchmark_ppo_instance(
    ppo_model: MaskablePPO,
    cfg: FeatureConfig,
    cycles: int,
    min_nodes: int,
    max_nodes: int,
    min_services: int,
    max_services: int,
    seed: int,
    milp_time_limit: int,
    decode_mode: str = "beam",
    beam_width: int = 16,
    topk_actions: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for cycle in range(1, cycles + 1):
        n_nodes = rng.randint(min_nodes, max_nodes)
        n_services = rng.randint(min_services, max_services)
        w_c, w_d, w_a = _normalize_weights(
            rng.uniform(0.1, 0.6), rng.uniform(0.05, 0.4), rng.uniform(0.2, 0.9)
        )
        ds = generate_dataset(
            num_nodes=n_nodes,
            num_services=n_services,
            seed=seed + cycle,
            v_storm_max=rng.randint(2, max(2, min(6, n_services))),
            w_c=w_c,
            w_d=w_d,
            w_a=w_a,
            theta_max=rng.uniform(1.15, 1.35),
        )
        try:
            milp_result = solve_placement(ds, time_limit=milp_time_limit, verbose=False)
        except Exception:
            continue
        if milp_result is None:
            continue
        ep = _build_episode(ds, milp_result.objective_value, milp_result.placement, f"benchppo_{cycle:06d}")
        t_decision = time.perf_counter()
        drl_placement, migration_used, timing = _sequential_infer_placement_ppo(
            ep,
            ppo_model,
            cfg,
            decode_mode=decode_mode,
            beam_width=beam_width,
            topk_actions=topk_actions,
        )
        decision_time_ms = (time.perf_counter() - t_decision) * 1000.0
        drl_obj = evaluate_objective_for_placement(ds, drl_placement).objective_j
        milp_obj = milp_result.objective_value
        delta_j = drl_obj - milp_obj
        rows.append(
            {
                "cycle": cycle,
                "milp_J": milp_obj,
                "drl_J": drl_obj,
                "delta_J": delta_j,
                "gap_pct": (delta_j / (abs(milp_obj) + 1e-8)) * 100.0,
                "storm_ok": migration_used <= ds.v_storm_max,
                "drl_migrations": migration_used,
                "storm_max": ds.v_storm_max,
                "decode_mode": timing["decode_mode"],
                "policy_inference_ms": float(timing["policy_inference_ms"]),
                "policy_calls": int(timing["policy_calls"]),
                "decision_time_ms": float(decision_time_ms),
            }
        )
    if not rows:
        raise RuntimeError("No valid benchmark rows produced.")
    avg_delta_j = float(np.mean([r["delta_J"] for r in rows]))
    summary = {
        "rows": len(rows),
        "avg_delta_j": avg_delta_j,
        "p95_delta_j": float(np.percentile([r["delta_J"] for r in rows], 95)),
        "avg_gap_pct_points": avg_delta_j * 100.0,
        "avg_gap_pct": float(np.mean([r["gap_pct"] for r in rows])),
        "p95_gap_pct": float(np.percentile([r["gap_pct"] for r in rows], 95)),
        "storm_violation_rate": float(np.mean([0.0 if r["storm_ok"] else 1.0 for r in rows])),
        "decode_mode": _normalize_decode_mode(decode_mode),
        "avg_policy_inference_ms": float(np.mean([r["policy_inference_ms"] for r in rows])),
        "p50_policy_inference_ms": float(np.percentile([r["policy_inference_ms"] for r in rows], 50)),
        "p95_policy_inference_ms": float(np.percentile([r["policy_inference_ms"] for r in rows], 95)),
        "avg_decision_time_ms": float(np.mean([r["decision_time_ms"] for r in rows])),
        "p50_decision_time_ms": float(np.percentile([r["decision_time_ms"] for r in rows], 50)),
        "p95_decision_time_ms": float(np.percentile([r["decision_time_ms"] for r in rows], 95)),
        "accept_gap_le_2pct_points": avg_delta_j <= 0.02,
        "accept_zero_storm_violation": float(np.mean([0.0 if r["storm_ok"] else 1.0 for r in rows])) <= 0.0,
    }
    return summary, rows


def benchmark_ppo_model(
    ppo_model_path: Path,
    dataset_path: Path,
    output_path: Path | None,
    cycles: int,
    min_nodes: int,
    max_nodes: int,
    min_services: int,
    max_services: int,
    seed: int,
    milp_time_limit: int,
    decode_mode: str = "beam",
    beam_width: int = 16,
    topk_actions: int = 6,
) -> dict[str, Any]:
    ppo_model = MaskablePPO.load(str(ppo_model_path))
    episodes = _load_episodes(dataset_path)
    cfg = _build_feature_config(episodes)
    summary, rows = _benchmark_ppo_instance(
        ppo_model=ppo_model,
        cfg=cfg,
        cycles=cycles,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        min_services=min_services,
        max_services=max_services,
        seed=seed,
        milp_time_limit=milp_time_limit,
        decode_mode=decode_mode,
        beam_width=beam_width,
        topk_actions=topk_actions,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return summary


def monitor_ppo_quality_latency(
    ppo_model_path: Path,
    dataset_path: Path,
    output_path: Path | None,
    cycles: int,
    min_nodes: int,
    max_nodes: int,
    min_services: int,
    max_services: int,
    seed: int,
    milp_time_limit: int,
    beam_width: int = 16,
    topk_actions: int = 6,
) -> dict[str, Any]:
    ppo_model = MaskablePPO.load(str(ppo_model_path))
    episodes = _load_episodes(dataset_path)
    cfg = _build_feature_config(episodes)

    beam_summary, beam_rows = _benchmark_ppo_instance(
        ppo_model=ppo_model,
        cfg=cfg,
        cycles=cycles,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        min_services=min_services,
        max_services=max_services,
        seed=seed,
        milp_time_limit=milp_time_limit,
        decode_mode="beam",
        beam_width=beam_width,
        topk_actions=topk_actions,
    )
    greedy_summary, greedy_rows = _benchmark_ppo_instance(
        ppo_model=ppo_model,
        cfg=cfg,
        cycles=cycles,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        min_services=min_services,
        max_services=max_services,
        seed=seed,
        milp_time_limit=milp_time_limit,
        decode_mode="greedy",
        beam_width=beam_width,
        topk_actions=topk_actions,
    )

    beam_by_cycle = {int(r["cycle"]): r for r in beam_rows}
    merged_rows: list[dict[str, Any]] = []
    for r in greedy_rows:
        cycle = int(r["cycle"])
        b = beam_by_cycle.get(cycle)
        if b is None:
            continue
        merged_rows.append(
            {
                "cycle": cycle,
                "milp_J": r["milp_J"],
                "drl_J_greedy": r["drl_J"],
                "drl_J_beam": b["drl_J"],
                "delta_J_greedy": r["delta_J"],
                "delta_J_beam": b["delta_J"],
                "greedy_minus_beam_delta_j": float(r["delta_J"]) - float(b["delta_J"]),
                "gap_pct_greedy": r["gap_pct"],
                "gap_pct_beam": b["gap_pct"],
                "policy_inference_ms_greedy": r["policy_inference_ms"],
                "policy_inference_ms_beam": b["policy_inference_ms"],
                "decision_time_ms_greedy": r["decision_time_ms"],
                "decision_time_ms_beam": b["decision_time_ms"],
                "storm_ok_greedy": r["storm_ok"],
                "storm_ok_beam": b["storm_ok"],
            }
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in merged_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    quality_delta_avg = float(greedy_summary["avg_delta_j"]) - float(beam_summary["avg_delta_j"])
    policy_avg_beam = float(beam_summary["avg_policy_inference_ms"])
    policy_avg_greedy = float(greedy_summary["avg_policy_inference_ms"])
    speedup = policy_avg_beam / max(policy_avg_greedy, 1e-9)
    summary = {
        "rows": len(merged_rows),
        "beam": beam_summary,
        "greedy": greedy_summary,
        "quality_delta_avg_delta_j_greedy_minus_beam": quality_delta_avg,
        "policy_speedup_beam_over_greedy_x": speedup,
        "accept_greedy_quality_close_to_beam": quality_delta_avg <= 0.01,
        "accept_greedy_p50_policy_ms_le_10": float(greedy_summary["p50_policy_inference_ms"]) <= 10.0,
        "accept_greedy_zero_storm_violation": bool(greedy_summary["accept_zero_storm_violation"]),
    }
    return summary


def _sequential_infer_placement(
    ep: EpisodeRecord,
    model: DynamicSequentialPolicy,
    cfg: FeatureConfig,
    beam_width: int = 8,
    topk_actions: int = 4,
) -> tuple[dict[str, tuple[str, str]], int]:
    services = sorted(ep.services, key=lambda s: s.service_id)
    node_map, _ = _build_index_maps(ep)
    reward_ds = _dataset_from_episode(ep)

    @dataclass
    class BeamState:
        placement: dict[str, tuple[str, str]]
        residual_cpu: list[float]
        residual_mem: list[float]
        migration_used: int
        logp: float

    beams = [
        BeamState(
            placement={},
            residual_cpu=[n.cap_cpu for n in ep.nodes],
            residual_mem=[n.cap_mem_gb for n in ep.nodes],
            migration_used=0,
            logp=0.0,
        )
    ]

    with torch.no_grad():
        for step_idx, svc in enumerate(services):
            expanded: list[BeamState] = []
            for b in beams:
                mask = _build_mask_for_service(
                    ep, svc, b.residual_cpu, b.residual_mem, b.migration_used, cfg
                )
                state_vec = _vectorize_state_for_service(
                    ep, svc, step_idx, b.residual_cpu, b.residual_mem, b.migration_used, cfg
                )
                states = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                masks = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                logits = model(states)
                masked_logits = logits.masked_fill(masks <= 0.5, -1e9)
                probs = torch.softmax(masked_logits, dim=1).squeeze(0)
                k = max(1, min(topk_actions, int((mask > 0.5).sum())))
                topv, topi = torch.topk(probs, k=k)

                for prob, action_tensor in zip(topv.tolist(), topi.tolist()):
                    action_idx = int(action_tensor)
                    vi, ni = _decode_action_index(action_idx, cfg.max_nodes)
                    if ni >= len(ep.nodes) or vi >= len(svc.variants):
                        continue

                    chosen_var = svc.variants[vi]
                    chosen_node = ep.nodes[ni].node_id

                    nxt = BeamState(
                        placement=dict(b.placement),
                        residual_cpu=list(b.residual_cpu),
                        residual_mem=list(b.residual_mem),
                        migration_used=b.migration_used,
                        logp=b.logp + math.log(max(prob, 1e-12)),
                    )
                    nxt.placement[svc.service_id] = (chosen_var.variant_id, chosen_node)
                    nxt.residual_cpu[ni] -= chosen_var.cpu
                    nxt.residual_mem[ni] -= chosen_var.mem_gb
                    if svc.prev_variant != chosen_var.variant_id or svc.prev_node != chosen_node:
                        nxt.migration_used += 1
                    expanded.append(nxt)

            if not expanded:
                break
            expanded.sort(key=lambda x: x.logp, reverse=True)
            beams = expanded[: max(1, beam_width)]

    # Complete any unfinished assignment with stay-put fallback.
    for b in beams:
        for svc in services:
            if svc.service_id in b.placement:
                continue
            b.placement[svc.service_id] = (svc.prev_variant, svc.prev_node)

    # Choose candidate with best true objective, then by model log-probability.
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
        # Last-resort safe fallback.
        placement = {svc.service_id: (svc.prev_variant, svc.prev_node) for svc in services}
        return placement, 0

    # Recompute migrations from final placement to keep count consistent.
    migration_used = 0
    for svc in services:
        var, node = best.placement[svc.service_id]
        if svc.prev_variant != var or svc.prev_node != node:
            migration_used += 1
    return best.placement, migration_used


def _dataset_from_episode(ep: EpisodeRecord) -> MILPDataset:
    from src.solver.dataset_generator import NodeSpec, ServiceSpec

    nodes = [
        NodeSpec(
            node_id=n.node_id,
            cap_cpu=n.cap_cpu,
            cap_mem_gb=n.cap_mem_gb,
            energy_cost=n.energy_cost,
            e_mem_unit=n.e_mem_unit,
        )
        for n in ep.nodes
    ]
    services = [
        ServiceSpec(
            service_id=s.service_id,
            service_type=s.service_type,
            migration_cost=s.migration_cost,
            valid_variants=[v.variant_id for v in s.variants],
        )
        for s in ep.services
    ]

    r_req: dict[tuple[str, str], float] = {}
    r_mem: dict[tuple[str, str], float] = {}
    acc: dict[tuple[str, str], float] = {}
    x_prev: dict[tuple[str, str, str], float] = {}

    for s in ep.services:
        for v in s.variants:
            r_req[(s.service_id, v.variant_id)] = v.cpu
            r_mem[(s.service_id, v.variant_id)] = v.mem_gb
            acc[(s.service_id, v.variant_id)] = v.acc
            for n in ep.nodes:
                x_prev[(s.service_id, v.variant_id, n.node_id)] = 0.0
        x_prev[(s.service_id, s.prev_variant, s.prev_node)] = 1.0

    return MILPDataset(
        nodes=nodes,
        services=services,
        r_req=r_req,
        r_mem=r_mem,
        acc=acc,
        x_prev=x_prev,
        theta_max=1.3,
        v_storm_max=ep.storm_max,
        w_c=ep.w_c,
        w_d=ep.w_d,
        w_a=ep.w_a,
    )


def benchmark_model(
    model_path: Path,
    output_path: Path | None,
    cycles: int,
    min_nodes: int,
    max_nodes: int,
    min_services: int,
    max_services: int,
    seed: int,
    milp_time_limit: int,
    beam_width: int = 8,
    topk_actions: int = 4,
) -> dict[str, Any]:
    model, cfg = _load_model_checkpoint(model_path)
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    for cycle in range(1, cycles + 1):
        n_nodes = rng.randint(min_nodes, max_nodes)
        n_services = rng.randint(min_services, max_services)

        w_c, w_d, w_a = _normalize_weights(rng.uniform(0.1, 0.6), rng.uniform(0.05, 0.4), rng.uniform(0.2, 0.9))
        ds = generate_dataset(
            num_nodes=n_nodes,
            num_services=n_services,
            seed=seed + cycle,
            v_storm_max=rng.randint(2, max(2, min(6, n_services))),
            w_c=w_c,
            w_d=w_d,
            w_a=w_a,
            theta_max=rng.uniform(1.15, 1.35),
        )
        try:
            milp_result = solve_placement(ds, time_limit=milp_time_limit, verbose=False)
        except Exception:
            continue
        if milp_result is None:
            continue

        ep = _build_episode(ds, milp_result.objective_value, milp_result.placement, f"bench_{cycle:06d}")
        drl_placement, migration_used = _sequential_infer_placement(
            ep,
            model,
            cfg,
            beam_width=beam_width,
            topk_actions=topk_actions,
        )
        drl_obj = evaluate_objective_for_placement(ds, drl_placement).objective_j
        milp_obj = milp_result.objective_value

        delta_j = drl_obj - milp_obj
        gap_pct = (delta_j / (abs(milp_obj) + 1e-8)) * 100.0
        storm_ok = migration_used <= ds.v_storm_max

        rows.append(
            {
                "cycle": cycle,
                "num_nodes": n_nodes,
                "num_services": n_services,
                "milp_J": milp_obj,
                "drl_J": drl_obj,
                "delta_J": delta_j,
                "gap_pct": gap_pct,
                "storm_max": ds.v_storm_max,
                "drl_migrations": migration_used,
                "storm_ok": storm_ok,
            }
        )

    if not rows:
        raise RuntimeError("No valid benchmark rows produced.")

    avg_gap = float(np.mean([r["gap_pct"] for r in rows]))
    p95_gap = float(np.percentile([r["gap_pct"] for r in rows], 95))
    avg_delta_j = float(np.mean([r["delta_J"] for r in rows]))
    p95_delta_j = float(np.percentile([r["delta_J"] for r in rows], 95))
    storm_violation_rate = float(np.mean([0.0 if r["storm_ok"] else 1.0 for r in rows]))

    summary = {
        "rows": len(rows),
        "avg_delta_j": avg_delta_j,
        "p95_delta_j": p95_delta_j,
        "avg_gap_pct_points": avg_delta_j * 100.0,
        "avg_gap_pct": avg_gap,
        "p95_gap_pct": p95_gap,
        "storm_violation_rate": storm_violation_rate,
        "accept_gap_le_2pct_points": avg_delta_j <= 0.02,
        "accept_gap_le_2pct_relative": avg_gap <= 2.0,
        "accept_zero_storm_violation": storm_violation_rate <= 0.0,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return summary


def run_pipeline(args: argparse.Namespace) -> int:
    gen = generate_oracle_dataset(
        output_path=Path(args.dataset_out),
        samples=args.samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        min_services=args.min_services,
        max_services=args.max_services,
        seed=args.seed,
        milp_time_limit=args.milp_time_limit,
    )
    print(json.dumps({"stage": "generate", **gen}, ensure_ascii=True))

    train = train_bc(
        dataset_path=Path(args.dataset_out),
        output_model=Path(args.model_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps({"stage": "train", **train}, ensure_ascii=True))

    bench = benchmark_model(
        model_path=Path(args.model_out),
        output_path=Path(args.benchmark_out),
        cycles=args.benchmark_cycles,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        min_services=args.min_services,
        max_services=args.max_services,
        seed=args.seed + 999,
        milp_time_limit=args.milp_time_limit,
        beam_width=args.beam_width,
        topk_actions=args.topk_actions,
    )
    print(json.dumps({"stage": "benchmark", **bench}, ensure_ascii=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dynamic DRL v2 pipeline (offline BC bootstrap)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate MILP-oracle episodes JSONL")
    g.add_argument("--output", default="results/drl_v2_oracle_episodes.jsonl")
    g.add_argument("--samples", type=int, default=600)
    g.add_argument("--min-nodes", type=int, default=4)
    g.add_argument("--max-nodes", type=int, default=8)
    g.add_argument("--min-services", type=int, default=6)
    g.add_argument("--max-services", type=int, default=20)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--milp-time-limit", type=int, default=20)

    t = sub.add_parser("train", help="Train BC model from oracle episodes")
    t.add_argument("--dataset", default="results/drl_v2_oracle_episodes.jsonl")
    t.add_argument("--model-out", default="src/drl/models/drl_v2_bc.pt")
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--learning-rate", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=42)

    b = sub.add_parser("benchmark", help="Benchmark BC model against MILP")
    b.add_argument("--model", default="src/drl/models/drl_v2_bc.pt")
    b.add_argument("--output", default="results/drl_v2_bc_benchmark.jsonl")
    b.add_argument("--cycles", type=int, default=200)
    b.add_argument("--min-nodes", type=int, default=4)
    b.add_argument("--max-nodes", type=int, default=8)
    b.add_argument("--min-services", type=int, default=6)
    b.add_argument("--max-services", type=int, default=20)
    b.add_argument("--seed", type=int, default=43)
    b.add_argument("--milp-time-limit", type=int, default=20)
    b.add_argument("--beam-width", type=int, default=8)
    b.add_argument("--topk-actions", type=int, default=4)

    allin = sub.add_parser("pipeline", help="Run generate -> train -> benchmark")
    allin.add_argument("--dataset-out", default="results/drl_v2_oracle_episodes.jsonl")
    allin.add_argument("--model-out", default="src/drl/models/drl_v2_bc.pt")
    allin.add_argument("--benchmark-out", default="results/drl_v2_bc_benchmark.jsonl")
    allin.add_argument("--samples", type=int, default=600)
    allin.add_argument("--epochs", type=int, default=25)
    allin.add_argument("--batch-size", type=int, default=256)
    allin.add_argument("--learning-rate", type=float, default=3e-4)
    allin.add_argument("--benchmark-cycles", type=int, default=200)
    allin.add_argument("--min-nodes", type=int, default=4)
    allin.add_argument("--max-nodes", type=int, default=8)
    allin.add_argument("--min-services", type=int, default=6)
    allin.add_argument("--max-services", type=int, default=20)
    allin.add_argument("--seed", type=int, default=42)
    allin.add_argument("--milp-time-limit", type=int, default=20)
    allin.add_argument("--beam-width", type=int, default=8)
    allin.add_argument("--topk-actions", type=int, default=4)

    ppo_t = sub.add_parser("train-ppo", help="Fine-tune PPO from BC checkpoint (dynamic v2 env)")
    ppo_t.add_argument("--dataset", default="results/drl_v2_oracle_episodes.jsonl")
    ppo_t.add_argument("--bc-model", default="src/drl/models/drl_v2_bc.pt")
    ppo_t.add_argument("--ppo-out", default="src/drl/models/drl_v2_ppo.zip")
    ppo_t.add_argument("--timesteps", type=int, default=120000)
    ppo_t.add_argument("--learning-rate", type=float, default=1e-4)
    ppo_t.add_argument("--eval-every", type=int, default=5000)
    ppo_t.add_argument("--eval-cycles", type=int, default=40)
    ppo_t.add_argument("--min-nodes", type=int, default=4)
    ppo_t.add_argument("--max-nodes", type=int, default=8)
    ppo_t.add_argument("--min-services", type=int, default=6)
    ppo_t.add_argument("--max-services", type=int, default=20)
    ppo_t.add_argument("--milp-time-limit", type=int, default=20)
    ppo_t.add_argument("--eval-decode-mode", choices=["greedy", "beam"], default="greedy")
    ppo_t.add_argument("--beam-width", type=int, default=16)
    ppo_t.add_argument("--topk-actions", type=int, default=6)
    ppo_t.add_argument("--seed", type=int, default=42)

    ppo_b = sub.add_parser("benchmark-ppo", help="Benchmark PPO v2 model against MILP")
    ppo_b.add_argument("--ppo-model", default="src/drl/models/drl_v2_ppo.zip")
    ppo_b.add_argument("--dataset", default="results/drl_v2_oracle_episodes.jsonl")
    ppo_b.add_argument("--output", default="results/drl_v2_ppo_benchmark.jsonl")
    ppo_b.add_argument("--cycles", type=int, default=200)
    ppo_b.add_argument("--min-nodes", type=int, default=4)
    ppo_b.add_argument("--max-nodes", type=int, default=8)
    ppo_b.add_argument("--min-services", type=int, default=6)
    ppo_b.add_argument("--max-services", type=int, default=20)
    ppo_b.add_argument("--seed", type=int, default=43)
    ppo_b.add_argument("--milp-time-limit", type=int, default=20)
    ppo_b.add_argument("--decode-mode", choices=["greedy", "beam"], default="greedy")
    ppo_b.add_argument("--beam-width", type=int, default=16)
    ppo_b.add_argument("--topk-actions", type=int, default=6)

    ppo_m = sub.add_parser("monitor-ppo", help="Monitor PPO quality/latency: greedy vs beam on same episodes")
    ppo_m.add_argument("--ppo-model", default="src/drl/models/drl_v2_ppo.zip")
    ppo_m.add_argument("--dataset", default="results/drl_v2_oracle_episodes.jsonl")
    ppo_m.add_argument("--output", default="results/drl_v2_ppo_monitor.jsonl")
    ppo_m.add_argument("--cycles", type=int, default=200)
    ppo_m.add_argument("--min-nodes", type=int, default=4)
    ppo_m.add_argument("--max-nodes", type=int, default=8)
    ppo_m.add_argument("--min-services", type=int, default=6)
    ppo_m.add_argument("--max-services", type=int, default=20)
    ppo_m.add_argument("--seed", type=int, default=43)
    ppo_m.add_argument("--milp-time-limit", type=int, default=20)
    ppo_m.add_argument("--beam-width", type=int, default=16)
    ppo_m.add_argument("--topk-actions", type=int, default=6)

    ppo_all = sub.add_parser("pipeline-ppo", help="Run BC pipeline then PPO train + PPO benchmark")
    ppo_all.add_argument("--dataset-out", default="results/drl_v2_oracle_episodes_ppo.jsonl")
    ppo_all.add_argument("--bc-model-out", default="src/drl/models/drl_v2_bc_ppo_seed.pt")
    ppo_all.add_argument("--ppo-model-out", default="src/drl/models/drl_v2_ppo.zip")
    ppo_all.add_argument("--bc-benchmark-out", default="results/drl_v2_bc_benchmark_ppo_seed.jsonl")
    ppo_all.add_argument("--ppo-benchmark-out", default="results/drl_v2_ppo_benchmark.jsonl")
    ppo_all.add_argument("--samples", type=int, default=700)
    ppo_all.add_argument("--epochs", type=int, default=25)
    ppo_all.add_argument("--batch-size", type=int, default=256)
    ppo_all.add_argument("--learning-rate", type=float, default=3e-4)
    ppo_all.add_argument("--benchmark-cycles", type=int, default=150)
    ppo_all.add_argument("--ppo-timesteps", type=int, default=120000)
    ppo_all.add_argument("--ppo-learning-rate", type=float, default=1e-4)
    ppo_all.add_argument("--ppo-eval-every", type=int, default=5000)
    ppo_all.add_argument("--ppo-eval-cycles", type=int, default=40)
    ppo_all.add_argument("--min-nodes", type=int, default=4)
    ppo_all.add_argument("--max-nodes", type=int, default=8)
    ppo_all.add_argument("--min-services", type=int, default=6)
    ppo_all.add_argument("--max-services", type=int, default=20)
    ppo_all.add_argument("--seed", type=int, default=42)
    ppo_all.add_argument("--milp-time-limit", type=int, default=20)
    ppo_all.add_argument("--eval-decode-mode", choices=["greedy", "beam"], default="greedy")
    ppo_all.add_argument("--benchmark-decode-mode", choices=["greedy", "beam"], default="greedy")
    ppo_all.add_argument("--beam-width", type=int, default=16)
    ppo_all.add_argument("--topk-actions", type=int, default=6)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "generate":
        out = generate_oracle_dataset(
            output_path=Path(args.output),
            samples=args.samples,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            min_services=args.min_services,
            max_services=args.max_services,
            seed=args.seed,
            milp_time_limit=args.milp_time_limit,
        )
        print(json.dumps(out, ensure_ascii=True))
        return 0

    if args.cmd == "train":
        out = train_bc(
            dataset_path=Path(args.dataset),
            output_model=Path(args.model_out),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        print(json.dumps(out, ensure_ascii=True))
        return 0

    if args.cmd == "benchmark":
        out = benchmark_model(
            model_path=Path(args.model),
            output_path=Path(args.output),
            cycles=args.cycles,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            min_services=args.min_services,
            max_services=args.max_services,
            seed=args.seed,
            milp_time_limit=args.milp_time_limit,
            beam_width=args.beam_width,
            topk_actions=args.topk_actions,
        )
        print(json.dumps(out, ensure_ascii=True))
        return 0

    if args.cmd == "pipeline":
        return run_pipeline(args)

    if args.cmd == "train-ppo":
        out = train_ppo_from_bc(
            dataset_path=Path(args.dataset),
            bc_model_path=Path(args.bc_model),
            output_model_path=Path(args.ppo_out),
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            eval_every=args.eval_every,
            eval_cycles=args.eval_cycles,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            min_services=args.min_services,
            max_services=args.max_services,
            milp_time_limit=args.milp_time_limit,
            eval_decode_mode=args.eval_decode_mode,
            beam_width=args.beam_width,
            topk_actions=args.topk_actions,
        )
        print(json.dumps(out, ensure_ascii=True))
        return 0

    if args.cmd == "benchmark-ppo":
        out = benchmark_ppo_model(
            ppo_model_path=Path(args.ppo_model),
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            cycles=args.cycles,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            min_services=args.min_services,
            max_services=args.max_services,
            seed=args.seed,
            milp_time_limit=args.milp_time_limit,
            decode_mode=args.decode_mode,
            beam_width=args.beam_width,
            topk_actions=args.topk_actions,
        )
        print(json.dumps(out, ensure_ascii=True))
        return 0

    if args.cmd == "monitor-ppo":
        out = monitor_ppo_quality_latency(
            ppo_model_path=Path(args.ppo_model),
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            cycles=args.cycles,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            min_services=args.min_services,
            max_services=args.max_services,
            seed=args.seed,
            milp_time_limit=args.milp_time_limit,
            beam_width=args.beam_width,
            topk_actions=args.topk_actions,
        )
        print(json.dumps(out, ensure_ascii=True))
        return 0

    if args.cmd == "pipeline-ppo":
        # Stage A: BC bootstrap
        stage_a = argparse.Namespace(
            dataset_out=args.dataset_out,
            model_out=args.bc_model_out,
            benchmark_out=args.bc_benchmark_out,
            samples=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            benchmark_cycles=args.benchmark_cycles,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            min_services=args.min_services,
            max_services=args.max_services,
            seed=args.seed,
            milp_time_limit=args.milp_time_limit,
            beam_width=args.beam_width,
            topk_actions=args.topk_actions,
        )
        run_pipeline(stage_a)
        # Stage B: PPO fine-tune
        ppo_train = train_ppo_from_bc(
            dataset_path=Path(args.dataset_out),
            bc_model_path=Path(args.bc_model_out),
            output_model_path=Path(args.ppo_model_out),
            timesteps=args.ppo_timesteps,
            learning_rate=args.ppo_learning_rate,
            seed=args.seed,
            eval_every=args.ppo_eval_every,
            eval_cycles=args.ppo_eval_cycles,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            min_services=args.min_services,
            max_services=args.max_services,
            milp_time_limit=args.milp_time_limit,
            eval_decode_mode=args.eval_decode_mode,
            beam_width=args.beam_width,
            topk_actions=args.topk_actions,
        )
        print(json.dumps({"stage": "train_ppo", **ppo_train}, ensure_ascii=True))
        ppo_bench = benchmark_ppo_model(
            ppo_model_path=Path(args.ppo_model_out),
            dataset_path=Path(args.dataset_out),
            output_path=Path(args.ppo_benchmark_out),
            cycles=args.benchmark_cycles,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            min_services=args.min_services,
            max_services=args.max_services,
            seed=args.seed + 2024,
            milp_time_limit=args.milp_time_limit,
            decode_mode=args.benchmark_decode_mode,
            beam_width=args.beam_width,
            topk_actions=args.topk_actions,
        )
        print(json.dumps({"stage": "benchmark_ppo", **ppo_bench}, ensure_ascii=True))
        return 0

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
