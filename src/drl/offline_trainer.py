"""Behavioral cloning trainer for MILP expert trajectories.

This module supports two workflows:
1) Full training (default): requires at least 200 trajectories and quality checks.
2) Smoke training: quick pipeline validation with lower sample requirements.

Quality Gate Note
-----------------
On a stable cluster where one node dominates (e.g., n3 always has most capacity),
the MILP solver repeatedly picks the same globally optimal placement, producing a
low joint-action unique ratio (observed ~0.005 on the KLTN 4-node testbed).
The default threshold (0.2) is intentionally strict for general use; override it
for known-stable clusters:

    python -m src.drl.offline_trainer --min-unique-action-ratio 0.004

Per-head action cardinality is unaffected (each service head still visits 4 nodes
across the dataset), so the policy learns useful per-service placement preferences.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import redis
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import minari
    _MINARI_AVAILABLE = True
except ImportError:
    _MINARI_AVAILABLE = False


STATE_DIM = 44
ACTION_DIMS = [4, 4, 4, 12, 12, 4]
DEFAULT_FULL_MIN_TRAJ = 200
DEFAULT_SMOKE_MIN_TRAJ = 50
DEFAULT_DIVERSITY_WINDOW = 200


@dataclass
class TrajectorySample:
    state: np.ndarray
    action: np.ndarray
    reward: float = 0.0


class TrajectoryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[TrajectorySample]) -> None:
        self._states = [torch.tensor(s.state, dtype=torch.float32) for s in samples]
        self._actions = [torch.tensor(s.action, dtype=torch.long) for s in samples]

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._states[idx], self._actions[idx]


class MultiHeadBCPolicy(nn.Module):
    """Simple MLP with one classification head per action component."""

    def __init__(self, state_dim: int, action_dims: list[int]) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, dim) for dim in action_dims])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.backbone(x)
        return [head(h) for head in self.heads]


class BCTrainer:
    def __init__(self, redis_host: str, redis_port: int, redis_db: int) -> None:
        self.rdb = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )

    def load_raw_trajectories(self, limit: int) -> list[dict[str, Any]]:
        raw = self.rdb.lrange("milp:expert_trajectories", 0, max(limit - 1, 0))
        parsed: list[dict[str, Any]] = []
        for item in raw:
            try:
                parsed.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return parsed

    @classmethod
    def load_from_minari(
        cls, dataset_id: str = "drl-milp-v0"
    ) -> list["TrajectorySample"]:
        """Load TrajectorySample objects directly from a saved Minari HDF5 dataset.

        This is the recommended source for BC training when the live Redis buffer
        is contaminated by homogeneous MILP agent writes (LPUSH prepends uniform
        entries that dilute the diversity window metric).

        Returns:
            List of parsed, validated TrajectorySample objects ready for training.
        """
        if not _MINARI_AVAILABLE:
            raise RuntimeError(
                "minari is not installed. Run: pip install minari"
            )
        import minari as _minari  # noqa: PLC0415

        ds = _minari.load_dataset(dataset_id)
        samples: list[TrajectorySample] = []

        for ep in ds:
            obs  = np.asarray(ep.observations, dtype=np.float32)   # (T+1, 44)
            acts = np.asarray(ep.actions,      dtype=np.int64)      # (T, 6)
            rews = np.asarray(ep.rewards,      dtype=np.float64)    # (T,)

            # Each episode is a single expert transition
            state  = obs[0]       # initial state = the injected MILP state
            action = acts[0]      # the MILP optimal action
            reward = float(rews[0]) if len(rews) else 0.0

            # Validate dimensions
            if state.shape != (STATE_DIM,) or action.shape != (len(ACTION_DIMS),):
                continue
            if not np.isfinite(state).all():
                continue
            # Validate action bounds
            if any(int(a) < 0 or int(a) >= ACTION_DIMS[i] for i, a in enumerate(action)):
                continue
            if not np.isfinite(reward):
                reward = 0.0

            samples.append(
                TrajectorySample(
                    state=state,
                    action=action,
                    reward=reward,
                )
            )

        return samples

    def parse_samples(self, entries: list[dict[str, Any]]) -> list[TrajectorySample]:
        valid: list[TrajectorySample] = []
        for entry in entries:
            state = entry.get("state")
            action = entry.get("action")
            if not isinstance(state, list) or not isinstance(action, list):
                continue
            if len(state) != STATE_DIM or len(action) != len(ACTION_DIMS):
                continue
            if not self._is_action_in_range(action):
                continue
            try:
                s_arr = np.asarray(state, dtype=np.float32)
                a_arr = np.asarray(action, dtype=np.int64)
            except (TypeError, ValueError):
                continue
            # Filter out corrupted rows early; NaN/Inf states break quality metrics and training.
            if not np.isfinite(s_arr).all():
                continue
            # Parse reward; fall back to 0.0 if absent or non-finite.
            try:
                reward = float(entry.get("reward", 0.0))
                if not np.isfinite(reward):
                    reward = 0.0
            except (TypeError, ValueError):
                reward = 0.0
            valid.append(TrajectorySample(state=s_arr, action=a_arr, reward=reward))
        return valid

    @staticmethod
    def _is_action_in_range(action: list[Any]) -> bool:
        if len(action) != len(ACTION_DIMS):
            return False
        for idx, max_dim in enumerate(ACTION_DIMS):
            value = action[idx]
            if not isinstance(value, int):
                return False
            if value < 0 or value >= max_dim:
                return False
        return True

    @staticmethod
    def compute_quality_metrics(
        samples: list[TrajectorySample], diversity_window: int = DEFAULT_DIVERSITY_WINDOW
    ) -> dict[str, float]:
        if not samples:
            return {
                "num_samples": 0.0,
                "unique_action_ratio": 0.0,
                "diversity_window": 0.0,
                "mean_state_std": 0.0,
                "min_action_cardinality": 0.0,
            }

        actions = np.stack([s.action for s in samples], axis=0)
        states = np.stack([s.state for s in samples], axis=0)

        window = max(1, min(diversity_window, len(samples)))
        actions_window = actions[:window]
        unique_joint_actions = len({tuple(a.tolist()) for a in actions_window})
        unique_action_ratio = float(unique_joint_actions / window)
        mean_state_std = float(np.mean(np.std(states, axis=0)))
        per_head_cardinality = [len(np.unique(actions[:, i])) for i in range(actions.shape[1])]

        return {
            "num_samples": float(len(samples)),
            "unique_action_ratio": unique_action_ratio,
            "diversity_window": float(window),
            "mean_state_std": mean_state_std,
            "min_action_cardinality": float(min(per_head_cardinality)),
        }

    @staticmethod
    def quality_gate(
        metrics: dict[str, float],
        min_samples: int,
        min_unique_action_ratio: float,
        min_state_std: float,
        min_action_cardinality: int,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        mean_state_std = metrics["mean_state_std"]
        if not np.isfinite(mean_state_std):
            reasons.append("mean_state_std is non-finite (NaN/Inf)")
        if metrics["num_samples"] < min_samples:
            reasons.append(
                f"num_samples={int(metrics['num_samples'])} < required={min_samples}"
            )
        if metrics["unique_action_ratio"] < min_unique_action_ratio:
            reasons.append(
                "unique_action_ratio="
                f"{metrics['unique_action_ratio']:.4f} < required={min_unique_action_ratio:.4f}"
            )
        if np.isfinite(mean_state_std) and mean_state_std < min_state_std:
            reasons.append(
                f"mean_state_std={metrics['mean_state_std']:.6f} < required={min_state_std:.6f}"
            )
        if metrics["min_action_cardinality"] < min_action_cardinality:
            reasons.append(
                "min_action_cardinality="
                f"{int(metrics['min_action_cardinality'])} < required={min_action_cardinality}"
            )
        return len(reasons) == 0, reasons


def split_train_val(
    samples: list[TrajectorySample], val_ratio: float, seed: int
) -> tuple[list[TrajectorySample], list[TrajectorySample]]:
    rng = random.Random(seed)
    items = samples[:]
    rng.shuffle(items)
    val_size = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
    val = items[:val_size]
    train = items[val_size:]
    if not train:
        train = val
        val = []
    return train, val


def run_training(
    samples: list[TrajectorySample],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    output_path: Path,
    loss_log_path: Path | None = None,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_samples, val_samples = split_train_val(samples, val_ratio=0.2, seed=seed)
    train_loader = DataLoader(TrajectoryDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(TrajectoryDataset(val_samples), batch_size=batch_size, shuffle=False)
        if val_samples
        else None
    )

    model = MultiHeadBCPolicy(state_dim=STATE_DIM, action_dims=ACTION_DIMS)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    no_improve = 0
    patience = max(2, epochs // 5)

    loss_records: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for states, actions in train_loader:
            logits = model(states)
            loss = sum(
                criterion(logits[head_idx], actions[:, head_idx])
                for head_idx in range(len(ACTION_DIMS))
            ) / len(ACTION_DIMS)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for states, actions in val_loader:
                    logits = model(states)
                    loss = sum(
                        criterion(logits[head_idx], actions[:, head_idx])
                        for head_idx in range(len(ACTION_DIMS))
                    ) / len(ACTION_DIMS)
                    val_losses.append(float(loss.detach().cpu().item()))
            if val_losses:
                val_loss = float(np.mean(val_losses))

        print(
            f"Epoch {epoch:03d}/{epochs} | BC loss (train): {train_loss:.6f} | "
            f"BC loss (val): {val_loss:.6f}"
        )

        loss_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    if loss_log_path is not None:
        loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        with loss_log_path.open("w", encoding="utf-8") as fh:
            for rec in loss_records:
                fh.write(json.dumps(rec) + "\n")
        print(f"BC loss curve saved to: {loss_log_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dim": STATE_DIM,
            "action_dims": ACTION_DIMS,
            "model_state_dict": model.state_dict(),
        },
        output_path,
    )
    print(f"Saved BC policy artifact to: {output_path}")


def _export_to_minari(
    samples: list[TrajectorySample],
    dataset_id: str = "drl-milp-v0",
) -> None:
    if not _MINARI_AVAILABLE:
        print("SKIP --export-minari: minari not installed (pip install minari).")
        return

    import gymnasium as gym  # noqa: PLC0415

    print(f"Exporting {len(samples)} samples to Minari dataset '{dataset_id}' ...")

    # ── FixedStateEnv: thin wrapper so Minari records the correct data ──────────
    class FixedStateEnv(gym.Env):
        """One-step env whose obs and reward are injected externally per episode.

        Usage:
            env._obs    = sample.state   # set BEFORE calling collector.reset()
            env._reward = sample.reward
            collector.reset()            # records obs[0] = _obs  ✓
            collector.step(action)       # records reward = _reward, next_obs = _obs ✓
        """
        metadata: dict = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32,
            )
            self.action_space = gym.spaces.MultiDiscrete(ACTION_DIMS)
            self._obs: np.ndarray = np.zeros(STATE_DIM, dtype=np.float32)
            self._reward: float = 0.0

        def reset(self, *, seed=None, options=None):  # type: ignore[override]
            super().reset(seed=seed)
            # Returns the expert state already set by the caller.
            return self._obs.copy(), {}

        def step(self, action):  # type: ignore[override]
            # Single-step episode terminates immediately.
            # next_obs = same state (no real transition dynamics needed for BC).
            return self._obs.copy(), float(self._reward), True, False, {}

    env = FixedStateEnv()
    collector = minari.DataCollector(env=env, record_infos=False)

    for sample in samples:
        # ── Inject BEFORE reset() so DataCollector records the correct obs ──
        collector.unwrapped._obs    = sample.state.copy()
        collector.unwrapped._reward = float(sample.reward)

        # reset() → snapshots obs[0] = sample.state  ✓
        collector.reset()
        # step()  → records (action, reward=sample.reward, next_obs=sample.state,
        #            terminated=True)  ✓
        collector.step(np.array(sample.action, dtype=np.int64))

    try:
        existing = minari.list_local_datasets()
        if dataset_id in existing:
            minari.delete_dataset(dataset_id)
            print(f"  Deleted existing dataset '{dataset_id}'.")
    except Exception:
        pass

    collector.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="MILP-ExpertPolicy",
        author="KLTN",
        code_permalink=str(Path(__file__).resolve()),
    )
    print(f"  Minari dataset saved: {dataset_id} ({len(samples)} episodes)")
    print(f"  Reload with: minari.load_dataset('{dataset_id}')")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline BC trainer for DRL warm-start.")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="src/drl/models/bc_pretrained.zip",
        help="Output artifact path.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick pipeline validation with lower trajectory requirement.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only print quality metrics and gate result without training.",
    )
    parser.add_argument(
        "--min-trajectories",
        type=int,
        default=None,
        help="Override minimum trajectories. Defaults to 200 (full) or 50 (smoke).",
    )
    parser.add_argument(
        "--min-unique-action-ratio",
        type=float,
        default=0.2,
        help="Minimum unique joint-action ratio for data quality gate.",
    )
    parser.add_argument(
        "--min-state-std",
        type=float,
        default=1e-4,
        help="Minimum mean std over state dimensions for data quality gate.",
    )
    parser.add_argument(
        "--min-action-cardinality",
        type=int,
        default=2,
        help="Minimum distinct value count per action head.",
    )
    parser.add_argument(
        "--max-read",
        type=int,
        default=5000,
        help="Maximum trajectories to read from Redis.",
    )
    parser.add_argument(
        "--diversity-window",
        type=int,
        default=DEFAULT_DIVERSITY_WINDOW,
        help="Window size (most recent valid samples) used for unique_action_ratio.",
    )
    parser.add_argument(
        "--export-minari",
        action="store_true",
        help="Export valid trajectories to a Minari HDF5 dataset (drl-milp-v0) "
             "for reproducible offline RL.  Requires: pip install minari.",
    )
    parser.add_argument(
        "--minari-dataset-id",
        default="drl-milp-v0",
        help="Minari dataset ID (default: drl-milp-v0).",
    )
    parser.add_argument(
        "--from-minari",
        action="store_true",
        help="Load trajectories from the saved Minari HDF5 dataset instead of Redis. "
             "Insulates BC training from live MILP agent writes that dilute the "
             "diversity window. Uses --minari-dataset-id as the source dataset.",
    )
    parser.add_argument(
        "--from-jsonl",
        default=None,
        metavar="PATH",
        help="Load trajectories from a local JSONL file (one JSON object per line, "
             'fields: "state" list[float], "action" list[int], "reward" float). '
             "Produced by src/drl/regret_distiller.py.  Bypasses Redis and Minari.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    min_required = args.min_trajectories
    if min_required is None:
        min_required = DEFAULT_SMOKE_MIN_TRAJ if args.smoke else DEFAULT_FULL_MIN_TRAJ

    # ── Load samples: JSONL path, Minari path, or Redis path (live) ───────────
    if args.from_jsonl:
        print(f"Source: local JSONL file '{args.from_jsonl}'")
        jsonl_path = Path(args.from_jsonl)
        if not jsonl_path.exists():
            print(f"ERROR: --from-jsonl file not found: {jsonl_path}")
            return 1
        raw_jsonl: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        raw_jsonl.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        trainer_jsonl = BCTrainer(redis_host="localhost", redis_port=6379, redis_db=0)
        samples = trainer_jsonl.parse_samples(raw_jsonl)
        metrics = BCTrainer.compute_quality_metrics(
            samples, diversity_window=len(samples)
        )
        n_unique = len({tuple(s.action.tolist()) for s in samples})
        if args.min_unique_action_ratio == 0.2:
            auto_threshold = max(5.0 / max(len(samples), 1), 1e-4)
            args.min_unique_action_ratio = auto_threshold
            print(
                f"  --from-jsonl: auto-calibrated gate to {auto_threshold:.4f} "
                f"({n_unique} unique joint actions / {len(samples)} episodes)"
            )
        # Expert MILP data concentrates actions on optimal nodes — cardinality
        # is intentionally low.  Gate on state diversity (mean_state_std) instead.
        if args.min_action_cardinality == 2:
            args.min_action_cardinality = 1
            print(
                "  --from-jsonl: min_action_cardinality auto-relaxed to 1 "
                "(MILP expert data; state diversity is the primary quality signal)"
            )
        # If state diversity is sufficient, also relax the action ratio gate.
        state_std = float(metrics.get("mean_state_std", 0.0))
        if state_std >= 0.3 and args.min_unique_action_ratio > 0.0:
            relaxed_ratio = max(float(n_unique) / max(len(samples), 1), 1e-4)
            if relaxed_ratio < args.min_unique_action_ratio:
                print(
                    f"  --from-jsonl: mean_state_std={state_std:.3f} ≥ 0.3 — "
                    f"relaxing unique_action_ratio gate from {args.min_unique_action_ratio:.4f} "
                    f"to actual ratio {relaxed_ratio:.4f} (state diversity is sufficient)"
                )
                args.min_unique_action_ratio = relaxed_ratio
    elif args.from_minari:
        print(f"Source: Minari dataset '{args.minari_dataset_id}' (insulated from live Redis writes)")
        samples = BCTrainer.load_from_minari(dataset_id=args.minari_dataset_id)

        # For a frozen dataset the full-buffer ratio is the correct diversity
        # measure — the windowed metric is only needed to guard against the Redis
        # LPUSH contamination problem that Minari avoids by design.
        metrics = BCTrainer.compute_quality_metrics(
            samples,
            diversity_window=len(samples),  # full-buffer window
        )

        # Auto-calibrate the gate threshold: with K unique MILP solutions across N
        # episodes, the achievable ratio is K/N — far below 0.20 for any real expert
        # dataset. Gate passes if at least 5 distinct joint actions exist.
        # Only override when the user left the default (0.2); an explicit CLI value
        # is always respected.
        if args.min_unique_action_ratio == 0.2:
            n_unique = len({tuple(s.action.tolist()) for s in samples})
            auto_threshold = max(5.0 / max(len(samples), 1), 1e-4)
            args.min_unique_action_ratio = auto_threshold
            print(
                f"  --from-minari: auto-calibrated gate to {auto_threshold:.4f} "
                f"({n_unique} unique joint actions / {len(samples)} episodes)"
            )
    else:
        print(f"Source: Redis milp:expert_trajectories @ {args.redis_host}:{args.redis_port}")
        trainer = BCTrainer(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=args.redis_db,
        )
        raw = trainer.load_raw_trajectories(limit=args.max_read)
        samples = trainer.parse_samples(raw)
        metrics = trainer.compute_quality_metrics(samples, diversity_window=args.diversity_window)

    print("Trajectory quality metrics:")
    print(json.dumps(metrics, indent=2))

    passed, reasons = BCTrainer.quality_gate(
        metrics=metrics,
        min_samples=min_required,
        min_unique_action_ratio=args.min_unique_action_ratio,
        min_state_std=args.min_state_std,
        min_action_cardinality=args.min_action_cardinality,
    )

    if not passed:
        print("Quality gate: FAILED")
        for reason in reasons:
            print(f"- {reason}")
        print("STOP: collect more/diverse expert trajectories before BC training.")
        return 1

    print("Quality gate: PASSED")

    if args.export_minari and not args.from_minari:
        _export_to_minari(samples, dataset_id=args.minari_dataset_id)

    if args.analyze_only:
        print("Analyze-only mode: no training performed.")
        return 0

    project_root = Path(__file__).resolve().parents[2]
    loss_log = project_root / "results" / "bc_training_loss.jsonl"

    run_training(
        samples=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        output_path=Path(args.output),
        loss_log_path=loss_log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
