"""Train a PPO policy in the simulator environment (Phase 2, D2.3)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on the path regardless of working directory or PYTHONPATH.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

log = logging.getLogger(__name__)

from drl.edge_env import EdgeEnv                    # noqa: E402
from drl.scenario_generator import ScenarioGenerator  # noqa: E402


class EpisodeRewardTracker(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True

    def mean_reward(self, window: int = 100) -> float:
        if not self.episode_rewards:
            return float("nan")
        if len(self.episode_rewards) < window:
            return float(np.mean(self.episode_rewards))
        return float(np.mean(self.episode_rewards[-window:]))


def _mask_fn(env: EdgeEnv) -> np.ndarray:
    return env.action_masks()


def _make_env(
    mock: bool,
    use_scenarios: bool = False,
    seed: int = 0,
    curriculum_stage: int | None = None,
    objective_profile: str | None = None,
    fixed_scenario: str | None = None,
    storm_boost: float | None = None,
) -> Monitor:
    if use_scenarios:
        gen = ScenarioGenerator(
            seed=seed,
            storm_boost=storm_boost,
            curriculum_stage=curriculum_stage,
            objective_profile=objective_profile,
            fixed_scenario=fixed_scenario,
        )
    else:
        gen = None
    env = EdgeEnv(mock=mock, scenario_generator=gen)
    env = ActionMasker(env, _mask_fn)
    return Monitor(env)


# BC backbone dimensions — must match offline_trainer.py
_BC_HIDDEN = 128
_BC_NET_ARCH = [_BC_HIDDEN, _BC_HIDDEN]


def _load_bc_backbone(bc_checkpoint: str) -> dict[str, torch.Tensor] | None:
    """Load MultiHeadBCPolicy backbone weights from a torch.save checkpoint.

    Returns a mapping of policy_net layer indices to (weight, bias) tensors,
    or None if the checkpoint cannot be loaded.
    """
    try:
        ckpt = torch.load(bc_checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt["model_state_dict"]
        backbone: dict[str, torch.Tensor] = {
            k: v for k, v in sd.items() if k.startswith("backbone.")
        }
        if not backbone:
            log.warning("BC checkpoint contains no backbone.* keys — skipping warm-start")
            return None
        log.info("BC backbone loaded from %s (%d tensors)", bc_checkpoint, len(backbone))
        return backbone
    except Exception as exc:
        log.warning("Could not load BC checkpoint: %s — training from scratch", exc)
        return None


def _apply_bc_weights(model: PPO, backbone: dict[str, torch.Tensor]) -> None:
    """Copy BC backbone weights into the PPO policy's mlp_extractor.policy_net.

    BC backbone layers (backbone.0, backbone.2) map 1-to-1 onto
    SB3 policy_net layers (0, 2) when net_arch=[128, 128] + ReLU is used.
    """
    policy_net = model.policy.mlp_extractor.policy_net
    # BC sequential indices: 0=Linear(obs->128), 1=ReLU, 2=Linear(128->128), 3=ReLU
    mapping = {
        "backbone.0.weight": (0, "weight"),
        "backbone.0.bias":   (0, "bias"),
        "backbone.2.weight": (2, "weight"),
        "backbone.2.bias":   (2, "bias"),
    }
    transferred = 0
    with torch.no_grad():
        for bc_key, (layer_idx, attr) in mapping.items():
            if bc_key not in backbone:
                continue
            bc_tensor = backbone[bc_key]
            ppo_param = getattr(policy_net[layer_idx], attr)
            if ppo_param.shape != bc_tensor.shape:
                log.warning(
                    "Shape mismatch for %s: BC %s vs PPO %s — skipping",
                    bc_key, bc_tensor.shape, ppo_param.shape,
                )
                continue
            ppo_param.copy_(bc_tensor)
            transferred += 1
    log.info("BC warm-start: transferred %d/%d tensors into PPO policy_net", transferred, len(mapping))


def train(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def env_fn() -> Monitor:
        return _make_env(
            mock=not args.no_mock,
            use_scenarios=args.use_scenarios,
            seed=args.seed,
            curriculum_stage=args.curriculum_stage,
            objective_profile=args.scenario_profile,
            fixed_scenario=getattr(args, "fixed_scenario", None),
            storm_boost=getattr(args, "storm_boost", None),
        )

    vec_env = DummyVecEnv([env_fn])

    # When a BC checkpoint is provided, use matching architecture and warm-start.
    bc_backbone: dict[str, torch.Tensor] | None = None
    if args.bc_checkpoint:
        bc_backbone = _load_bc_backbone(args.bc_checkpoint)

    policy_kwargs: dict[str, Any] = {}
    if bc_backbone is not None:
        policy_kwargs = dict(net_arch=_BC_NET_ARCH, activation_fn=nn.ReLU)
        log.info("PPO will use BC-matched architecture: net_arch=%s, activation=ReLU", _BC_NET_ARCH)

    # ── Fine-tune from existing SB3 model (takes priority over BC warm-start) ─
    if getattr(args, "finetune_from", None):
        log.info("Fine-tuning from existing model: %s", args.finetune_from)
        try:
            model = MaskablePPO.load(
                args.finetune_from,
                env=vec_env,
                tensorboard_log=str(log_dir),
                verbose=1,
                seed=args.seed,
            )
            # Allow overriding learning-rate for fine-tuning (lower = less forgetting)
            if getattr(args, "learning_rate", None) is not None:
                model.learning_rate = args.learning_rate
                model.lr_schedule = get_schedule_fn(args.learning_rate)
                for pg in model.policy.optimizer.param_groups:
                    pg["lr"] = args.learning_rate
                log.info("Fine-tune learning rate overridden to %.2e", args.learning_rate)
            if getattr(args, "ent_coef", None) is not None:
                model.ent_coef = args.ent_coef
                log.info("Fine-tune ent_coef overridden to %.3f", args.ent_coef)
            log.info("Fine-tune base model loaded OK")
        except Exception as exc:
            log.warning("Fine-tune load failed (%s) — training from scratch", exc)
            model = MaskablePPO(
                "MlpPolicy", vec_env, verbose=1, seed=args.seed,
                ent_coef=0.01, tensorboard_log=str(log_dir),
                policy_kwargs=policy_kwargs if policy_kwargs else None,
            )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            ent_coef=0.01,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
        )

    if bc_backbone is not None:
        _apply_bc_weights(model, bc_backbone)
        log.info("BC warm-start applied — proceeding with PPO fine-tuning")

    tracker = EpisodeRewardTracker()
    model.learn(total_timesteps=args.timesteps, callback=tracker)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    mean_reward = tracker.mean_reward(window=100)
    if np.isfinite(mean_reward):
        print(f"mean_episode_reward_last_100={mean_reward:.6f}")
    else:
        print("mean_episode_reward_last_100=nan")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO in EdgeEnv simulator.")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate (useful for fine-tuning with lower LR, e.g. 1e-4).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        dest="ent_coef",
        help="Override entropy coefficient (default: keep saved value, typically 0.01). "
             "Higher values (e.g. 0.10) maintain policy diversity during fine-tuning.",
    )
    parser.add_argument(
        "--storm-boost",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Override storm_test sampling weight in ScenarioGenerator (e.g. 0.18). "
             "Automatically redistributes weight from other scenarios. Requires --use-scenarios.",
    )
    parser.add_argument("--no-mock", action="store_true", help="Use live Redis state.")
    parser.add_argument(
        "--use-scenarios",
        action="store_true",
        help="Augment training with diverse traffic scenarios "
             "(traffic burst, node failure, thermal throttle, load imbalance).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--log-dir",
        type=str,
        default="src/drl/logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="src/drl/models/ppo_simulator.zip",
        help="Output model path (use src/drl/models/ppo_adaptive.zip for adaptive runs).",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Curriculum stage (1=easy … 4=adversarial, default: all scenarios).",
    )
    parser.add_argument(
        "--scenario-profile",
        default=None,
        choices=["balanced", "energy", "storm", "quality", "idle"],
        help="Objective weight profile applied on top of each scenario "
             "(balanced|energy|storm|quality).  Requires --use-scenarios.",
    )
    parser.add_argument(
        "--fixed-scenario",
        default=None,
        help="Lock training to a single scenario type (e.g. idle_cluster). "
             "Requires --use-scenarios.  Overrides curriculum stage sampling.",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        default=None,
        help="Path to bc_pretrained.zip (torch.save format) for BC warm-start. "
             "When provided, PPO uses net_arch=[128,128]+ReLU to match BC architecture.",
    )
    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help="Path to an existing MaskablePPO .zip to load and continue training from. "
             "Takes priority over --bc-checkpoint. Preserves all learned weights.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
