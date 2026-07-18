from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import torch


DEFAULT_BC_PATH = Path("src/drl/models/drl_bc.pt")
DEFAULT_PPO_PATH = Path("src/drl/models/drl_ppo.zip")


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        visible = {k: v for k, v in value.items() if k != ":serialized:"}
        return json.dumps(visible, indent=2, default=str)
    return str(value)


def inspect_bc_model(path: Path) -> None:
    print(f"\n=== BC model: {path} ===")
    if not path.exists():
        print("File not found.")
        return

    print(f"size: {path.stat().st_size:,} bytes")
    checkpoint = torch.load(path, map_location="cpu")
    print(f"type: {type(checkpoint).__name__}")

    if not isinstance(checkpoint, dict):
        print("This checkpoint is not a dict, so no keys can be listed.")
        return

    print(f"keys: {list(checkpoint.keys())}")

    if "state_dim" in checkpoint:
        print(f"state_dim: {checkpoint['state_dim']}")
    if "action_dims" in checkpoint:
        print(f"action_dims: {checkpoint['action_dims']}")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        print("model_state_dict: not found")
        return

    print(f"model_state_dict tensors: {len(state_dict)}")
    for name, tensor in state_dict.items():
        if hasattr(tensor, "shape"):
            print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        else:
            print(f"  {name}: {type(tensor).__name__}")


def inspect_ppo_model(path: Path) -> None:
    print(f"\n=== PPO model: {path} ===")
    if not path.exists():
        print("File not found.")
        return

    print(f"size: {path.stat().st_size:,} bytes")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        print("archive files:")
        for name in names:
            info = archive.getinfo(name)
            print(f"  {name}: {info.file_size:,} bytes")

        if "_stable_baselines3_version" in names:
            version = archive.read("_stable_baselines3_version").decode().strip()
            print(f"stable_baselines3_version: {version}")

        if "system_info.txt" in names:
            print("\nsystem_info.txt:")
            print(archive.read("system_info.txt").decode().strip())

        if "data" not in names:
            print("data: not found")
            return

        data = json.loads(archive.read("data"))
        interesting_keys = [
            "policy_class",
            "observation_space",
            "action_space",
            "n_envs",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "learning_rate",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "seed",
            "num_timesteps",
        ]

        print("\ntraining/model metadata:")
        for key in interesting_keys:
            if key in data:
                print(f"{key}: {_format_value(data[key])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect exported DRL model metadata.")
    parser.add_argument("--bc", type=Path, default=DEFAULT_BC_PATH, help="Path to BC .pt checkpoint.")
    parser.add_argument("--ppo", type=Path, default=DEFAULT_PPO_PATH, help="Path to PPO .zip model.")
    parser.add_argument("--skip-bc", action="store_true", help="Do not inspect the BC checkpoint.")
    parser.add_argument("--skip-ppo", action="store_true", help="Do not inspect the PPO archive.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_bc:
        inspect_bc_model(args.bc)
    if not args.skip_ppo:
        inspect_ppo_model(args.ppo)


if __name__ == "__main__":
    main()
