"""Generate decoded CelebA-HQ face samples from a Looped-DiT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import LoopedDiT, euler_sample
from train import _save_image_grid
from vae_loader import load_frozen_vae


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("samples.png"))
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_model(checkpoint_path: Path, device: torch.device) -> LoopedDiT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = checkpoint.get("config", {}).get("model_config", {})
    model = LoopedDiT(**model_config).to(device)
    state_dict = checkpoint.get("ema", checkpoint.get("model"))
    if state_dict is None:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate_samples(args: argparse.Namespace) -> Path:
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = _load_model(args.checkpoint, device)
    if args.class_label is None:
        labels = torch.randint(0, model.num_classes, (args.num_samples,), device=device)
    else:
        if not 0 <= args.class_label < model.num_classes:
            raise ValueError(f"class_label must be between 0 and {model.num_classes - 1}")
        labels = torch.full((args.num_samples,), args.class_label, device=device, dtype=torch.long)
    latents = euler_sample(
        model,
        shape=(args.num_samples, model.in_channels, *model.image_size),
        y=labels,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        num_loops=args.loops,
    )
    if args.vae_checkpoint is None:
        vae = load_frozen_vae(device=device)
    else:
        vae = load_frozen_vae(checkpoint_path=args.vae_checkpoint, device=device)
    images = vae.decode(latents)
    _save_image_grid(images, args.output)
    return args.output


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output = generate_samples(args)
    print(f"saved samples to {output}")


if __name__ == "__main__":
    main()
