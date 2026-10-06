"""Train Looped-DiT on cached CelebA-HQ VAE latents."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import CachedLatentDataset
from models import LoopedDiT, euler_sample, training_loss
from tracker import ExperimentTracker
from vae_loader import load_frozen_vae


class ModelEMA:
    """Exponential moving average of a model used for checkpointing and samples."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source_parameters = dict(model.named_parameters())
        for name, averaged in self.module.named_parameters():
            averaged.mul_(self.decay).add_(source_parameters[name].detach(), alpha=1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, averaged in self.module.named_buffers():
            averaged.copy_(source_buffers[name].detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state_dict)


def _save_image_grid(images: torch.Tensor, output_path: Path) -> None:
    from PIL import Image

    images = images.detach().float().cpu().clamp(-1.0, 1.0)
    images = ((images + 1.0) * 127.5).to(torch.uint8)
    batch, channels, height, width = images.shape
    if channels == 1:
        images = images.expand(batch, 3, height, width)
    elif channels != 3:
        raise ValueError("decoded images must have one or three channels")
    columns = max(1, math.ceil(math.sqrt(batch)))
    rows = math.ceil(batch / columns)
    grid = torch.zeros(3, rows * height, columns * width, dtype=torch.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        grid[:, row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    array = grid.permute(1, 2, 0).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path)


def _save_checkpoint(
    path: Path,
    model: LoopedDiT,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model.model_config(),
            "config": config,
        },
        path,
    )


@torch.no_grad()
def _periodic_sample(
    ema: ModelEMA,
    tracker: ExperimentTracker,
    step: int,
    device: torch.device,
    count: int,
    loops: int,
    cfg_scale: float,
    sample_steps: int,
    vae_checkpoint: str | None,
    vae: torch.nn.Module | None,
) -> torch.nn.Module | None:
    if vae is None:
        if vae_checkpoint is None:
            vae = load_frozen_vae(device=device)
        else:
            vae = load_frozen_vae(checkpoint_path=vae_checkpoint, device=device)
    labels = torch.randint(0, ema.module.num_classes, (count,), device=device)
    latents = euler_sample(
        ema.module,
        shape=(count, ema.module.in_channels, *ema.module.image_size),
        y=labels,
        cfg_scale=cfg_scale,
        steps=sample_steps,
        num_loops=loops,
    )
    images = vae.decode(latents)
    _save_image_grid(images, tracker.figures_dir / f"sample_step_{step:06d}.png")
    return vae


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("cached_data/latents.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/looped_dit"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=6.0)
    parser.add_argument("--loop-split", type=int, nargs=3, default=(2, 4, 2), metavar=("PRE", "CORE", "POST"))
    parser.add_argument("--num-classes", type=int, default=64)
    parser.add_argument("--class-dropout-prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=1_000)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--vae-checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--amp-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_resume(
    checkpoint_path: Path,
    model: LoopedDiT,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("epoch", 0))


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.max_steps <= 0 or args.batch_size <= 0:
        raise ValueError("max_steps and batch_size must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = CachedLatentDataset(args.cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = LoopedDiT(
        image_size=16,
        in_channels=32,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        loop_split=tuple(args.loop_split),
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
        default_num_loops=args.num_loops,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEMA(model, decay=args.ema_decay)
    ema.module.to(device)

    config = vars(args).copy()
    config["cache"] = str(args.cache)
    config["output_dir"] = str(args.output_dir)
    config["device"] = str(device)
    if args.resume is not None:
        config["resume"] = str(args.resume)
    tracker = ExperimentTracker(args.output_dir, config=config)
    start_step = 0
    epoch = 0
    if args.resume is not None:
        start_step, epoch = _load_resume(args.resume, model, ema, optimizer, device)

    use_amp = bool(args.amp_bf16 and device.type in {"cuda", "cpu"})
    vae: torch.nn.Module | None = None
    step = start_step
    while step < args.max_steps:
        model.train()
        for latents, labels in loader:
            if step >= args.max_steps:
                break
            latents = latents.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = training_loss(model, latents, y=labels, num_loops=args.num_loops)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}: {loss.item()}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)
            step += 1

            if step % args.log_interval == 0 or step == 1:
                tracker.log_metrics(
                    step,
                    epoch,
                    {"loss": loss.detach().float().item(), "lr": optimizer.param_groups[0]["lr"]},
                )
                tracker.plot_loss_curve()
            if step % args.checkpoint_interval == 0 or step == args.max_steps:
                _save_checkpoint(
                    tracker.checkpoints_dir / f"checkpoint_{step:06d}.pt",
                    model,
                    ema,
                    optimizer,
                    step,
                    epoch,
                    config,
                )
            if args.sample_interval > 0 and step % args.sample_interval == 0:
                try:
                    vae = _periodic_sample(
                        ema,
                        tracker,
                        step,
                        device,
                        args.sample_count,
                        args.num_loops,
                        args.cfg_scale,
                        args.sample_steps,
                        args.vae_checkpoint,
                        vae,
                    )
                except Exception as error:
                    print(f"warning: periodic sampling skipped at step {step}: {error}", file=sys.stderr)
        epoch += 1

    tracker.plot_loss_curve()
    with open(tracker.output_dir / "training_complete.json", "w") as file:
        json.dump({"step": step, "epoch": epoch}, file, indent=2)


if __name__ == "__main__":
    main()
