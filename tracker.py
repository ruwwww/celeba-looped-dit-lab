"""Standard logger and figure plotter for generative experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


class ExperimentTracker:
    """Manages experiment directory structure, metrics.jsonl, and figure generation."""

    def __init__(self, output_dir: str | Path, config: dict[str, Any] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.figures_dir = self.output_dir / "figures"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.config_file = self.output_dir / "config.json"

        if config is not None and not self.config_file.exists():
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)

    def log_metrics(self, step: int, epoch: int, metrics: dict[str, float]) -> None:
        payload = {
            "step": int(step),
            "epoch": int(epoch),
            "timestamp": time.time(),
            **{k: float(v) for k, v in metrics.items()},
        }
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def plot_loss_curve(self, output_path: str | Path | None = None) -> Path | None:
        """Plot training loss curves from metrics.jsonl."""
        if not self.metrics_file.exists():
            return None

        steps: list[int] = []
        losses: list[float] = []
        extra_keys: dict[str, list[float]] = {}

        with open(self.metrics_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "step" in data and "loss" in data:
                        steps.append(data["step"])
                        losses.append(data["loss"])
                        for k, v in data.items():
                            if k.startswith("loss_") and isinstance(v, (int, float)):
                                extra_keys.setdefault(k, []).append(v)
                except Exception:
                    continue

        if not steps:
            return None

        plt.figure(figsize=(10, 5), dpi=150)
        plt.plot(steps, losses, label="Total Loss", color="#1f77b4", alpha=0.8, linewidth=1.5)
        for k, v in extra_keys.items():
            if len(v) == len(steps):
                plt.plot(steps, v, label=k, linestyle="--", alpha=0.6, linewidth=1.0)

        plt.xlabel("Training Steps")
        plt.ylabel("Optimal Transport / Flow Loss")
        plt.title(f"Training Convergence Curve - {self.output_dir.name}")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        target_path = Path(output_path) if output_path else self.figures_dir / "loss_curve.png"
        plt.savefig(target_path)
        plt.close()
        return target_path
