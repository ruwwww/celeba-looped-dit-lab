"""Optimal-transport flow matching and Euler sampling utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def deep_supervision_weights(
    num_loops: int,
    weighting: str = "exponential",
) -> torch.Tensor:
    """Return one multiplier for each intermediate exit, indexed by ``r - 1``.

    The final exit has an implicit loss weight of one.  Exponential weighting
    makes earlier exits weaker while keeping the final intermediate multiplier
    at one, so the final prediction remains the dominant supervision signal.
    """
    if num_loops <= 0:
        raise ValueError("num_loops must be positive")
    if weighting == "exponential":
        return torch.pow(2.0, torch.arange(1 - num_loops, 1, dtype=torch.float32))
    if weighting == "uniform":
        return torch.ones(num_loops, dtype=torch.float32)
    raise ValueError("weighting must be 'exponential' or 'uniform'")


def _model_dtype(model: nn.Module, fallback: torch.dtype) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return fallback


def _prediction(result: Any) -> torch.Tensor:
    if isinstance(result, tuple):
        return result[0]
    return result


def _exit_weight(
    exit_weights: torch.Tensor | Mapping[int, float] | list[float] | tuple[float, ...],
    loop_index: int,
) -> float:
    if isinstance(exit_weights, Mapping):
        return float(exit_weights.get(loop_index, 0.0))
    if loop_index > len(exit_weights):
        return 0.0
    return float(exit_weights[loop_index - 1])


def training_loss(
    model: nn.Module,
    x1: torch.Tensor,
    y: torch.Tensor | None = None,
    exit_weights: torch.Tensor | Mapping[int, float] | list[float] | tuple[float, ...] | None = None,
    t: torch.Tensor | None = None,
    x0: torch.Tensor | None = None,
    num_loops: int | None = None,
    weighting: str = "exponential",
) -> torch.Tensor:
    """Compute OT flow-matching MSE with optional deep-supervision exits."""
    if x1.ndim < 2:
        raise ValueError("x1 must include a batch dimension")
    batch_size = x1.shape[0]
    if t is None:
        t = torch.rand(batch_size, device=x1.device, dtype=torch.float32)
    else:
        t = t.to(device=x1.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(batch_size)
        if t.numel() != batch_size:
            raise ValueError("t must have one value per batch element")
    if x0 is None:
        x0 = torch.randn_like(x1)
    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have the same shape")

    num_loops = int(num_loops or getattr(model, "default_num_loops", 4))
    if num_loops <= 0:
        raise ValueError("num_loops must be positive")
    if exit_weights is None:
        exit_weights = deep_supervision_weights(num_loops, weighting)

    view_shape = (batch_size,) + (1,) * (x1.ndim - 1)
    interpolation = t.reshape(view_shape)
    x_t = interpolation * x1 + (1.0 - interpolation) * x0
    target = x1 - x0
    model_input = x_t.to(dtype=_model_dtype(model, x_t.dtype))
    requested_exits = tuple(range(1, num_loops))
    prediction, exits = model(
        model_input,
        t,
        y=y,
        num_loops=num_loops,
        exit_loops=requested_exits,
    )
    loss = F.mse_loss(prediction.float(), target.float())
    for loop_index, exit_prediction in exits.items():
        weight = _exit_weight(exit_weights, loop_index)
        if weight:
            loss = loss + weight * F.mse_loss(exit_prediction.float(), target.float())
    return loss


class FlowMatching(nn.Module):
    """Small stateful wrapper that binds a velocity model to the loss API."""

    def __init__(
        self,
        model: nn.Module,
        num_loops: int = 4,
        weighting: str = "exponential",
    ) -> None:
        super().__init__()
        self.model = model
        self.num_loops = num_loops
        self.weighting = weighting

    def loss(
        self,
        x1: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        exit_weights: torch.Tensor | Mapping[int, float] | list[float] | tuple[float, ...] | None = None,
        t: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return training_loss(
            self.model,
            x1,
            y=y,
            exit_weights=exit_weights,
            t=t,
            x0=x0,
            num_loops=self.num_loops,
            weighting=self.weighting,
        )

    def forward(self, x1: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.loss(x1, y=y)


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    shape: tuple[int, ...],
    y: torch.Tensor | None = None,
    cfg_scale: float = 4.0,
    steps: int = 50,
    num_loops: int = 4,
) -> torch.Tensor:
    """Integrate ``dx/dt = model(x, t)`` from Gaussian noise to data space."""
    if len(shape) != 4 or any(d <= 0 for d in shape):
        raise ValueError("shape must be a positive [batch, channels, height, width] tuple")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if num_loops <= 0:
        raise ValueError("num_loops must be positive")

    try:
        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype if parameter.dtype.is_floating_point else torch.float32
    except StopIteration:
        device = y.device if y is not None else torch.device("cpu")
        dtype = torch.float32
    labels = y.to(device=device) if y is not None else None
    samples = torch.randn(shape, device=device, dtype=dtype)
    was_training = model.training
    model.eval()
    try:
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((shape[0],), step / steps, device=device, dtype=torch.float32)
            if labels is not None and cfg_scale != 1.0:
                unconditional = _prediction(
                    model(samples, time, y=None, num_loops=num_loops, exit_loops=())
                )
                conditional = _prediction(
                    model(samples, time, y=labels, num_loops=num_loops, exit_loops=())
                )
                velocity = unconditional + cfg_scale * (conditional - unconditional)
            else:
                velocity = _prediction(
                    model(samples, time, y=labels, num_loops=num_loops, exit_loops=())
                )
            samples = samples + dt * velocity.to(dtype=samples.dtype)
    finally:
        model.train(was_training)
    return samples


__all__ = [
    "FlowMatching",
    "deep_supervision_weights",
    "euler_sample",
    "training_loss",
]
