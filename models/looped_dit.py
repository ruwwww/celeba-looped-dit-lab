"""Looped DiT backbone for 32-channel 16x16 VAE latents with stabilized dynamics.

Incorporates:
1. QK-Norm (RMSNorm on Query and Key) to prevent attention logit drift.
2. Exact token-aligned XSA (Exclusive Self-Attention) per head.
3. Bounded adaLN modulation to prevent compound exponential scaling across recurrent loops.
4. Loop-boundary RMSNorm to stabilize hidden representations over multiple loop passes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Create sinusoidal embeddings for scalar timesteps."""
    timesteps = timesteps.reshape(-1).float()
    half = dim // 2
    if half == 0:
        return timesteps[:, None]
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    arguments = timesteps[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep features followed by a learned 2-layer projection."""

    def __init__(self, hidden_size: int, frequency_size: int | None = None) -> None:
        super().__init__()
        self.frequency_size = frequency_size or hidden_size
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = RMSNorm(hidden_size)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        sin_cos = timestep_embedding(timesteps, self.frequency_size)
        return self.norm(self.mlp(sin_cos.to(dtype=self.mlp[0].weight.dtype)))


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        intermediate_size = max(1, int(hidden_size * mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK-Norm and per-head Exclusive Self-Attention (XSA)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        regulate_attention: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.regulate_attention = regulate_attention

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, tokens, head_dim]
        query, key, value = qkv.unbind(0)

        # QK-Norm prevents query-key inner products from drifting to extremes
        query = self.q_norm(query)
        key = self.k_norm(key)

        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )

        if self.regulate_attention:
            # Per-head Exclusive Self-Attention (XSA, Zhai 2026):
            # Remove from each token's attention output the component along its own value vector.
            v_hat = F.normalize(value.float(), dim=-1, eps=1e-6)
            att_float = attention.float()
            parallel = (att_float * v_hat).sum(dim=-1, keepdim=True) * v_hat
            attention = (att_float - parallel).to(dtype=attention.dtype)

        attention = attention.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(attention)


class AdaLNBlock(nn.Module):
    """Transformer block with adaptive LayerNorm and scale bounding."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        regulate_attention: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            regulate_attention=regulate_attention,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout=dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_att, scale_att, gate_att, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        # Numerical protection: clamp scale to [-2, 2] and gate to [-4, 4]
        # Prevents compounding exponential explosions across recurrent loops
        scale_att = torch.clamp(scale_att, min=-2.0, max=2.0)
        scale_mlp = torch.clamp(scale_mlp, min=-2.0, max=2.0)
        gate_att = torch.clamp(gate_att, min=-4.0, max=4.0)
        gate_mlp = torch.clamp(gate_mlp, min=-4.0, max=4.0)

        attention_input = self.norm1(x) * (1.0 + scale_att[:, None, :]) + shift_att[:, None, :]
        x = x + gate_att[:, None, :] * self.attention(attention_input)

        mlp_input = self.norm2(x) * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        return x + gate_mlp[:, None, :] * self.mlp(mlp_input)


class LoopedDiT(nn.Module):
    """Parameter-efficient DiT with a physically shared looped core and stabilized dynamics.

    Default split: (pre=2, core=4, post=2) with hidden_size=512, mlp_ratio=6.0.
    Physical parameters: ~49.0M parameters.
    Effective recurrent depth: 2 + (4 * 4) + 2 = 20 transformer blocks.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] = 16,
        in_channels: int = 32,
        patch_size: int = 1,
        hidden_size: int = 512,
        depth: int | None = None,
        num_heads: int | None = None,
        mlp_ratio: float = 6.0,
        loop_split: tuple[int, int, int] = (2, 4, 2),
        num_classes: int = 64,
        class_dropout_prob: float = 0.15,
        dropout: float = 0.0,
        default_num_loops: int = 4,
    ) -> None:
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size must be a positive integer or a (height, width) pair")
        if patch_size <= 0 or image_size[0] % patch_size or image_size[1] % patch_size:
            raise ValueError("patch_size must divide both image dimensions")
        if in_channels <= 0 or hidden_size <= 0 or num_classes <= 0:
            raise ValueError("in_channels, hidden_size, and num_classes must be positive")
        if len(loop_split) != 3 or any(blocks <= 0 for blocks in loop_split):
            raise ValueError("loop_split must contain three positive block counts")
        if depth is not None and depth != sum(loop_split):
            raise ValueError("depth must equal sum(loop_split)")
        if not 0.0 <= class_dropout_prob <= 1.0:
            raise ValueError("class_dropout_prob must be between 0 and 1")
        if default_num_loops <= 0:
            raise ValueError("default_num_loops must be positive")

        self.image_size = tuple(image_size)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads or max(1, hidden_size // 64)
        self.mlp_ratio = mlp_ratio
        self.loop_split = tuple(loop_split)
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.default_num_loops = default_num_loops
        self.num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

        self.x_embedder = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.time_embed = TimestepEmbedder(hidden_size)
        self.class_embed = nn.Embedding(num_classes + 1, hidden_size)
        self.condition_norm = RMSNorm(hidden_size)
        self.loop_norm = RMSNorm(hidden_size)

        pre_depth, core_depth, post_depth = self.loop_split
        self.pre_blocks = nn.ModuleList(
            AdaLNBlock(hidden_size, self.num_heads, mlp_ratio, dropout=dropout, regulate_attention=False)
            for _ in range(pre_depth)
        )
        self.core_blocks = nn.ModuleList(
            AdaLNBlock(hidden_size, self.num_heads, mlp_ratio, dropout=dropout, regulate_attention=True)
            for _ in range(core_depth)
        )
        self.post_blocks = nn.ModuleList(
            AdaLNBlock(hidden_size, self.num_heads, mlp_ratio, dropout=dropout, regulate_attention=False)
            for _ in range(post_depth)
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.output_projection = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def model_config(self) -> dict[str, int | float | list[int]]:
        """Return JSON-serializable constructor arguments for checkpoints."""
        return {
            "image_size": list(self.image_size),
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "depth": sum(self.loop_split),
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "loop_split": list(self.loop_split),
            "num_classes": self.num_classes,
            "class_dropout_prob": self.class_dropout_prob,
            "default_num_loops": self.default_num_loops,
        }

    def _condition(self, batch_size: int, timesteps: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        timesteps = timesteps.to(device=self.pos_embed.device)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.numel() != batch_size:
            raise ValueError("t must have one value per batch element")

        null_label = self.num_classes
        if labels is None:
            labels = torch.full((batch_size,), null_label, device=self.pos_embed.device, dtype=torch.long)
        else:
            labels = labels.to(device=self.pos_embed.device, dtype=torch.long).reshape(-1)
            if labels.numel() == 1:
                labels = labels.expand(batch_size)
            if labels.numel() != batch_size:
                raise ValueError("y must have one label per batch element")
            if torch.any((labels < 0) | (labels > null_label)):
                raise ValueError(f"y values must be between 0 and {null_label}")
            if self.training and self.class_dropout_prob > 0.0:
                dropped = torch.rand(batch_size, device=labels.device) < self.class_dropout_prob
                labels = torch.where(dropped, torch.full_like(labels, null_label), labels)

        condition = self.time_embed(timesteps) + self.class_embed(labels)
        return self.condition_norm(condition)

    def _decode(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.post_blocks:
            hidden = block(hidden, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        scale = torch.clamp(scale, min=-2.0, max=2.0)
        hidden = self.final_norm(hidden) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        patches = self.output_projection(hidden)
        batch, tokens, channels = patches.shape
        grid_height = self.image_size[0] // self.patch_size
        grid_width = self.image_size[1] // self.patch_size
        patches = patches.reshape(
            batch,
            grid_height,
            grid_width,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch,
            self.in_channels,
            self.image_size[0],
            self.image_size[1],
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        num_loops: int | None = None,
        exit_loops: Iterable[int] = (),
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Predict velocity and optionally decode intermediate loop exits."""
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, channels, height, width]")
        batch, channels, height, width = x.shape
        if channels != self.in_channels or (height, width) != self.image_size:
            raise ValueError(
                f"expected x shape [B, {self.in_channels}, {self.image_size[0]}, {self.image_size[1]}]"
            )
        num_loops = self.default_num_loops if num_loops is None else int(num_loops)
        if num_loops <= 0:
            raise ValueError("num_loops must be positive")
        requested = tuple(sorted(set(int(loop) for loop in exit_loops)))
        if any(loop <= 0 or loop > num_loops for loop in requested):
            raise ValueError("exit_loops must be between 1 and num_loops")

        condition = self._condition(batch, t, y)
        hidden = self.x_embedder(x).flatten(2).transpose(1, 2)
        hidden = hidden + self.pos_embed.to(dtype=hidden.dtype)
        for block in self.pre_blocks:
            hidden = block(hidden, condition)

        exits: dict[int, torch.Tensor] = {}
        for loop_index in range(1, num_loops + 1):
            # Normalize hidden state across loop transitions to prevent compounding drift
            hidden = self.loop_norm(hidden)
            for block in self.core_blocks:
                hidden = block(hidden, condition)
            if loop_index in requested:
                exits[loop_index] = self._decode(hidden, condition)

        final_output = exits.get(num_loops)
        if final_output is None:
            final_output = self._decode(hidden, condition)
        return final_output, exits


__all__ = ["LoopedDiT"]
