from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import Dataset


class CachedLatentDataset(Dataset):
    """Loads pre-encoded VAE latents and DINOv2 cluster labels directly in memory."""

    def __init__(self, cache_path: str | Path):
        super().__init__()
        data = torch.load(cache_path, weights_only=True)
        self.latents = data["latents"]  # [N, 32, 16, 16] bf16
        self.labels = data["labels"].long()  # [N] int64

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.labels[idx]
