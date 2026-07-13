"""Reconstruction-only evaluation for the initial scaffold."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .metrics import (
    global_structural_similarity_index,
    mean_absolute_error,
    peak_signal_to_noise_ratio,
    structural_similarity_index,
)


@torch.no_grad()
def evaluate_reconstruction(
    generator: torch.nn.Module,
    data_loader: Any,
    device: torch.device,
    data_range: float = 2.0,
    ssim_variant: str = "tensorflow_windowed",
) -> dict[str, float]:
    generator.eval()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for batch in data_loader:
        attacked = batch["input"].to(device)
        target = batch["target"].to(device)
        prediction = generator(attacked)
        totals["l1"] += float(mean_absolute_error(prediction, target))
        totals["psnr"] += float(peak_signal_to_noise_ratio(prediction, target, data_range))
        if ssim_variant == "tensorflow_windowed":
            totals["ssim"] += float(structural_similarity_index(prediction, target, data_range))
        elif ssim_variant == "global":
            totals["ssim_global"] += float(
                global_structural_similarity_index(prediction, target, data_range)
            )
        else:
            raise ValueError(f"Unknown SSIM variant: {ssim_variant}")
        batches += 1
    if not batches:
        raise ValueError("Evaluation data loader is empty")
    return {name: value / batches for name, value in totals.items()}
