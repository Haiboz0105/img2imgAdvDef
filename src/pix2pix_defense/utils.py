"""Configuration, reproducibility, device, and checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def get_device(device: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda/mps or another explicit PyTorch device string."""
    normalized = device.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if normalized.startswith("mps"):
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS device requested but MPS is unavailable")
    try:
        return torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid PyTorch device string: {device}") from error


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def resolve_learning_rates(config: dict[str, Any]) -> tuple[float, float]:
    common = float(config["learning_rate"])
    return (
        float(config["generator_learning_rate"] or common),
        float(config["discriminator_learning_rate"] or common),
    )


def checkpoint_state(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "config": config,
    }


def save_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return destination


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)
