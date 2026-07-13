"""Paired-image datasets compatible with the legacy side-by-side format."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg")


def discover_images(root: str | Path, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> list[Path]:
    """Return supported image files recursively in deterministic path order."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Paired-image directory does not exist: {root}")
    allowed = {extension.lower() for extension in extensions}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def split_paired_image(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Split `[clean target | adversarial input]` into `(input, target)` images."""
    image = image.convert("RGB")
    width, height = image.size
    if width % 2:
        raise ValueError(f"Paired image width must be even, got {width}")
    midpoint = width // 2
    target = image.crop((0, 0, midpoint, height))
    attacked_input = image.crop((midpoint, 0, width, height))
    return attacked_input, target


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    """Convert RGB PIL data to CHW float tensor normalized to [-1, 1]."""
    array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


class PairedImageDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Load horizontal clean/adversarial pairs with synchronized augmentation."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        jitter_size: int = 286,
        training: bool = False,
        random_jitter: bool = True,
        random_horizontal_flip: bool = True,
        extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    ) -> None:
        if image_size <= 0 or jitter_size < image_size:
            raise ValueError("image_size must be positive and jitter_size must be >= image_size")
        self.paths = discover_images(root, extensions)
        if not self.paths:
            raise ValueError(f"No paired images found under {Path(root)}")
        self.image_size = image_size
        self.jitter_size = jitter_size
        self.training = training
        self.random_jitter = random_jitter
        self.random_horizontal_flip = random_horizontal_flip

    def __len__(self) -> int:
        return len(self.paths)

    def _transform_pair(self, attacked: Image.Image, target: Image.Image) -> tuple[Image.Image, Image.Image]:
        resampling = Image.Resampling.NEAREST
        if self.training and self.random_jitter:
            size = (self.jitter_size, self.jitter_size)
            attacked = attacked.resize(size, resampling)
            target = target.resize(size, resampling)
            limit = self.jitter_size - self.image_size
            left = random.randint(0, limit)
            top = random.randint(0, limit)
            box = (left, top, left + self.image_size, top + self.image_size)
            attacked, target = attacked.crop(box), target.crop(box)
        else:
            size = (self.image_size, self.image_size)
            attacked = attacked.resize(size, resampling)
            target = target.resize(size, resampling)

        if self.training and self.random_horizontal_flip and random.random() < 0.5:
            attacked = attacked.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return attacked, target

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with Image.open(path) as paired:
            attacked, target = split_paired_image(paired)
            attacked, target = self._transform_pair(attacked, target)
        return {
            "input": pil_to_normalized_tensor(attacked),
            "target": pil_to_normalized_tensor(target),
            "path": str(path),
        }


class SyntheticPairedDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Small deterministic in-memory dataset for tests; not research data."""

    def __init__(self, length: int = 2, image_size: int = 64, channels: int = 3, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.rand(length, channels, image_size, image_size, generator=generator) * 2 - 1
        self.targets = torch.clamp(self.inputs * 0.75 + 0.1, -1, 1)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {"input": self.inputs[index], "target": self.targets[index], "path": f"synthetic:{index}"}
