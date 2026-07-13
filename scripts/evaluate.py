#!/usr/bin/env python3
"""Evaluate a PyTorch generator checkpoint on paired validation images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from pix2pix_defense.datasets import PairedImageDataset  # noqa: E402
from pix2pix_defense.evaluate import evaluate_reconstruction  # noqa: E402
from pix2pix_defense.models import build_models_from_config  # noqa: E402
from pix2pix_defense.utils import get_device, load_checkpoint, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/default.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", help="Override config device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["dataset_path"] is None:
        raise ValueError("Set dataset_path before evaluation")
    device = get_device(args.device or config["device"])
    generator, _ = build_models_from_config(config)
    checkpoint = load_checkpoint(args.checkpoint, device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device)
    dataset = PairedImageDataset(
        Path(config["dataset_path"]).expanduser() / config["validation_split"],
        image_size=int(config["image_size"]),
        jitter_size=int(config["jitter_size"]),
        training=False,
        extensions=config["file_extensions"],
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), num_workers=int(config["num_workers"]))
    metrics = evaluate_reconstruction(
        generator,
        loader,
        device,
        data_range=float(config["evaluation"]["data_range"]),
        ssim_variant=str(config["evaluation"]["ssim_variant"]),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
