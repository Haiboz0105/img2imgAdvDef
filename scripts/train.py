#!/usr/bin/env python3
"""Train the residual pix2pix model from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pix2pix_defense.datasets import PairedImageDataset  # noqa: E402
from pix2pix_defense.models import build_models_from_config  # noqa: E402
from pix2pix_defense.train import CompositePix2PixLoss, VGG19PerceptualLoss, train_step  # noqa: E402
from pix2pix_defense.utils import (  # noqa: E402
    checkpoint_state,
    get_device,
    load_config,
    resolve_learning_rates,
    save_checkpoint,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/default.yaml")
    parser.add_argument("--device", help="Override config device (auto/cpu/cuda/mps/explicit torch device)")
    parser.add_argument("--max-steps", type=int, help="Optional development cap; not a parity setting")
    return parser.parse_args()


def build_criterion(config: dict, device: torch.device) -> CompositePix2PixLoss:
    loss_config = config["loss"]
    perceptual_config = loss_config["perceptual_loss"]
    perceptual = None
    if loss_config["use_perceptual_loss"] and perceptual_config["enabled"]:
        if perceptual_config["network"] != "vgg19":
            raise ValueError("Only the vgg19 perceptual network is supported")
        perceptual = VGG19PerceptualLoss(
            feature_indices=list(perceptual_config["layers"]),
            layer_weights=list(perceptual_config["layer_weights"]) or None,
            pretrained=bool(perceptual_config["pretrained"]),
            preprocessing=str(perceptual_config["preprocessing"]),
        ).to(device)
    return CompositePix2PixLoss(
        lambda_gan=float(loss_config["lambda_gan"]),
        lambda_l1=float(loss_config["lambda_l1"]),
        lambda_perceptual=float(loss_config["lambda_perceptual"]),
        perceptual_loss=perceptual,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["dataset_path"] is None:
        raise ValueError("Set dataset_path in the config or a derived config before training")
    set_random_seed(int(config["random_seed"]))
    device = get_device(args.device or config["device"])
    split_root = Path(config["dataset_path"]).expanduser() / config["train_split"]
    augmentation = config["augmentation"]
    dataset = PairedImageDataset(
        split_root,
        image_size=int(config["image_size"]),
        jitter_size=int(config["jitter_size"]),
        training=True,
        random_jitter=bool(augmentation["random_jitter"]),
        random_horizontal_flip=bool(augmentation["random_horizontal_flip"]),
        extensions=config["file_extensions"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
    )
    generator, discriminator = build_models_from_config(config)
    generator, discriminator = generator.to(device), discriminator.to(device)
    generator_lr, discriminator_lr = resolve_learning_rates(config)
    betas = (float(config["adam_beta1"]), float(config["adam_beta2"]))
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=generator_lr,
        betas=betas,
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=discriminator_lr,
        betas=betas,
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = build_criterion(config, device)
    output_dir = REPOSITORY_ROOT / config["output_dir"]
    global_step = 0
    for epoch in range(int(config["num_epochs"])):
        generator.train()
        discriminator.train()
        latest: dict[str, float] = {}
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}"):
            latest = train_step(
                generator,
                discriminator,
                batch,
                generator_optimizer,
                discriminator_optimizer,
                criterion,
                device,
            )
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        print(json.dumps({"epoch": epoch + 1, **latest}, sort_keys=True))
        if (epoch + 1) % int(config["checkpoint_every"]) == 0:
            state = checkpoint_state(
                generator, discriminator, generator_optimizer, discriminator_optimizer, epoch + 1, config
            )
            save_checkpoint(state, output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt")
        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
