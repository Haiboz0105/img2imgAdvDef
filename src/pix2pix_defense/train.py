"""Minimal training components for the residual pix2pix scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader

from .datasets import SyntheticPairedDataset
from .metrics import peak_signal_to_noise_ratio, structural_similarity_index
from .models import PatchGANDiscriminator, UNetGenerator
from .utils import get_device, set_random_seed


def keras_vgg19_preprocess(inputs: torch.Tensor) -> torch.Tensor:
    """Apply Keras VGG19 preprocessing to NCHW RGB tensors in [-1, 1].

    The transform converts values to [0, 255], changes RGB to BGR channel order,
    and subtracts the ImageNet BGR means used by Keras VGG19.
    """
    if inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError("Keras VGG19 preprocessing expects NCHW tensors with 3 channels")
    values = (inputs + 1.0) * 127.5
    values = values[:, [2, 1, 0], :, :]
    means = values.new_tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
    return values - means


class VGG19PerceptualLoss(nn.Module):
    """Torchvision VGG19 feature loss with Keras-compatible preprocessing.

    Construction with pretrained weights can download them once into PyTorch's
    standard model cache. The synthetic smoke test does not construct this class.
    """

    def __init__(
        self,
        feature_indices: list[int],
        layer_weights: list[float] | None = None,
        pretrained: bool = True,
        preprocessing: str = "keras_vgg19",
    ) -> None:
        super().__init__()
        if not feature_indices:
            raise ValueError("feature_indices must not be empty")
        from torchvision.models import VGG19_Weights, vgg19

        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg19(weights=weights).features
        self.blocks = nn.ModuleList()
        previous = 0
        for index in sorted(feature_indices):
            self.blocks.append(nn.Sequential(*list(features.children())[previous : index + 1]))
            previous = index + 1
        self.layer_weights = layer_weights or list(range(len(self.blocks), 0, -1))
        if len(self.layer_weights) != len(self.blocks):
            raise ValueError("layer_weights must match feature_indices")
        if preprocessing not in {"keras_vgg19", "torchvision_imagenet"}:
            raise ValueError(f"Unsupported perceptual preprocessing: {preprocessing}")
        self.preprocessing = preprocessing
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.preprocessing == "keras_vgg19":
            prediction = keras_vgg19_preprocess(prediction)
            target = keras_vgg19_preprocess(target)
        else:
            prediction = ((prediction + 1) / 2 - self.mean) / self.std
            target = ((target + 1) / 2 - self.mean) / self.std
        total = prediction.new_zeros(())
        for block, weight in zip(self.blocks, self.layer_weights):
            prediction, target = block(prediction), block(target)
            # Keras mean_squared_error reduces the channels-last axis first; in
            # NCHW that is channel dimension 1. The notebook then sqrt()s this
            # per-pixel value before reducing its mean.
            channel_mse = (prediction - target).square().mean(dim=1)
            total = total + float(weight) * torch.sqrt(channel_mse + 1e-12).mean()
        return total


@dataclass
class GeneratorLosses:
    total: torch.Tensor
    gan: torch.Tensor
    l1: torch.Tensor
    perceptual: torch.Tensor


class CompositePix2PixLoss(nn.Module):
    def __init__(
        self,
        lambda_gan: float = 1.0,
        lambda_l1: float = 100.0,
        lambda_perceptual: float = 1.0,
        perceptual_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.perceptual_loss = perceptual_loss
        self.adversarial = nn.BCEWithLogitsLoss()

    def generator(self, fake_logits: torch.Tensor, generated: torch.Tensor, target: torch.Tensor) -> GeneratorLosses:
        gan = self.adversarial(fake_logits, torch.ones_like(fake_logits))
        l1 = functional.l1_loss(generated, target)
        perceptual = generated.new_zeros(())
        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(generated, target)
        total = self.lambda_gan * gan + self.lambda_l1 * l1 + self.lambda_perceptual * perceptual
        return GeneratorLosses(total=total, gan=gan, l1=l1, perceptual=perceptual)

    def discriminator(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real = self.adversarial(real_logits, torch.ones_like(real_logits))
        fake = self.adversarial(fake_logits, torch.zeros_like(fake_logits))
        return real + fake


def train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    batch: dict[str, Any],
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    criterion: CompositePix2PixLoss,
    device: torch.device,
) -> dict[str, float]:
    attacked = batch["input"].to(device)
    target = batch["target"].to(device)

    discriminator_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        detached_generated = generator(attacked)
    real_logits = discriminator(attacked, target)
    fake_logits = discriminator(attacked, detached_generated)
    discriminator_loss = criterion.discriminator(real_logits, fake_logits)
    discriminator_loss.backward()
    discriminator_optimizer.step()

    generator_optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    generated = generator(attacked)
    fake_logits = discriminator(attacked, generated)
    losses = criterion.generator(fake_logits, generated, target)
    losses.total.backward()
    generator_optimizer.step()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    return {
        "generator_total": float(losses.total.detach()),
        "generator_gan": float(losses.gan.detach()),
        "generator_l1": float(losses.l1.detach()),
        "generator_perceptual": float(losses.perceptual.detach()),
        "discriminator": float(discriminator_loss.detach()),
    }


def run_synthetic_smoke_test(device_name: str = "cpu") -> dict[str, float]:
    """Perform one tiny CPU-safe update with no files or external weights."""
    set_random_seed(7)
    device = get_device(device_name)
    dataset = SyntheticPairedDataset(length=1, image_size=32, seed=7)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    generator = UNetGenerator(
        base_channels=4,
        max_channels=16,
        num_down_blocks=3,
        residual=True,
        residual_layers_per_block=1,
        dropout_blocks=0,
    ).to(device)
    discriminator = PatchGANDiscriminator(
        base_channels=4, residual=True, residual_layers_per_block=1
    ).to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-7
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-7
    )
    criterion = CompositePix2PixLoss(lambda_perceptual=0.0)
    results = train_step(
        generator,
        discriminator,
        batch,
        generator_optimizer,
        discriminator_optimizer,
        criterion,
        device,
    )
    generator.eval()
    with torch.no_grad():
        attacked = batch["input"].to(device)
        target = batch["target"].to(device)
        reconstructed = generator(attacked)
        results["reconstruction_psnr"] = float(
            peak_signal_to_noise_ratio(reconstructed, target, data_range=2.0)
        )
        results["reconstruction_ssim"] = float(
            structural_similarity_index(reconstructed, target, data_range=2.0)
        )
    return results
