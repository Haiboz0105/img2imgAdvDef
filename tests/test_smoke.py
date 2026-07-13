"""Fast synthetic tests; no real data, downloads, TensorFlow, CUDA, or file output."""

from __future__ import annotations

import math

import torch
from PIL import Image
from torch.utils.data import DataLoader

from pix2pix_defense.datasets import SyntheticPairedDataset, split_paired_image
from pix2pix_defense.evaluate import evaluate_reconstruction
from pix2pix_defense.metrics import peak_signal_to_noise_ratio, structural_similarity_index
from pix2pix_defense.models import PatchGANDiscriminator, UNetGenerator
from pix2pix_defense.train import keras_vgg19_preprocess, run_synthetic_smoke_test
from pix2pix_defense.utils import get_device


def test_side_by_side_pair_order() -> None:
    paired = Image.new("RGB", (16, 8), color=(0, 0, 0))
    paired.paste(Image.new("RGB", (8, 8), color=(255, 0, 0)), (0, 0))
    paired.paste(Image.new("RGB", (8, 8), color=(0, 0, 255)), (8, 0))
    attacked, target = split_paired_image(paired)
    assert target.getpixel((0, 0)) == (255, 0, 0)
    assert attacked.getpixel((0, 0)) == (0, 0, 255)


def test_model_shapes_on_cpu() -> None:
    generator = UNetGenerator(
        base_channels=4, max_channels=16, num_down_blocks=3, residual_layers_per_block=1, dropout_blocks=0
    )
    discriminator = PatchGANDiscriminator(
        base_channels=4, residual=True, residual_layers_per_block=1
    )
    inputs = torch.randn(1, 3, 32, 32)
    outputs = generator(inputs)
    logits = discriminator(inputs, outputs)
    assert outputs.shape == inputs.shape
    assert logits.shape[:2] == (1, 1)


def test_synthetic_train_step_is_finite() -> None:
    losses = run_synthetic_smoke_test("cpu")
    assert losses
    assert all(math.isfinite(value) for value in losses.values())


def test_reconstruction_metrics_and_evaluation() -> None:
    target = torch.zeros(1, 3, 16, 16)
    assert peak_signal_to_noise_ratio(target, target).isfinite()
    assert torch.isclose(structural_similarity_index(target, target), torch.tensor(1.0))
    dataset = SyntheticPairedDataset(length=1, image_size=32)
    generator = UNetGenerator(
        base_channels=4, max_channels=16, num_down_blocks=3, residual_layers_per_block=1, dropout_blocks=0
    )
    metrics = evaluate_reconstruction(generator, DataLoader(dataset, batch_size=1), get_device("cpu"))
    assert set(metrics) == {"l1", "psnr", "ssim"}


def test_keras_vgg19_preprocessing() -> None:
    red_rgb = torch.tensor([[[[1.0]], [[-1.0]], [[-1.0]]]])
    actual = keras_vgg19_preprocess(red_rgb).flatten()
    expected = torch.tensor([-103.939, -116.779, 131.32])
    assert torch.allclose(actual, expected, atol=1e-3)
