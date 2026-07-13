"""Reconstruction metrics for tensors normalized to a known value range."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def mean_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return functional.l1_loss(prediction, target)


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor, data_range: float = 2.0
) -> torch.Tensor:
    """Batch-mean PSNR; `data_range=2` is correct for tensors in [-1, 1]."""
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    maximum = torch.as_tensor(data_range, dtype=prediction.dtype, device=prediction.device)
    values = 20 * torch.log10(maximum) - 10 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).eps))
    return values.mean()


def global_structural_similarity_index(
    prediction: torch.Tensor, target: torch.Tensor, data_range: float = 2.0
) -> torch.Tensor:
    """Compute the optional whole-image SSIM approximation."""
    dims = (-2, -1)
    mean_x, mean_y = prediction.mean(dims), target.mean(dims)
    var_x = prediction.var(dims, unbiased=False)
    var_y = target.var(dims, unbiased=False)
    covariance = ((prediction - mean_x[..., None, None]) * (target - mean_y[..., None, None])).mean(dims)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).mean()


def _gaussian_kernel(
    window_size: int, sigma: float, channels: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    coordinates = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def structural_similarity_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Windowed SSIM matching the documented `tf.image.ssim` formulation.

    TensorFlow uses an 11x11 Gaussian window (sigma 1.5), VALID spatial
    convolution, and averages the resulting SSIM map. Inputs must be NCHW and at
    least `window_size` pixels in both spatial dimensions.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have identical NCHW shapes")
    if min(prediction.shape[-2:]) < window_size:
        raise ValueError(f"SSIM inputs must be at least {window_size}x{window_size}")
    channels = prediction.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, prediction.dtype, prediction.device)
    mean_x = functional.conv2d(prediction, kernel, groups=channels)
    mean_y = functional.conv2d(target, kernel, groups=channels)
    mean_x2, mean_y2, mean_xy = mean_x.square(), mean_y.square(), mean_x * mean_y
    variance_x = functional.conv2d(prediction.square(), kernel, groups=channels) - mean_x2
    variance_y = functional.conv2d(target.square(), kernel, groups=channels) - mean_y2
    covariance = functional.conv2d(prediction * target, kernel, groups=channels) - mean_xy
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    luminance = (2 * mean_xy + c1) / (mean_x2 + mean_y2 + c1)
    contrast_structure = (2 * covariance + c2) / (variance_x + variance_y + c2)
    return (luminance * contrast_structure).mean()
