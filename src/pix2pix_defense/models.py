"""U-Net generators and conditional PatchGAN discriminator."""

from __future__ import annotations

import torch
from torch import nn


def initialize_pix2pix_weights(module: nn.Module) -> None:
    """Match the legacy normal(0, 0.02) convolution initialization."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SingleValueSafeBatchNorm2d(nn.BatchNorm2d):
    """Keras-style BatchNorm that also supports a batch-size-1, 1x1 bottleneck.

    PyTorch stores an unbiased running variance and rejects one scalar per channel.
    Keras uses population variance, accepts that bottleneck, and defines momentum
    as the old-state weight. This forward path follows the Keras conventions.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
    ) -> None:
        # Keras momentum=0.99 weights the old state; PyTorch momentum=0.01
        # weights the new batch and is therefore the corresponding convention.
        super().__init__(num_features, eps=eps, momentum=momentum)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dimensions = (0, 2, 3)
        if self.training:
            mean = inputs.mean(dim=dimensions)
            variance = inputs.var(dim=dimensions, unbiased=False)
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    self.running_mean.lerp_(mean.detach(), self.momentum)
                    self.running_var.lerp_(variance.detach(), self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        shape = (1, -1, 1, 1)
        outputs = (inputs - mean.view(shape)) * torch.rsqrt(variance.view(shape) + self.eps)
        if self.affine:
            outputs = outputs * self.weight.view(shape) + self.bias.view(shape)
        return outputs


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
        negative_slope: float = 0.3,
        norm_momentum: float = 0.01,
        norm_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if use_norm:
            layers.append(
                SingleValueSafeBatchNorm2d(
                    out_channels, momentum=norm_momentum, eps=norm_epsilon
                )
            )
        layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ResidualDownBlock(nn.Module):
    """Legacy downsample followed by configurable same-width residual convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual_layers: int = 7,
        use_norm: bool = True,
        negative_slope: float = 0.3,
        norm_momentum: float = 0.01,
        norm_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.down = DownBlock(
            in_channels,
            out_channels,
            use_norm=use_norm,
            negative_slope=negative_slope,
            norm_momentum=norm_momentum,
            norm_epsilon=norm_epsilon,
        )
        self.residual_layers = nn.ModuleList(
            nn.Sequential(
                # TensorFlow SAME with kernel=4/stride=1 pads 1 before and 2 after.
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
                SingleValueSafeBatchNorm2d(
                    out_channels, momentum=norm_momentum, eps=norm_epsilon
                ),
                nn.LeakyReLU(negative_slope, inplace=True),
            )
            for _ in range(residual_layers)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.down(inputs)
        for residual in self.residual_layers:
            outputs = outputs + residual(outputs)
        return outputs


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: bool = False,
        norm_momentum: float = 0.01,
        norm_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            SingleValueSafeBatchNorm2d(
                out_channels, momentum=norm_momentum, eps=norm_epsilon
            ),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNetGenerator(nn.Module):
    """Configurable baseline or residual pix2pix U-Net generator."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 512,
        num_down_blocks: int = 8,
        residual: bool = True,
        residual_layers_per_block: int = 7,
        dropout_blocks: int = 3,
        negative_slope: float = 0.3,
        norm_momentum: float = 0.01,
        norm_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        if num_down_blocks < 2:
            raise ValueError("num_down_blocks must be at least 2")
        channels = [min(base_channels * (2**index), max_channels) for index in range(num_down_blocks)]
        down_blocks: list[nn.Module] = []
        in_channels = input_channels
        block_type = ResidualDownBlock if residual else DownBlock
        for index, out_channels in enumerate(channels):
            use_norm = index >= 2  # Matches the primary residual notebook's first two blocks.
            kwargs = {"residual_layers": residual_layers_per_block} if residual else {}
            down_blocks.append(
                block_type(
                    in_channels,
                    out_channels,
                    use_norm=use_norm,
                    negative_slope=negative_slope,
                    norm_momentum=norm_momentum,
                    norm_epsilon=norm_epsilon,
                    **kwargs,
                )
            )
            in_channels = out_channels
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks: list[nn.Module] = []
        current_channels = channels[-1]
        for index, skip_channels in enumerate(reversed(channels[:-1])):
            up_blocks.append(
                UpBlock(
                    current_channels,
                    skip_channels,
                    dropout=index < dropout_blocks,
                    norm_momentum=norm_momentum,
                    norm_epsilon=norm_epsilon,
                )
            )
            current_channels = skip_channels + skip_channels
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(current_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.apply(initialize_pix2pix_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        skips: list[torch.Tensor] = []
        for block in self.down_blocks:
            outputs = block(outputs)
            skips.append(outputs)
        for block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            outputs = block(outputs)
            outputs = torch.cat((outputs, skip), dim=1)
        return self.output_layer(outputs)


class PatchGANDiscriminator(nn.Module):
    """Conditional baseline or residual PatchGAN discriminator."""

    def __init__(
        self,
        input_channels: int = 3,
        target_channels: int = 3,
        base_channels: int = 64,
        residual: bool = True,
        residual_layers_per_block: int = 7,
        negative_slope: float = 0.3,
        norm_momentum: float = 0.01,
        norm_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        combined = input_channels + target_channels
        block_type = ResidualDownBlock if residual else DownBlock
        block_kwargs = {
            "negative_slope": negative_slope,
            "norm_momentum": norm_momentum,
            "norm_epsilon": norm_epsilon,
        }
        if residual:
            block_kwargs["residual_layers"] = residual_layers_per_block
        self.features = nn.Sequential(
            block_type(combined, base_channels, use_norm=False, **block_kwargs),
            block_type(base_channels, base_channels * 2, use_norm=True, **block_kwargs),
            block_type(base_channels * 2, base_channels * 4, use_norm=True, **block_kwargs),
            nn.ZeroPad2d(1),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, bias=False),
            SingleValueSafeBatchNorm2d(
                base_channels * 8, momentum=norm_momentum, eps=norm_epsilon
            ),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1),
        )
        self.apply(initialize_pix2pix_weights)

    def forward(self, attacked_input: torch.Tensor, candidate_target: torch.Tensor) -> torch.Tensor:
        return self.features(torch.cat((attacked_input, candidate_target), dim=1))


def build_models_from_config(config: dict) -> tuple[UNetGenerator, PatchGANDiscriminator]:
    """Build both networks without coupling package code to a CLI script."""
    model = config["model"]
    residual_generator = model["generator_type"] == "residual_unet"
    generator = UNetGenerator(
        input_channels=int(config["input_channels"]),
        output_channels=int(config["output_channels"]),
        base_channels=int(model["base_channels"]),
        max_channels=int(model["max_channels"]),
        num_down_blocks=int(model["num_down_blocks"]),
        residual=residual_generator,
        residual_layers_per_block=int(model["residual_layers_per_block"]),
        dropout_blocks=int(model["dropout_blocks"]),
        negative_slope=float(model["leaky_relu_slope"]),
        norm_momentum=float(model["batchnorm_pytorch_momentum"]),
        norm_epsilon=float(model["batchnorm_epsilon"]),
    )
    discriminator = PatchGANDiscriminator(
        input_channels=int(config["input_channels"]),
        target_channels=int(config["output_channels"]),
        base_channels=int(model["base_channels"]),
        residual=bool(model["discriminator_residual_blocks"]),
        residual_layers_per_block=int(model["residual_layers_per_block"]),
        negative_slope=float(model["leaky_relu_slope"]),
        norm_momentum=float(model["batchnorm_pytorch_momentum"]),
        norm_epsilon=float(model["batchnorm_epsilon"]),
    )
    return generator, discriminator
