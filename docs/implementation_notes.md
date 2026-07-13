# Implementation Notes

## Scope

This release trains and evaluates an image-reconstruction defense from prepared side-by-side clean/adversarial pairs. Dataset generation and classifier benchmarking are kept separate from the reconstruction package.

## Architecture

The default model uses an eight-level residual U-Net generator. Every encoder level applies seven same-width residual convolutions after downsampling. The conditional PatchGAN discriminator uses corresponding residual downsampling blocks. TensorFlow-style SAME padding, Keras BatchNorm conventions, LeakyReLU slope `0.3`, and Adam epsilon `1e-7` are represented explicitly.

## Objective

The generator objective is:

```text
lambda_gan * adversarial + lambda_l1 * L1 + lambda_perceptual * VGG19
```

The default weights are `1`, `100`, and `1`, respectively. VGG19 features use the configured endpoints and descending layer weights. Keras VGG19 preprocessing converts normalized RGB tensors to mean-centered BGR values in `[0,255]` scale.

## Metrics

PSNR uses data range `2.0` for tensors in `[-1,1]`. SSIM uses an 11×11 Gaussian window, sigma `1.5`, and VALID convolution, matching the TensorFlow SSIM formulation.

## Reproducibility

The configuration records random seed, optimizer parameters, network dimensions, augmentation, loss weights, and evaluation settings. Checkpoint files include both networks, both optimizer states, epoch number, format version, and the active configuration.
