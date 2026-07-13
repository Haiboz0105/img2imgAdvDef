# Leveraging Image-to-Image Translation for Adversarial Defense

PyTorch implementation of the residual pix2pix-style defense described in *Leveraging Generalizability of Image-to-Image Translation for Enhanced Adversarial Defense*. The model learns to reconstruct clean images from adversarial inputs using a residual U-Net generator, a conditional PatchGAN discriminator, and a composite adversarial, L1, and VGG19 perceptual objective.

The repository operates on prepared paired images and includes training, reconstruction evaluation, metrics, checkpointing, device selection, and synthetic CPU validation.

## Installation

Python 3.10 or newer is required. Create an isolated environment and install the package with its test tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

Install PyTorch according to the official instructions for your operating system and hardware when the standard PyPI package is not appropriate. The repository does not require a particular GPU or CUDA version.

## Dataset format

Each sample is one side-by-side RGB image with an even width:

```text
| clean target (left) | adversarial input (right) |
```

Organize samples by split:

```text
<dataset-root>/
├── train/
│   └── *.png
├── val/
│   └── *.png
└── test/                 # optional
    └── *.png
```

The loader returns the right half as the adversarial input and the left half as the clean target. Both are resized and normalized to `[-1, 1]`. Training applies synchronized resize, random crop, and horizontal flip operations to both halves.

Inspect a prepared split before training:

```bash
python scripts/inspect_data.py --data-dir /path/to/paired-data/train
```

ImageNet and derived adversarial images are not distributed with this repository.

## Configuration

Copy `configs/default.yaml` to a Git-ignored file such as `local_config.yaml`, then set `dataset_path` and any experiment-specific output settings.

The public defaults use the standard pix2pix optimizer configuration:

```yaml
learning_rate: 0.0002
generator_learning_rate: 0.0002
discriminator_learning_rate: 0.0002
adam_beta1: 0.5
adam_beta2: 0.999
```

Other configuration fields control model depth, residual blocks, image size, augmentation, composite-loss weights, checkpoint frequency, SSIM behavior, and device selection.

## VGG19 perceptual loss

VGG19 perceptual loss is enabled in the default training configuration with pretrained torchvision weights. The configured feature endpoints correspond to Keras VGG19 layers 2–20.

Input tensors are prepared with Keras VGG19 conventions entirely in PyTorch:

1. convert RGB tensors from `[-1, 1]` to `[0, 255]`;
2. reorder channels from RGB to BGR;
3. subtract the ImageNet BGR means `[103.939, 116.779, 123.68]`.

The first training run may download torchvision's pretrained VGG19 weights into the standard PyTorch model cache. The synthetic smoke test uses a lightweight loss setting and does not download model weights.

## CPU smoke test

```bash
python scripts/run_smoke_test.py --device cpu
python -m pytest tests/test_smoke.py
```

The smoke test creates paired tensors in memory, performs one small generator/discriminator update, and computes reconstruction metrics. It does not require real data, checkpoints, TensorFlow, CUDA, or pretrained VGG weights.

## Training

```bash
python scripts/train.py --config local_config.yaml
```

Checkpoints are written under `<output_dir>/checkpoints/` at the configured interval. Dataset paths and generated outputs are excluded by `.gitignore`.

## Evaluation

```bash
python scripts/evaluate.py \
  --config local_config.yaml \
  --checkpoint outputs/checkpoints/epoch_0150.pt
```

Evaluation reports reconstruction L1, PSNR, and TensorFlow-style windowed SSIM as JSON.

## Hardware and device selection

The `device` configuration accepts:

- `auto`: CUDA when available, then Apple MPS, then CPU;
- `cpu`: portable CPU execution;
- `cuda`: NVIDIA CUDA execution when supported by the local PyTorch installation;
- `mps`: Apple silicon acceleration when supported;
- another valid explicit PyTorch device string.

No GPU ID, CUDA version, or `CUDA_VISIBLE_DEVICES` value is hard-coded.

## Citation

```bibtex
@article{zhang2026leveraging,
  title={Leveraging Generalizability of Image-to-Image Translation for Enhanced Adversarial Defense},
  author={Zhang, Haibo and Yao, Zhihua and Sakurai, Kouichi and Saitoh, Takeshi},
  journal={SN Computer Science},
  volume={7},
  number={6},
  pages={630},
  year={2026},
  publisher={Springer}
}
```

The same entry is available in `CITATION.bib`.

## License

This project is released under the MIT License. See `LICENSE`.
