#!/usr/bin/env python3
"""Run one synthetic update without data, downloads, or accelerator requirements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pix2pix_defense.train import run_synthetic_smoke_test  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="PyTorch device; CPU is the smoke-test default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_synthetic_smoke_test(args.device)
    print(json.dumps({"status": "ok", "device": args.device, "losses": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
