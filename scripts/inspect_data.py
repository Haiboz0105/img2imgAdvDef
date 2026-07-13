#!/usr/bin/env python3
"""Inspect the shape and split convention of a paired-image directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from PIL import Image  # noqa: E402

from pix2pix_defense.datasets import discover_images, split_paired_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of side-by-side paired images")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of images to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_images(args.data_dir)
    inspected: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for path in paths[: args.limit]:
        try:
            with Image.open(path) as paired:
                attacked, target = split_paired_image(paired)
                inspected.append({"path": str(path), "pair_size": paired.size, "half_size": target.size})
                if attacked.size != target.size:
                    raise ValueError("input and target halves differ in size")
        except Exception as error:  # Inspection should report malformed files together.
            failures.append({"path": str(path), "error": str(error)})
    print(
        json.dumps(
            {"root": str(args.data_dir), "total_images": len(paths), "inspected": inspected, "failures": failures},
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
