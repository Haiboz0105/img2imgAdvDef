# Upload Manifest

## Included files

```text
.gitignore
CITATION.bib
LICENSE
README.md
UPLOAD_MANIFEST.md
pyproject.toml
requirements.txt
requirements-dev.txt
configs/default.yaml
docs/implementation_notes.md
scripts/evaluate.py
scripts/inspect_data.py
scripts/run_smoke_test.py
scripts/train.py
src/pix2pix_defense/__init__.py
src/pix2pix_defense/datasets.py
src/pix2pix_defense/evaluate.py
src/pix2pix_defense/metrics.py
src/pix2pix_defense/models.py
src/pix2pix_defense/train.py
src/pix2pix_defense/utils.py
tests/conftest.py
tests/test_smoke.py
```

## Intentionally excluded

- Step 1–4 audit, migration, disk, dependency, cleanup, smoke, and release-management reports: internal process records rather than user documentation.
- `docs/original_code_map.md` and `docs/cleanup_notes.md`: local notebook archaeology and disk-management notes.
- `requirements-tf-legacy.txt` and `legacy_tf_notes/`: the public package contains no TensorFlow code.
- `src/pix2pix_defense/attacks.py`, `src/pix2pix_defense/classifier_eval.py`, and `scripts/evaluate_classifier.py`: incomplete APIs outside the prepared-pair reconstruction scope.
- `src/pix2pix_defense/plotting.py`: optional helper not used by the documented public commands.
- `.venv/`, caches, AppleDouble metadata, editable-install metadata, notebooks, datasets, checkpoints, weights, logs, and generated experiment outputs.

## Release checks

- No private local path was found in the upload tree.
- No included file is larger than 10 MiB.
- `.venv/` is excluded and ignored.
- TensorFlow is not required or imported by the PyTorch release.
- The MIT `LICENSE` and `CITATION.bib` are included.
- CPU smoke test: **passed** with status `ok`.
- Pytest: **5 passed**.
- Python syntax, package import, YAML/TOML parsing, and Ruff checks: **passed**.

## Manual checks before publication

1. Confirm that `Copyright (c) 2026 Haibo Zhang` is the correct copyright ownership statement for the repository.
2. Confirm author/contact metadata and the repository description on GitHub.
3. Review dataset acquisition and redistribution instructions against ImageNet licensing.
4. Inspect `git status --short --ignored` before the first commit to confirm that only manifest-listed files are staged.
