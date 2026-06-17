#!/usr/bin/env bash
# Minimal DLA (arXiv 2606.10650) mechanism reproduction. CPU-only, no training.
# Installs a light dependency set (CPU torch) and runs the proof-of-concept.
set -euo pipefail

echo "[run.sh] installing CPU dependencies..."
python -m pip install --quiet --upgrade pip
# CPU torch keeps the image small; the PoC is pure tensor ops on small arrays.
python -m pip install --quiet \
  torch --index-url https://download.pytorch.org/whl/cpu || \
  python -m pip install --quiet torch
python -m pip install --quiet numpy einops jaxtyping typing_extensions matplotlib

echo "[run.sh] running DLA proof of concept..."
python dla_poc.py
