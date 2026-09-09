#!/bin/bash
# One-shot setup for the HSPP replication on AnuraSet.
#
# What this does, in order:
#   1. Creates the pinned-Python virtualenv (.venv).
#   2. Installs dependencies from requirements.txt.
#   3. Extracts the audio archives into data/ (if not already extracted).
#   4. Builds the grouped 3-way split + leakage asserts (src.splits).
#   5. Precomputes HPSS features into data/features.h5 (~16 min on the
#      reference machine; see README.md for CUDA-machine expectations).
#   6. Runs the verification test suite (19 tests).
#
# Usage:
#   ./setup.sh                 # full setup, uses ~/.local/bin/python3.11
#   PYTHON_BIN=/path/to/python3.11 ./setup.sh
#
# Safe to re-run: every step is idempotent (uv venv reuses an existing venv,
# the split/precompute steps check for their output files first).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.local/bin/python3.11}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.11 not found at $PYTHON_BIN" >&2
    echo "Python 3.11 is pinned deliberately -- see README.md 'Instalación'." >&2
    echo "Install it, or pass PYTHON_BIN=/path/to/python3.11 ./setup.sh" >&2
    exit 1
fi

echo "== [1/6] Virtual environment (.venv, python=$PYTHON_BIN) =="
if [ ! -d .venv ]; then
    uv venv --python "$PYTHON_BIN" .venv
else
    echo ".venv already exists, skipping creation"
fi

echo "== [2/6] Dependencies (requirements.txt) =="
# NOTE on device support: this installs torch via PyPI's default resolution.
# On a machine with an NVIDIA GPU, install the CUDA build of torch FIRST if
# you want a specific CUDA version pinned -- see README.md "Device support:
# NVIDIA / CUDA" for the exact command. Re-running this script afterwards
# will not downgrade an already-installed CUDA wheel.
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

echo "== [3/6] Audio data (data/train, data/test) =="
mkdir -p data
if [ -f train.7z ] && [ ! -d data/train ]; then
    echo "Extracting train.7z (62,191 clips, 7.7 GB)..."
    tar -xf train.7z -C data/
else
    echo "data/train already present or train.7z not found, skipping"
fi
if [ -f test.7z ] && [ ! -d data/test ]; then
    echo "Extracting test.7z (31,187 clips)..."
    tar -xf test.7z -C data/
else
    echo "data/test already present or test.7z not found, skipping"
fi

echo "== [4/6] Grouped 3-way split =="
if [ ! -d data/splits ]; then
    .venv/bin/python -m src.splits
else
    echo "data/splits already exists, skipping (delete it to force a rebuild)"
fi

echo "== [5/6] HPSS feature precompute (data/features.h5) =="
if [ ! -f data/features.h5 ]; then
    .venv/bin/python -m src.precompute
else
    echo "data/features.h5 already exists, skipping (delete it to force a rebuild)"
fi

echo "== [6/6] Verification tests (19 tests, ~4.5 min) =="
.venv/bin/python -m pytest tests/ -q

echo
echo "Setup complete. Detected training device:"
.venv/bin/python -c "from src.engine import pick_device; print(' ->', pick_device())"
echo
echo "Next steps:"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i .venv/bin/python -u -m src.train --arm dual"
echo "  ./run_final_ablation.sh"
