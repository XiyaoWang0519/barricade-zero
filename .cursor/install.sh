#!/usr/bin/env bash
# Idempotent Cloud Agent setup for Barricade Zero.
set -euo pipefail

cd "$(dirname "$0")/.."

# ensurepip for `python3 -m venv` is shipped in a separate Debian/Ubuntu package.
if ! dpkg -s python3.12-venv >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.12-venv
fi

# Create the virtual environment only when it is missing.
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

# Build the optional C++ rules backend (falls back to Python if absent).
.venv/bin/python scripts/build_native.py
