#!/usr/bin/env python3
"""Build the optional C++ rules backend."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "rules.cpp"
OUTPUT = ROOT / "native" / "libbarricade_rules.so"

subprocess.run(
    ["g++", "-O3", "-DNDEBUG", "-std=c++20", "-fPIC", "-shared", str(SOURCE), "-o", str(OUTPUT)],
    check=True,
)
print(OUTPUT)
