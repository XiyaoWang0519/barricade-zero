"""Deterministic vectorized exploration-noise helpers."""

from __future__ import annotations

import random

import numpy as np


def segmented_dirichlet_noise(
    rng: random.Random,
    segment_lengths,
    alpha: float,
) -> np.ndarray:
    """Sample concatenated Dirichlet vectors for positive segment lengths."""
    lengths = np.asarray(segment_lengths, dtype=np.intp)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("segment lengths must be a non-empty vector of positive integers")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    offsets = np.empty(lengths.size + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    samples = np.random.default_rng(rng.getrandbits(128)).gamma(
        alpha, 1.0, size=int(offsets[-1])
    )
    totals = np.add.reduceat(samples, offsets[:-1])
    return samples / np.repeat(totals, lengths)
