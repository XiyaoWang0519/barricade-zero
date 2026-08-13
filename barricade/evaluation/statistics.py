"""Small dependency-free statistics helpers for paired game evaluation."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ScoreInterval:
    mean: float
    lower: float
    upper: float
    confidence: float
    pairs: int
    resamples: int
    method: str = "paired_bootstrap_percentile"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceDecision:
    status: str
    eligible: bool
    threshold: float
    minimum_pairs: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def paired_bootstrap_interval(
    pair_scores: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> ScoreInterval:
    """Bootstrap opening-pair scores, preserving the two-game pairing."""
    scores = [float(score) for score in pair_scores]
    if not scores:
        raise ValueError("at least one opening-pair score is required")
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("scores must be between zero and one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    mean = sum(scores) / len(scores)
    rng = random.Random(seed)
    count = len(scores)
    sampled_means = sorted(
        sum(scores[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return ScoreInterval(
        mean=mean,
        lower=_quantile(sampled_means, tail),
        upper=_quantile(sampled_means, 1.0 - tail),
        confidence=confidence,
        pairs=count,
        resamples=resamples,
    )


def confidence_decision(
    interval: ScoreInterval,
    *,
    threshold: float = 0.5,
    minimum_pairs: int = 100,
) -> ConfidenceDecision:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if minimum_pairs <= 0:
        raise ValueError("minimum_pairs must be positive")
    if interval.pairs < minimum_pairs:
        return ConfidenceDecision(
            status="insufficient_data",
            eligible=False,
            threshold=threshold,
            minimum_pairs=minimum_pairs,
            reason=(
                f"only {interval.pairs} paired openings were evaluated; "
                f"at least {minimum_pairs} are required"
            ),
        )
    if interval.lower > threshold:
        return ConfidenceDecision(
            status="promote",
            eligible=True,
            threshold=threshold,
            minimum_pairs=minimum_pairs,
            reason=(
                f"the {interval.confidence:.0%} lower confidence bound "
                f"{interval.lower:.4f} exceeds {threshold:.4f}"
            ),
        )
    if interval.upper < threshold:
        return ConfidenceDecision(
            status="reject",
            eligible=True,
            threshold=threshold,
            minimum_pairs=minimum_pairs,
            reason=(
                f"the {interval.confidence:.0%} upper confidence bound "
                f"{interval.upper:.4f} is below {threshold:.4f}"
            ),
        )
    return ConfidenceDecision(
        status="inconclusive",
        eligible=True,
        threshold=threshold,
        minimum_pairs=minimum_pairs,
        reason=(
            f"the {interval.confidence:.0%} confidence interval crosses "
            f"{threshold:.4f}"
        ),
    )
