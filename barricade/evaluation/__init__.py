"""Reproducible, training-independent model evaluation."""

from .arena import MatchResult, play_paired_match
from .openings import OpeningSuite, generate_opening_suite, load_opening_suite
from .predictions import PredictionMetrics, evaluate_predictions

__all__ = [
    "MatchResult",
    "OpeningSuite",
    "PredictionMetrics",
    "evaluate_predictions",
    "generate_opening_suite",
    "load_opening_suite",
    "play_paired_match",
]
