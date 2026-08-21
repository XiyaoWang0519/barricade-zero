"""Paired-opening arena for reproducible checkpoint strength evaluation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from barricade.match_play import make_factory_side, play_two_player_games
from barricade.state import GameState

from .openings import OpeningSuite
from .statistics import (
    ConfidenceDecision,
    ScoreInterval,
    confidence_decision,
    paired_bootstrap_interval,
)


@dataclass(frozen=True)
class GameResult:
    opening_id: str
    candidate_player: int
    winner: int | None
    plies: int
    termination: str
    candidate_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OpeningPairResult:
    opening_id: str
    games: tuple[GameResult, GameResult]
    candidate_score: float

    def to_dict(self) -> dict:
        return {
            "opening_id": self.opening_id,
            "candidate_score": self.candidate_score,
            "games": [game.to_dict() for game in self.games],
        }


@dataclass(frozen=True)
class MatchResult:
    candidate: str
    opponent: str
    opening_suite_sha256: str
    seed: int
    max_plies: int
    wins: int
    draws: int
    losses: int
    interval: ScoreInterval
    decision: ConfidenceDecision
    pairs: tuple[OpeningPairResult, ...]

    @property
    def opening_pairs(self) -> int:
        return len(self.pairs)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "opponent": self.opponent,
            "opening_suite_sha256": self.opening_suite_sha256,
            "seed": self.seed,
            "max_plies": self.max_plies,
            "opening_pairs": self.opening_pairs,
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "score": self.score,
            "interval": self.interval.to_dict(),
            "decision": self.decision.to_dict(),
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


def play_paired_match(
    candidate_factory: Any,
    opponent_factory: Any,
    suite: OpeningSuite,
    *,
    max_plies: int = 500,
    seed: int = 1,
    confidence: float = 0.95,
    bootstrap_resamples: int = 10_000,
    minimum_pairs: int = 100,
    threshold: float = 0.5,
) -> MatchResult:
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    rng = random.Random(seed)
    states: list[GameState] = []
    seat_to_side: list[tuple[int, int]] = []
    candidate_seeds: list[int] = []
    opponent_seeds: list[int] = []
    pair_slots: list[tuple[str, int]] = []
    for opening in suite.openings:
        candidate_seed = rng.getrandbits(64)
        opponent_seed = rng.getrandbits(64)
        for candidate_player in (0, 1):
            states.append(opening.state)
            seat_to_side.append((0, 1) if candidate_player == 0 else (1, 0))
            candidate_seeds.append(candidate_seed)
            opponent_seeds.append(opponent_seed)
            pair_slots.append((opening.opening_id, candidate_player))
    sides = (
        make_factory_side(candidate_factory, states, candidate_seeds),
        make_factory_side(opponent_factory, states, opponent_seeds),
    )
    outcomes = play_two_player_games(sides, states, seat_to_side, max_plies)
    pair_results = []
    wins = draws = losses = 0
    pair_games: dict[str, list[GameResult]] = {}
    for (opening_id, candidate_player), outcome in zip(pair_slots, outcomes):
        if outcome.winner is None:
            score = 0.5
            draws += 1
        elif outcome.winner == candidate_player:
            score = 1.0
            wins += 1
        else:
            score = 0.0
            losses += 1
        game = GameResult(
            opening_id,
            candidate_player,
            outcome.winner,
            outcome.plies,
            outcome.termination,
            score,
        )
        pair_games.setdefault(opening_id, []).append(game)
    for opening in suite.openings:
        games = pair_games[opening.opening_id]
        pair_score = sum(game.candidate_score for game in games) / 2.0
        pair_results.append(
            OpeningPairResult(opening.opening_id, tuple(games), pair_score)
        )
    interval = paired_bootstrap_interval(
        [pair.candidate_score for pair in pair_results],
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    decision = confidence_decision(
        interval,
        threshold=threshold,
        minimum_pairs=minimum_pairs,
    )
    return MatchResult(
        candidate=candidate_factory.name,
        opponent=opponent_factory.name,
        opening_suite_sha256=suite.sha256,
        seed=seed,
        max_plies=max_plies,
        wins=wins,
        draws=draws,
        losses=losses,
        interval=interval,
        decision=decision,
        pairs=tuple(pair_results),
    )
