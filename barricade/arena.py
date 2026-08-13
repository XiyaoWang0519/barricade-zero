"""Deterministic checkpoint arena with balanced starting sides."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .mcts import Evaluator, MCTS
from .state import GameState


@dataclass(frozen=True)
class ArenaResult:
    candidate_wins: int
    champion_wins: int
    draws: int
    candidate_as_first_games: int
    candidate_as_second_games: int

    @property
    def games(self) -> int:
        return self.candidate_wins + self.champion_wins + self.draws

    @property
    def candidate_score(self) -> float:
        return (self.candidate_wins + 0.5 * self.draws) / self.games if self.games else 0.0


class Arena:
    def __init__(
        self,
        simulations: int = 50,
        board_size: int = 5,
        walls_per_player: int = 2,
        max_plies: int = 500,
        rng: random.Random | None = None,
    ) -> None:
        self.simulations = simulations
        self.board_size = board_size
        self.walls_per_player = walls_per_player
        self.max_plies = max_plies
        self.rng = rng or random.Random()

    def _play(self, evaluators: tuple[Evaluator, Evaluator]) -> int | None:
        state = GameState.initial(self.board_size, self.walls_per_player)
        repetitions: dict[bytes, int] = {}
        searches = tuple(
            MCTS(evaluator, simulations=self.simulations, rng=random.Random(self.rng.getrandbits(64)))
            for evaluator in evaluators
        )
        for _ in range(self.max_plies):
            key = state.canonical_key()
            repetitions[key] = repetitions.get(key, 0) + 1
            if repetitions[key] >= 3:
                return None
            result = searches[state.turn].search(state, temperature=0, add_noise=False)
            state = state.apply_action(result.best_action)
            if state.is_terminal():
                return state.winner
        return None

    def play_match(
        self, candidate: Evaluator, champion: Evaluator, games: int = 20
    ) -> ArenaResult:
        if games <= 0 or games % 2:
            raise ValueError("arena games must be a positive even number")
        candidate_wins = champion_wins = draws = 0
        for game in range(games):
            candidate_first = game % 2 == 0
            evaluators = (candidate, champion) if candidate_first else (champion, candidate)
            winner = self._play(evaluators)
            if winner is None:
                draws += 1
            elif (winner == 0) == candidate_first:
                candidate_wins += 1
            else:
                champion_wins += 1
        return ArenaResult(
            candidate_wins,
            champion_wins,
            draws,
            candidate_as_first_games=games // 2,
            candidate_as_second_games=games // 2,
        )
