"""Concurrent self-play synchronized by move, with batched MCTS inference."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .batched_mcts import BatchedMCTS
from .encoding import rotate_policy
from .self_play import TrainingExample
from .state import GameState


@dataclass
class _Game:
    state: GameState
    trajectory: list[tuple[GameState, list[float], int]] = field(default_factory=list)
    repetitions: dict[bytes, int] = field(default_factory=dict)
    plies: int = 0


@dataclass(frozen=True)
class ConcurrentSelfPlayResult:
    examples: list[TrainingExample]
    game_lengths: list[int]
    wins: tuple[int, int]
    draws: int
    forward_calls: int
    positions_evaluated: int

    @property
    def games(self) -> int:
        return len(self.game_lengths)

    @property
    def average_inference_batch_size(self) -> float:
        return self.positions_evaluated / self.forward_calls if self.forward_calls else 0.0


def _sample(policy: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for action, probability in enumerate(policy):
        cumulative += probability
        if threshold <= cumulative:
            return action
    return max(range(len(policy)), key=policy.__getitem__)


def _finish(game: _Game, winner: int | None) -> list[TrainingExample]:
    return [
        TrainingExample(state, policy, 0.0 if winner is None else (1.0 if player == winner else -1.0))
        for state, policy, player in game.trajectory
    ]


def play_concurrent_games(
    evaluator,
    games: int,
    simulations: int,
    board_size: int = 5,
    walls_per_player: int = 2,
    rng: random.Random | None = None,
    exploration_plies: int = 12,
    max_plies: int = 500,
    repetition_limit: int = 3,
    max_plies_as_draw: bool = True,
) -> ConcurrentSelfPlayResult:
    if games <= 0:
        raise ValueError("games must be positive")
    rng = rng or random.Random()
    active = [_Game(GameState.initial(board_size, walls_per_player)) for _ in range(games)]
    completed_examples: list[TrainingExample] = []
    lengths: list[int] = []
    wins = [0, 0]
    draws = 0
    starting_calls = evaluator.forward_calls
    starting_positions = evaluator.positions_evaluated

    while active:
        searchable = []
        still_active = []
        for game in active:
            key = game.state.canonical_key()
            game.repetitions[key] = game.repetitions.get(key, 0) + 1
            if game.repetitions[key] >= repetition_limit:
                completed_examples.extend(_finish(game, None))
                lengths.append(game.plies)
                draws += 1
            elif game.plies >= max_plies:
                if not max_plies_as_draw:
                    raise RuntimeError(f"concurrent game exceeded {max_plies} plies")
                completed_examples.extend(_finish(game, None))
                lengths.append(game.plies)
                draws += 1
            else:
                searchable.append(game)
                still_active.append(game)
        active = still_active
        if not searchable:
            continue

        temperature = 1.0 if min(game.plies for game in searchable) < exploration_plies else 0.0
        search = BatchedMCTS(
            evaluator,
            simulations=simulations,
            rng=random.Random(rng.getrandbits(64)),
        )
        results = search.search_batch(
            [game.state for game in searchable],
            temperature=temperature,
            add_noise=temperature > 0,
        )
        survivors = []
        for game, result in zip(searchable, results):
            canonical_policy = (
                result.policy if game.state.turn == 0 else rotate_policy(result.policy, board_size)
            )
            game.trajectory.append((game.state.canonical(), canonical_policy, game.state.turn))
            action = _sample(result.policy, rng) if temperature > 0 else result.best_action
            game.state = game.state.apply_action(action)
            game.plies += 1
            if game.state.is_terminal():
                completed_examples.extend(_finish(game, game.state.winner))
                lengths.append(game.plies)
                wins[game.state.winner] += 1
            else:
                survivors.append(game)
        active = survivors

    return ConcurrentSelfPlayResult(
        completed_examples,
        lengths,
        (wins[0], wins[1]),
        draws,
        evaluator.forward_calls - starting_calls,
        evaluator.positions_evaluated - starting_positions,
    )
