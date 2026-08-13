"""AlphaZero self-play trajectory generation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .encoding import rotate_policy
from .mcts import MCTS
from .state import GameState


@dataclass(frozen=True)
class TrainingExample:
    state: GameState
    policy: list[float]
    outcome: float


def _sample(policy: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for action, probability in enumerate(policy):
        cumulative += probability
        if threshold <= cumulative:
            return action
    return max(range(len(policy)), key=policy.__getitem__)


def play_self_play_game(
    state: GameState,
    mcts: MCTS,
    rng: random.Random | None = None,
    exploration_plies: int = 12,
    max_plies: int = 1000,
    repetition_limit: int = 3,
) -> list[TrainingExample]:
    rng = rng or random.Random()
    trajectory: list[tuple[GameState, list[float], int]] = []
    repetitions: dict[bytes, int] = {}
    for ply in range(max_plies):
        key = state.canonical_key()
        repetitions[key] = repetitions.get(key, 0) + 1
        if repetitions[key] >= repetition_limit:
            return [TrainingExample(recorded, policy, 0.0) for recorded, policy, _ in trajectory]
        temperature = 1.0 if ply < exploration_plies else 0.0
        result = mcts.search(state, temperature=temperature, add_noise=ply < exploration_plies)
        canonical_policy = result.policy if state.turn == 0 else rotate_policy(result.policy, state.size)
        trajectory.append((state.canonical(), canonical_policy, state.turn))
        action = _sample(result.policy, rng) if temperature > 0 else result.best_action
        state = state.apply_action(action)
        if state.is_terminal():
            return [
                TrainingExample(recorded, policy, 1.0 if player == state.winner else -1.0)
                for recorded, policy, player in trajectory
            ]
    raise RuntimeError(f"self-play game exceeded {max_plies} plies")
