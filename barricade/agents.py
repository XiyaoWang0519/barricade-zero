"""Simple non-neural agents used to validate and benchmark the engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .state import GameState


@dataclass
class RandomAgent:
    rng: random.Random = field(default_factory=random.Random)

    def choose_action(self, state: GameState) -> int:
        return self.rng.choice(state.legal_actions())


class ShortestPathAgent:
    def choose_action(self, state: GameState) -> int:
        player = state.turn
        candidates = []
        for action in state.legal_pawn_actions():
            successor = state.apply_action(action)
            distance = successor.shortest_path_distance(player)
            candidates.append((distance if distance is not None else state.size**2, action))
        return min(candidates)[1]


def play_game(agent0, agent1, state: GameState | None = None, max_plies: int = 1000) -> tuple[int, int]:
    state = state or GameState.initial()
    agents = (agent0, agent1)
    for ply in range(1, max_plies + 1):
        state = state.apply_action(agents[state.turn].choose_action(state))
        if state.is_terminal():
            return state.winner, ply
    raise RuntimeError(f"game exceeded {max_plies} plies")