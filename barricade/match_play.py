"""Wave-batched two-player matches for training and evaluation arenas."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .backend import load_rules_backend
from .batched_mcts import BatchedMCTS
from .state import GameState


@dataclass(frozen=True)
class GameOutcome:
    winner: int | None
    plies: int
    termination: str


@dataclass
class _LiveGame:
    state: GameState
    seat_to_side: tuple[int, int]
    repetitions: dict[bytes, int] = field(default_factory=dict)
    plies: int = 0
    outcome: GameOutcome | None = None


class SearchSide:
    """One evaluator's trees across many in-progress games."""

    def __init__(
        self,
        evaluator,
        simulations: int,
        seed: int,
        states: list[GameState],
    ) -> None:
        if not states:
            raise ValueError("search side requires at least one root")
        if not hasattr(evaluator, "evaluate_batch"):
            raise TypeError("search side requires an evaluator with evaluate_batch")
        backend = getattr(evaluator, "encoding_backend", None) or load_rules_backend()
        self._python = BatchedMCTS(
            evaluator,
            simulations=simulations,
            rng=random.Random(seed),
            rules_backend=backend,
        )
        self.roots = self._python.create_roots(states)

    def choose_actions(self, indices: list[int]) -> list[int]:
        if not indices:
            return []
        roots = [self.roots[index] for index in indices]
        results = self._python.search_roots(roots, temperature=0.0, add_noise=False)
        return [int(result.best_action) for result in results]

    def observe_actions(
        self, indices: list[int], actions: list[int], _states: list[GameState]
    ) -> None:
        if not indices:
            return
        roots = [self.roots[index] for index in indices]
        advanced = self._python.advance_roots_known_legal(roots, actions)
        for index, root in zip(indices, advanced):
            self.roots[index] = root

    def close(self) -> None:
        return None


class AgentSide:
    """Per-game agent instances that cannot share a batched search tree."""

    def __init__(self, players: list[Any]) -> None:
        if not players:
            raise ValueError("agent side requires at least one player")
        self.players = players
        self._states: list[GameState] | None = None

    def bind_states(self, states: list[GameState]) -> None:
        self._states = states

    def choose_actions(self, indices: list[int]) -> list[int]:
        if self._states is None:
            raise RuntimeError("agent side is not bound to live match states")
        return [int(self.players[index].choose_action(self._states[index])) for index in indices]

    def observe_actions(
        self, indices: list[int], actions: list[int], states: list[GameState]
    ) -> None:
        for index, action, state in zip(indices, actions, states):
            self.players[index].observe_action(int(action), state)

    def close(self) -> None:
        return None


def make_factory_side(factory, states: list[GameState], seeds: list[int]) -> Any:
    if len(states) != len(seeds):
        raise ValueError("each match game needs a factory seed")
    evaluator = getattr(factory, "evaluator", None)
    simulations = getattr(factory, "simulations", None)
    if (
        evaluator is not None
        and simulations is not None
        and hasattr(evaluator, "evaluate_batch")
    ):
        return SearchSide(evaluator, int(simulations), seeds[0], states)
    return AgentSide([factory.create(seed) for seed in seeds])


def play_two_player_games(
    sides: tuple[Any, Any],
    states: list[GameState],
    seat_to_side: list[tuple[int, int]],
    max_plies: int,
    repetition_limit: int = 3,
) -> list[GameOutcome]:
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if not states:
        raise ValueError("at least one match game is required")
    if len(states) != len(seat_to_side):
        raise ValueError("each match game needs a seat assignment")
    live_states = list(states)
    games = [
        _LiveGame(state, assignment) for state, assignment in zip(live_states, seat_to_side)
    ]
    for side in sides:
        bind = getattr(side, "bind_states", None)
        if bind is not None:
            bind(live_states)
    try:
        while True:
            pending = [index for index, game in enumerate(games) if game.outcome is None]
            if not pending:
                break
            movable: list[int] = []
            for index in pending:
                game = games[index]
                key = game.state.canonical_key()
                game.repetitions[key] = game.repetitions.get(key, 0) + 1
                if game.repetitions[key] >= repetition_limit:
                    game.outcome = GameOutcome(None, game.plies, "repetition")
                elif game.plies >= max_plies:
                    game.outcome = GameOutcome(None, max_plies, "max_plies")
                else:
                    movable.append(index)
            grouped: tuple[list[int], list[int]] = ([], [])
            for index in movable:
                game = games[index]
                grouped[game.seat_to_side[game.state.turn]].append(index)
            progressed = False
            for side_index, indices in enumerate(grouped):
                if not indices:
                    continue
                progressed = True
                actions = [int(action) for action in sides[side_index].choose_actions(indices)]
                new_states = []
                for index, action in zip(indices, actions):
                    game = games[index]
                    state = game.state.apply_known_legal_action(action)
                    game.state = state
                    live_states[index] = state
                    game.plies += 1
                    new_states.append(state)
                    if state.is_terminal():
                        game.outcome = GameOutcome(state.winner, game.plies, "goal")
                for side in sides:
                    side.observe_actions(indices, actions, new_states)
            if not progressed:
                break
    finally:
        for side in sides:
            side.close()
    return [
        game.outcome
        if game.outcome is not None
        else GameOutcome(None, game.plies, "max_plies")
        for game in games
    ]
