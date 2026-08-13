"""Player adapters used by the standalone evaluation arena."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from barricade.batched_mcts import BatchedMCTS
from barricade.mcts import MCTS, UniformEvaluator
from barricade.state import GameState


class AgentPlayer:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def choose_action(self, state: GameState) -> int:
        return int(self.agent.choose_action(state))

    def observe_action(self, action: int, state: GameState) -> None:
        return None


@dataclass(frozen=True)
class AgentPlayerFactory:
    name: str
    builder: Callable[[int], Any]

    def create(self, seed: int) -> AgentPlayer:
        return AgentPlayer(self.builder(seed))


class SearchPlayer:
    def __init__(self, evaluator: Any, simulations: int, seed: int) -> None:
        self.batched = hasattr(evaluator, "evaluate_batch")
        self.root = None
        if self.batched:
            self.search = BatchedMCTS(
                evaluator,
                simulations=simulations,
                rng=random.Random(seed),
            )
        else:
            self.search = MCTS(
                evaluator,
                simulations=simulations,
                rng=random.Random(seed),
            )

    def choose_action(self, state: GameState) -> int:
        if not self.batched:
            return int(self.search.search(state, temperature=0, add_noise=False).best_action)
        if self.root is None or self.root.state != state:
            self.root = self.search.create_roots([state])[0]
        result = self.search.search_roots(
            [self.root], temperature=0, add_noise=False
        )[0]
        return int(result.best_action)

    def observe_action(self, action: int, state: GameState) -> None:
        if not self.batched or self.root is None:
            return
        try:
            advanced = self.search.advance_roots([self.root], [int(action)])[0]
        except ValueError:
            self.root = None
            return
        self.root = advanced if advanced.state == state else None


@dataclass(frozen=True)
class SearchPlayerFactory:
    name: str
    evaluator: Any
    simulations: int

    def create(self, seed: int) -> SearchPlayer:
        return SearchPlayer(self.evaluator, self.simulations, seed)


def uniform_mcts_factory(simulations: int) -> SearchPlayerFactory:
    return SearchPlayerFactory("uniform_mcts", UniformEvaluator(), simulations)
