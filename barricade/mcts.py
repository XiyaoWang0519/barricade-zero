"""Reference PUCT Monte Carlo Tree Search."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .state import GameState


class Evaluator(Protocol):
    def evaluate(self, state: GameState) -> tuple[Sequence[float], float]: ...


class UniformEvaluator:
    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        legal = state.legal_actions()
        policy = [0.0] * state.action_size
        for action in legal:
            policy[action] = 1.0 / len(legal)
        return policy, 0.0

    def evaluate_batch(
        self, states: Sequence[GameState], mask_legal: bool = True
    ) -> list[tuple[list[float], float]]:
        return [self.evaluate(state) for state in states]


@dataclass
class EdgeStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    state: GameState
    edges: dict[int, EdgeStats] = field(default_factory=dict)
    children: dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False


@dataclass(frozen=True)
class SearchResult:
    policy: list[float]
    visits: list[int]
    root_priors: list[float]

    @property
    def best_action(self) -> int:
        return max(range(len(self.visits)), key=lambda action: self.visits[action])


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.25,
        rng: random.Random | None = None,
    ) -> None:
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.rng = rng or random.Random()

    def _expand(self, node: Node) -> float:
        if node.state.is_terminal():
            node.expanded = True
            return node.state.outcome_for_current_player()
        policy, value = self.evaluator.evaluate(node.state)
        legal = node.state.legal_actions()
        priors = [max(0.0, float(policy[action])) for action in legal]
        total = sum(priors)
        if total <= 0:
            priors = [1.0 / len(legal)] * len(legal)
        else:
            priors = [prior / total for prior in priors]
        node.edges = {action: EdgeStats(prior) for action, prior in zip(legal, priors)}
        node.expanded = True
        return max(-1.0, min(1.0, float(value)))

    def _add_root_noise(self, root: Node) -> None:
        actions = list(root.edges)
        noise = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in actions]
        total = sum(noise)
        noise = [value / total for value in noise]
        for action, sample in zip(actions, noise):
            edge = root.edges[action]
            edge.prior = (1 - self.noise_fraction) * edge.prior + self.noise_fraction * sample

    def _select(self, node: Node) -> int:
        total_visits = sum(edge.visits for edge in node.edges.values())
        exploration_scale = math.sqrt(total_visits + 1)
        return max(
            node.edges,
            key=lambda action: (
                node.edges[action].q
                + self.c_puct * node.edges[action].prior * exploration_scale / (1 + node.edges[action].visits),
                -action,
            ),
        )

    def _simulate(self, root: Node) -> None:
        node = root
        path: list[EdgeStats] = []
        while node.expanded and not node.state.is_terminal():
            action = self._select(node)
            edge = node.edges[action]
            path.append(edge)
            child = node.children.get(action)
            if child is None:
                child = Node(node.state.apply_action(action))
                node.children[action] = child
            node = child
        value = self._expand(node) if not node.expanded else node.state.outcome_for_current_player()
        for edge in reversed(path):
            value = -value
            edge.visits += 1
            edge.value_sum += value

    def search(self, state: GameState, temperature: float = 1.0, add_noise: bool = False) -> SearchResult:
        root = Node(state)
        self._expand(root)
        if add_noise:
            self._add_root_noise(root)
        for _ in range(self.simulations):
            self._simulate(root)
        visits = [0] * state.action_size
        priors = [0.0] * state.action_size
        for action, edge in root.edges.items():
            visits[action] = edge.visits
            priors[action] = edge.prior
        policy = [0.0] * state.action_size
        if temperature == 0:
            policy[max(root.edges, key=lambda action: visits[action])] = 1.0
        else:
            if temperature < 0:
                raise ValueError("temperature must be non-negative")
            weights = {action: visits[action] ** (1.0 / temperature) for action in root.edges}
            total = sum(weights.values())
            if total == 0:
                total = sum(priors)
                for action in root.edges:
                    policy[action] = priors[action] / total
            else:
                for action, weight in weights.items():
                    policy[action] = weight / total
        return SearchResult(policy, visits, priors)
