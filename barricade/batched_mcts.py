"""Wave-based MCTS that batches neural leaf evaluation across roots."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections.abc import Iterator
import numpy as np

from .mcts import Node, SearchResult
from .backend import load_rules_backend
from .state import GameState


def normalize_legal_priors(policy, legal: list[int]) -> list[float]:
    if isinstance(policy, np.ndarray):
        indices = np.asarray(legal, dtype=np.intp)
        priors = np.maximum(policy[indices], 0.0).astype(np.float64, copy=False)
        total = float(priors.sum())
        if total > 0.0:
            return (priors / total).tolist()
        return [1.0 / len(legal)] * len(legal)
    priors = [max(0.0, float(policy[action])) for action in legal]
    total = sum(priors)
    if total <= 0.0:
        return [1.0 / len(legal)] * len(legal)
    return [prior / total for prior in priors]


@dataclass
class _PendingSimulation:
    leaf: Node
    path: list[tuple["CompactEdges", int]]
    terminal_value: float | None = None


class _CompactEdgeView:
    __slots__ = ("storage", "index")

    def __init__(self, storage: "CompactEdges", index: int) -> None:
        self.storage = storage
        self.index = index

    @property
    def prior(self) -> float:
        return self.storage.priors[self.index]

    @prior.setter
    def prior(self, value: float) -> None:
        self.storage.priors[self.index] = value

    @property
    def visits(self) -> int:
        return self.storage.visits[self.index]

    @visits.setter
    def visits(self, value: int) -> None:
        delta = value - self.storage.visits[self.index]
        self.storage.visits[self.index] = value
        self.storage.total_visits += delta

    @property
    def value_sum(self) -> float:
        return self.storage.value_sums[self.index]

    @value_sum.setter
    def value_sum(self, value: float) -> None:
        self.storage.value_sums[self.index] = value

    @property
    def q(self) -> float:
        visits = self.storage.visits[self.index]
        return self.storage.value_sums[self.index] / visits if visits else 0.0


class CompactEdges:
    """Parallel-list edge storage with a read-compatible mapping facade."""

    __slots__ = ("actions", "priors", "visits", "value_sums", "indices", "total_visits")

    def __init__(self, actions: list[int], priors: list[float], action_size: int) -> None:
        self.actions = actions
        self.priors = priors
        self.visits = [0] * len(actions)
        self.value_sums = [0.0] * len(actions)
        self.indices = [-1] * action_size
        for index, action in enumerate(actions):
            self.indices[action] = index
        self.total_visits = 0

    def __iter__(self) -> Iterator[int]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, action: int) -> _CompactEdgeView:
        index = self.indices[action]
        if index < 0:
            raise KeyError(action)
        return _CompactEdgeView(self, index)

    def values(self) -> Iterator[_CompactEdgeView]:
        return (_CompactEdgeView(self, index) for index in range(len(self.actions)))

    def items(self) -> Iterator[tuple[int, _CompactEdgeView]]:
        return ((action, _CompactEdgeView(self, index)) for index, action in enumerate(self.actions))


class BatchedMCTS:
    def __init__(
        self,
        evaluator,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.25,
        rng: random.Random | None = None,
        rules_backend=None,
    ) -> None:
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        if not hasattr(evaluator, "evaluate_batch"):
            raise TypeError("batched MCTS requires an evaluator with evaluate_batch")
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.rng = rng or random.Random()
        self.rules_backend = rules_backend or load_rules_backend()

    def _expand_from_evaluation(
        self, node: Node, policy: list[float], legal=None
    ) -> None:
        if legal is None:
            legal = self.rules_backend.legal_actions(node.state)
        priors = normalize_legal_priors(policy, legal)
        node.edges = CompactEdges(list(legal), priors, node.state.action_size)
        node.expanded = True

    def _expand_batch(self, nodes: list[Node], evaluations) -> None:
        if not nodes:
            return
        offsets, actions = self.rules_backend.legal_actions_batch(
            [node.state for node in nodes]
        )
        for index, (node, (policy, _value)) in enumerate(zip(nodes, evaluations)):
            legal = [
                int(action)
                for action in actions[offsets[index]:offsets[index + 1]]
            ]
            self._expand_from_evaluation(node, policy, legal)

    def _add_noise(self, root: Node) -> None:
        actions = list(root.edges)
        samples = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in actions]
        total = sum(samples)
        for action, sample in zip(actions, samples):
            edge = root.edges[action]
            edge.prior = (
                (1 - self.noise_fraction) * edge.prior
                + self.noise_fraction * sample / total
            )

    def _select(self, node: Node) -> int:
        edges: CompactEdges = node.edges
        scale = math.sqrt(edges.total_visits + 1)
        best_index = 0
        best_score = -math.inf
        for index, action in enumerate(edges.actions):
            visits = edges.visits[index]
            q = edges.value_sums[index] / visits if visits else 0.0
            score = q + self.c_puct * edges.priors[index] * scale / (1 + visits)
            if score > best_score:
                best_score = score
                best_index = index
        return edges.actions[best_index]

    def _descend(self, root: Node) -> _PendingSimulation:
        node = root
        path: list[tuple[CompactEdges, int]] = []
        while node.expanded and not node.state.is_terminal():
            action = self._select(node)
            edges: CompactEdges = node.edges
            path.append((edges, edges.indices[action]))
            if action not in node.children:
                node.children[action] = Node(node.state.apply_known_legal_action(action))
            node = node.children[action]
        terminal_value = node.state.outcome_for_current_player() if node.state.is_terminal() else None
        return _PendingSimulation(node, path, terminal_value)

    @staticmethod
    def _backup(path: list[tuple[CompactEdges, int]], leaf_value: float) -> None:
        value = leaf_value
        for edges, index in reversed(path):
            value = -value
            edges.visits[index] += 1
            edges.value_sums[index] += value
            edges.total_visits += 1

    def _result(self, root: Node, temperature: float) -> SearchResult:
        size = root.state.action_size
        visits = [0] * size
        priors = [0.0] * size
        for action, edge in root.edges.items():
            visits[action] = edge.visits
            priors[action] = edge.prior
        policy = [0.0] * size
        if temperature == 0:
            policy[max(root.edges, key=lambda action: visits[action])] = 1.0
        elif temperature > 0:
            weights = {action: visits[action] ** (1.0 / temperature) for action in root.edges}
            total = sum(weights.values())
            source = weights if total else {action: priors[action] for action in root.edges}
            total = sum(source.values())
            for action, weight in source.items():
                policy[action] = weight / total
        else:
            raise ValueError("temperature must be non-negative")
        return SearchResult(policy, visits, priors)

    def create_roots(self, states: list[GameState]) -> list[Node]:
        if any(state.is_terminal() for state in states):
            raise ValueError("search roots must be non-terminal")
        return [Node(state) for state in states]

    def advance_roots(self, roots: list[Node], actions: list[int]) -> list[Node]:
        if len(roots) != len(actions):
            raise ValueError("roots and actions must have equal length")
        advanced = []
        for root, action in zip(roots, actions):
            if action not in self.rules_backend.legal_actions(root.state):
                raise ValueError(f"illegal root action: {action}")
            child = root.children.get(action)
            if child is None:
                child = Node(root.state.apply_known_legal_action(action))
            advanced.append(child)
        return advanced

    def _ensure_expanded(self, roots: list[Node]) -> None:
        pending = [root for root in roots if not root.expanded]
        if not pending:
            return
        if hasattr(self.evaluator, "evaluate_batch_arrays"):
            policies, values = self.evaluator.evaluate_batch_arrays(
                [root.state for root in pending], mask_legal=False
            )
            evaluations = zip(policies, values)
        else:
            evaluations = self.evaluator.evaluate_batch(
                [root.state for root in pending], mask_legal=False
            )
        self._expand_batch(pending, evaluations)

    def search_roots(
        self,
        roots: list[Node],
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> list[SearchResult]:
        if not roots:
            return []
        if any(root.state.is_terminal() for root in roots):
            raise ValueError("search roots must be non-terminal")
        self._ensure_expanded(roots)
        if add_noise:
            for root in roots:
                self._add_noise(root)

        for _ in range(self.simulations):
            pending = [self._descend(root) for root in roots]
            nonterminal = [item for item in pending if item.terminal_value is None]
            if nonterminal:
                if hasattr(self.evaluator, "evaluate_batch_arrays"):
                    policies, values = self.evaluator.evaluate_batch_arrays(
                        [item.leaf.state for item in nonterminal], mask_legal=False
                    )
                    leaf_evaluations = list(zip(policies, values))
                else:
                    leaf_evaluations = self.evaluator.evaluate_batch(
                        [item.leaf.state for item in nonterminal], mask_legal=False
                    )
                self._expand_batch(
                    [item.leaf for item in nonterminal], leaf_evaluations
                )
                for item, (_policy, value) in zip(nonterminal, leaf_evaluations):
                    self._backup(item.path, value)
            for item in pending:
                if item.terminal_value is not None:
                    self._backup(item.path, item.terminal_value)
        return [self._result(root, temperature) for root in roots]

    def search_batch(
        self,
        states: list[GameState],
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> list[SearchResult]:
        return self.search_roots(self.create_roots(states), temperature, add_noise)
