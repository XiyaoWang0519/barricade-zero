"""Wave-based MCTS that batches neural leaf evaluation across roots."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from collections.abc import Iterator
import numpy as np

from .mcts import Node, SearchResult
from .noise import segmented_dirichlet_noise
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
        self.tree_nodes_created = 0
        self.tree_expansions = 0
        self.tree_traversal_seconds = 0.0
        self.expansion_seconds = 0.0

    def _expand_from_evaluation(
        self, node: Node, policy: list[float], legal=None
    ) -> None:
        if legal is None:
            legal = self.rules_backend.legal_actions(node.state)
        priors = normalize_legal_priors(policy, legal)
        node.edges = CompactEdges(list(legal), priors, node.state.action_size)
        node.expanded = True

    def _expand_batch(
        self, nodes: list[Node], policies, legal_batch=None
    ) -> None:
        if not nodes:
            return
        started = time.perf_counter()
        if legal_batch is None:
            offsets, actions = self.rules_backend.legal_actions_batch(
                [node.state for node in nodes]
            )
        else:
            offsets, actions = legal_batch
        for index, (node, policy) in enumerate(zip(nodes, policies)):
            legal = [
                int(action)
                for action in actions[offsets[index]:offsets[index + 1]]
            ]
            self._expand_from_evaluation(node, policy, legal)
        self.tree_expansions += len(nodes)
        self.expansion_seconds += time.perf_counter() - started

    def _evaluate_nodes(self, nodes: list[Node]):
        states = [node.state for node in nodes]
        can_prepare = (
            hasattr(self.evaluator, "evaluate_prepared_batch_arrays")
            and getattr(self.evaluator, "encoding_backend", None) is self.rules_backend
        )
        if can_prepare:
            policies, values, offsets, actions = (
                self.evaluator.evaluate_prepared_batch_arrays(states)
            )
            return policies, values, (offsets, actions)
        if hasattr(self.evaluator, "evaluate_batch_arrays"):
            policies, values = self.evaluator.evaluate_batch_arrays(
                states, mask_legal=False
            )
            return policies, values, None
        evaluations = self.evaluator.evaluate_batch(states, mask_legal=False)
        policies = [policy for policy, _value in evaluations]
        values = [value for _policy, value in evaluations]
        return policies, values, None

    def _add_noise(self, root: Node, samples) -> None:
        edges: CompactEdges = root.edges
        for index, sample in enumerate(samples):
            edges.priors[index] = (
                (1 - self.noise_fraction) * edges.priors[index]
                + self.noise_fraction * float(sample)
            )

    def _select_index(self, node: Node) -> int:
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
        return best_index

    def _descend(self, root: Node) -> _PendingSimulation:
        started = time.perf_counter()
        node = root
        path: list[tuple[CompactEdges, int]] = []
        while node.expanded and not node.state.is_terminal():
            edges: CompactEdges = node.edges
            index = self._select_index(node)
            action = edges.actions[index]
            path.append((edges, index))
            if action not in node.children:
                node.children[action] = Node(node.state.apply_known_legal_action(action))
                self.tree_nodes_created += 1
            node = node.children[action]
        terminal_value = node.state.outcome_for_current_player() if node.state.is_terminal() else None
        self.tree_traversal_seconds += time.perf_counter() - started
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
        edges: CompactEdges = root.edges
        for index, action in enumerate(edges.actions):
            visits[action] = edges.visits[index]
            priors[action] = edges.priors[index]
        policy = [0.0] * size
        if temperature == 0:
            best_index = max(
                range(len(edges.actions)), key=edges.visits.__getitem__
            )
            policy[edges.actions[best_index]] = 1.0
        elif temperature > 0:
            weights = [visit ** (1.0 / temperature) for visit in edges.visits]
            total = sum(weights)
            source = weights if total else edges.priors
            total = sum(source)
            for action, weight in zip(edges.actions, source):
                policy[action] = weight / total
        else:
            raise ValueError("temperature must be non-negative")
        return SearchResult(policy, visits, priors)

    def create_roots(self, states: list[GameState]) -> list[Node]:
        if any(state.is_terminal() for state in states):
            raise ValueError("search roots must be non-terminal")
        roots = [Node(state) for state in states]
        self.tree_nodes_created += len(roots)
        return roots

    def advance_roots(self, roots: list[Node], actions: list[int]) -> list[Node]:
        if len(roots) != len(actions):
            raise ValueError("roots and actions must have equal length")
        for root, action in zip(roots, actions):
            if action not in self.rules_backend.legal_actions(root.state):
                raise ValueError(f"illegal root action: {action}")
        return self.advance_roots_known_legal(roots, actions)

    def advance_roots_known_legal(
        self, roots: list[Node], actions: list[int]
    ) -> list[Node]:
        if len(roots) != len(actions):
            raise ValueError("roots and actions must have equal length")
        advanced = []
        for root, action in zip(roots, actions):
            child = root.children.get(action)
            if child is None:
                child = Node(root.state.apply_known_legal_action(action))
                self.tree_nodes_created += 1
            advanced.append(child)
        return advanced

    def _ensure_expanded(self, roots: list[Node]) -> None:
        pending = [root for root in roots if not root.expanded]
        if not pending:
            return
        policies, _values, legal_batch = self._evaluate_nodes(pending)
        self._expand_batch(pending, policies, legal_batch)

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
            lengths = [len(root.edges) for root in roots]
            noise = segmented_dirichlet_noise(
                self.rng, lengths, self.dirichlet_alpha
            )
            offset = 0
            for root, length in zip(roots, lengths):
                self._add_noise(root, noise[offset:offset + length])
                offset += length

        for _ in range(self.simulations):
            pending = [self._descend(root) for root in roots]
            nonterminal = [item for item in pending if item.terminal_value is None]
            if nonterminal:
                policies, values, legal_batch = self._evaluate_nodes(
                    [item.leaf for item in nonterminal]
                )
                self._expand_batch(
                    [item.leaf for item in nonterminal], policies, legal_batch
                )
                for item, value in zip(nonterminal, values):
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
