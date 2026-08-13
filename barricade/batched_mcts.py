"""Wave-based MCTS that batches neural leaf evaluation across roots."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .mcts import EdgeStats, Node, SearchResult
from .state import GameState


@dataclass
class _PendingSimulation:
    leaf: Node
    path: list[EdgeStats]
    terminal_value: float | None = None


class BatchedMCTS:
    def __init__(
        self,
        evaluator,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.25,
        rng: random.Random | None = None,
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

    @staticmethod
    def _expand_from_evaluation(node: Node, policy: list[float]) -> None:
        legal = node.state.legal_actions()
        priors = [max(0.0, float(policy[action])) for action in legal]
        total = sum(priors)
        if total <= 0:
            priors = [1.0 / len(legal)] * len(legal)
        else:
            priors = [prior / total for prior in priors]
        node.edges = {action: EdgeStats(prior) for action, prior in zip(legal, priors)}
        node.expanded = True

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
        total_visits = sum(edge.visits for edge in node.edges.values())
        scale = math.sqrt(total_visits + 1)
        return max(
            node.edges,
            key=lambda action: (
                node.edges[action].q
                + self.c_puct
                * node.edges[action].prior
                * scale
                / (1 + node.edges[action].visits),
                -action,
            ),
        )

    def _descend(self, root: Node) -> _PendingSimulation:
        node = root
        path: list[EdgeStats] = []
        while node.expanded and not node.state.is_terminal():
            action = self._select(node)
            edge = node.edges[action]
            path.append(edge)
            if action not in node.children:
                node.children[action] = Node(node.state.apply_action(action))
            node = node.children[action]
        terminal_value = node.state.outcome_for_current_player() if node.state.is_terminal() else None
        return _PendingSimulation(node, path, terminal_value)

    @staticmethod
    def _backup(path: list[EdgeStats], leaf_value: float) -> None:
        value = leaf_value
        for edge in reversed(path):
            value = -value
            edge.visits += 1
            edge.value_sum += value

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
            if action not in root.state.legal_actions():
                raise ValueError(f"illegal root action: {action}")
            child = root.children.get(action)
            if child is None:
                child = Node(root.state.apply_action(action))
            advanced.append(child)
        return advanced

    def _ensure_expanded(self, roots: list[Node]) -> None:
        pending = [root for root in roots if not root.expanded]
        if not pending:
            return
        evaluations = self.evaluator.evaluate_batch([root.state for root in pending])
        for root, (policy, _value) in zip(pending, evaluations):
            self._expand_from_evaluation(root, policy)

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
                leaf_evaluations = self.evaluator.evaluate_batch([item.leaf.state for item in nonterminal])
                for item, (policy, value) in zip(nonterminal, leaf_evaluations):
                    self._expand_from_evaluation(item.leaf, policy)
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
