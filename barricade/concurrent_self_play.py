"""Concurrent self-play synchronized by move, with batched MCTS inference."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .batched_mcts import BatchedMCTS
from .backend import NativeRulesBackend, load_rules_backend
from .encoding import rotate_policy
from .self_play import TrainingExample
from .state import GameState


@dataclass
class _Game:
    state: GameState
    trajectory: list[tuple[GameState, list[float], int]] = field(default_factory=list)
    repetitions: dict[bytes, int] = field(default_factory=dict)
    plies: int = 0
    root: Any = None


@dataclass(frozen=True)
class ConcurrentSelfPlayResult:
    examples: list[TrainingExample]
    game_lengths: list[int]
    wins: tuple[int, int]
    draws: int
    forward_calls: int
    positions_evaluated: int
    reused_root_visits: int
    tree_nodes_created: int
    tree_expansions: int
    tree_traversal_seconds: float
    expansion_seconds: float
    search_backend: str
    native_search_boundary_calls: int
    native_search_boundary_seconds: float
    native_preparation_seconds: float
    native_encoding_seconds: float
    native_legality_seconds: float

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
    use_tree_reuse: bool = True,
    rules_backend=None,
    use_native_search: bool | None = None,
) -> ConcurrentSelfPlayResult:
    if games <= 0:
        raise ValueError("games must be positive")
    rng = rng or random.Random()
    active = [_Game(GameState.initial(board_size, walls_per_player)) for _ in range(games)]
    completed_examples: list[TrainingExample] = []
    lengths: list[int] = []
    wins = [0, 0]
    draws = 0
    reused_root_visits = 0
    starting_calls = evaluator.forward_calls
    starting_positions = evaluator.positions_evaluated
    effective_backend = rules_backend or getattr(evaluator, "encoding_backend", None)
    if effective_backend is None:
        effective_backend = load_rules_backend()
    search_rng = random.Random(rng.getrandbits(64))
    search = BatchedMCTS(
        evaluator,
        simulations=simulations,
        rng=search_rng,
        rules_backend=effective_backend,
    )
    native_search = None
    native_capable = (
        use_tree_reuse
        and isinstance(effective_backend, NativeRulesBackend)
        and getattr(evaluator, "encoding_backend", None) is effective_backend
        and hasattr(evaluator, "evaluate_encoded_batch_arrays")
    )
    if use_native_search is True and not native_capable:
        raise RuntimeError("native search was requested but its prerequisites are unavailable")
    if use_native_search is not False and native_capable:
        from .native_search import NativeSearchSession

        native_search = NativeSearchSession(
            [game.state for game in active], effective_backend,
        )
        for game, root_id in zip(active, native_search.root_ids):
            game.root = root_id

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
        roots = []
        for game in searchable:
            if native_search is None and (not use_tree_reuse or game.root is None):
                game.root = search.create_roots([game.state])[0]
            roots.append(game.root)
        if native_search is not None:
            results = native_search.search_roots(
                roots,
                evaluator,
                simulations=simulations,
                temperature=temperature,
                add_noise=temperature > 0,
                rng=search_rng,
            )
        else:
            results = search.search_roots(
                roots,
                temperature=temperature,
                add_noise=temperature > 0,
            )
        actions = []
        for game, result in zip(searchable, results):
            canonical_policy = (
                result.policy if game.state.turn == 0 else rotate_policy(result.policy, board_size)
            )
            game.trajectory.append((game.state.canonical(), canonical_policy, game.state.turn))
            actions.append(
                _sample(result.policy, rng) if temperature > 0 else result.best_action
            )
        if native_search is not None:
            advanced_roots, reused_visits = native_search.advance(roots, actions)
        else:
            advanced_nodes = search.advance_roots_known_legal(roots, actions)
            advanced_roots = advanced_nodes
            reused_visits = [
                getattr(advanced.edges, "total_visits", 0)
                for advanced in advanced_nodes
            ]
        survivors = []
        for game, action, advanced, retained_visits in zip(
            searchable, actions, advanced_roots, reused_visits
        ):
            if use_tree_reuse:
                reused_root_visits += retained_visits
                game.root = advanced
            else:
                game.root = None
            game.state = (
                game.state.apply_known_legal_action(action)
                if native_search is not None
                else advanced.state
            )
            game.plies += 1
            if game.state.is_terminal():
                completed_examples.extend(_finish(game, game.state.winner))
                lengths.append(game.plies)
                wins[game.state.winner] += 1
            else:
                survivors.append(game)
        active = survivors

    if native_search is not None:
        native_metrics = native_search.metrics
        tree_nodes_created = native_metrics.nodes
        tree_expansions = native_metrics.expansions
        tree_traversal_seconds = native_metrics.traversal_seconds
        expansion_seconds = native_metrics.expansion_seconds
        search_backend = "native"
        native_boundary_calls = native_metrics.boundary_calls
        native_boundary_seconds = native_metrics.boundary_seconds
        native_preparation_seconds = native_metrics.preparation_seconds
        native_encoding_seconds = native_metrics.encoding_seconds
        native_legality_seconds = native_metrics.legality_seconds
        native_search.close()
    else:
        tree_nodes_created = search.tree_nodes_created
        tree_expansions = search.tree_expansions
        tree_traversal_seconds = search.tree_traversal_seconds
        expansion_seconds = search.expansion_seconds
        search_backend = "python"
        native_boundary_calls = 0
        native_boundary_seconds = 0.0
        native_preparation_seconds = 0.0
        native_encoding_seconds = 0.0
        native_legality_seconds = 0.0
    return ConcurrentSelfPlayResult(
        completed_examples,
        lengths,
        (wins[0], wins[1]),
        draws,
        evaluator.forward_calls - starting_calls,
        evaluator.positions_evaluated - starting_positions,
        reused_root_visits,
        tree_nodes_created,
        tree_expansions,
        tree_traversal_seconds,
        expansion_seconds,
        search_backend,
        native_boundary_calls,
        native_boundary_seconds,
        native_preparation_seconds,
        native_encoding_seconds,
        native_legality_seconds,
    )
