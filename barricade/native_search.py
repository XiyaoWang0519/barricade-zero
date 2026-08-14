"""Native arena-backed MCTS session for the concurrent self-play hot path."""

from __future__ import annotations

import ctypes
import random
import time
from dataclasses import dataclass

import numpy as np

from .backend import NativeRulesBackend
from .mcts import SearchResult
from .noise import segmented_dirichlet_noise
from .state import GameState


_INT_POINTER = ctypes.POINTER(ctypes.c_int)
_INT64_POINTER = ctypes.POINTER(ctypes.c_longlong)
_UINT64_POINTER = ctypes.POINTER(ctypes.c_uint64)
_FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


@dataclass(frozen=True)
class NativeSearchMetrics:
    nodes: int
    expansions: int
    descents: int
    traversal_seconds: float
    preparation_seconds: float
    expansion_seconds: float
    encoding_seconds: float
    legality_seconds: float
    boundary_calls: int
    boundary_seconds: float


class NativeSearchSession:
    """Owns compact native nodes and executes complete batched MCTS waves."""

    def __init__(
        self,
        states: list[GameState],
        backend: NativeRulesBackend,
        c_puct: float = 1.5,
    ) -> None:
        if not states:
            raise ValueError("native search requires at least one root")
        if any(state.size != states[0].size for state in states):
            raise ValueError("native search roots must share board size")
        self.backend = backend
        self.library = backend.library
        self.size = states[0].size
        self.action_size = states[0].action_size
        self.boundary_calls = 0
        self.boundary_seconds = 0.0
        self._configure_abi()
        arrays = self._state_arrays(states)
        roots = np.empty(len(states), dtype=np.int32)
        started = time.perf_counter()
        self.handle = self.library.bz_search_create(
            self.size,
            len(states),
            *self._state_pointers(arrays),
            float(c_puct),
            roots.ctypes.data_as(_INT_POINTER),
        )
        self._record_boundary(started)
        if not self.handle:
            raise RuntimeError("native search allocation failed")
        self.root_ids = [int(root) for root in roots]

    def _configure_abi(self) -> None:
        library = self.library
        state_arguments = [
            ctypes.c_int,
            ctypes.c_int,
            _INT_POINTER,
            _INT_POINTER,
            _UINT64_POINTER,
            _UINT64_POINTER,
            _INT_POINTER,
            _INT_POINTER,
            _INT_POINTER,
            _INT_POINTER,
        ]
        library.bz_search_create.argtypes = state_arguments + [
            ctypes.c_double,
            _INT_POINTER,
        ]
        library.bz_search_create.restype = ctypes.c_void_p
        library.bz_search_destroy.argtypes = [ctypes.c_void_p]
        library.bz_search_destroy.restype = None
        prepare_arguments = [
            ctypes.c_void_p,
            _INT_POINTER,
            ctypes.c_int,
            _FLOAT_POINTER,
            ctypes.c_int,
            _INT_POINTER,
            _INT_POINTER,
            ctypes.c_int,
        ]
        library.bz_search_prepare_roots.argtypes = prepare_arguments
        library.bz_search_prepare_roots.restype = ctypes.c_int
        library.bz_search_descend_prepare.argtypes = prepare_arguments
        library.bz_search_descend_prepare.restype = ctypes.c_int
        library.bz_search_expand_backup.argtypes = [
            ctypes.c_void_p,
            _FLOAT_POINTER,
            _FLOAT_POINTER,
            ctypes.c_int,
            _INT_POINTER,
            _INT_POINTER,
        ]
        library.bz_search_expand_backup.restype = ctypes.c_int
        library.bz_search_root_actions.argtypes = [
            ctypes.c_void_p,
            _INT_POINTER,
            ctypes.c_int,
            _INT_POINTER,
            _INT_POINTER,
            ctypes.c_int,
        ]
        library.bz_search_root_actions.restype = ctypes.c_int
        library.bz_search_add_noise.argtypes = [
            ctypes.c_void_p,
            _INT_POINTER,
            ctypes.c_int,
            _INT_POINTER,
            _DOUBLE_POINTER,
            ctypes.c_double,
        ]
        library.bz_search_add_noise.restype = ctypes.c_int
        library.bz_search_results.argtypes = [
            ctypes.c_void_p,
            _INT_POINTER,
            ctypes.c_int,
            ctypes.c_double,
            _DOUBLE_POINTER,
            _INT_POINTER,
            _DOUBLE_POINTER,
            ctypes.c_int,
        ]
        library.bz_search_results.restype = ctypes.c_int
        library.bz_search_advance.argtypes = [
            ctypes.c_void_p,
            _INT_POINTER,
            _INT_POINTER,
            ctypes.c_int,
            _INT_POINTER,
            _INT_POINTER,
        ]
        library.bz_search_advance.restype = ctypes.c_int
        library.bz_search_get_states.argtypes = [ctypes.c_void_p, _INT_POINTER, ctypes.c_int] + [
            _INT_POINTER,
            _INT_POINTER,
            _UINT64_POINTER,
            _UINT64_POINTER,
            _INT_POINTER,
            _INT_POINTER,
            _INT_POINTER,
            _INT_POINTER,
        ]
        library.bz_search_get_states.restype = ctypes.c_int
        library.bz_search_stats.argtypes = [
            ctypes.c_void_p,
            _INT64_POINTER,
            _DOUBLE_POINTER,
        ]
        library.bz_search_stats.restype = ctypes.c_int

    @staticmethod
    def _state_arrays(states: list[GameState]) -> tuple[np.ndarray, ...]:
        arguments = [NativeRulesBackend._arguments(state) for state in states]
        return (
            np.asarray([item[1] for item in arguments], dtype=np.int32),
            np.asarray([item[2] for item in arguments], dtype=np.int32),
            np.asarray([item[3] for item in arguments], dtype=np.uint64),
            np.asarray([item[4] for item in arguments], dtype=np.uint64),
            np.asarray([state.walls_remaining[0] for state in states], dtype=np.int32),
            np.asarray([state.walls_remaining[1] for state in states], dtype=np.int32),
            np.asarray([state.turn for state in states], dtype=np.int32),
            np.asarray(
                [-1 if state.winner is None else state.winner for state in states],
                dtype=np.int32,
            ),
        )

    @staticmethod
    def _state_pointers(arrays: tuple[np.ndarray, ...]) -> tuple:
        return (
            arrays[0].ctypes.data_as(_INT_POINTER),
            arrays[1].ctypes.data_as(_INT_POINTER),
            arrays[2].ctypes.data_as(_UINT64_POINTER),
            arrays[3].ctypes.data_as(_UINT64_POINTER),
            arrays[4].ctypes.data_as(_INT_POINTER),
            arrays[5].ctypes.data_as(_INT_POINTER),
            arrays[6].ctypes.data_as(_INT_POINTER),
            arrays[7].ctypes.data_as(_INT_POINTER),
        )

    @staticmethod
    def _root_array(root_ids: list[int]) -> np.ndarray:
        return np.asarray(root_ids, dtype=np.int32)

    def _record_boundary(self, started: float) -> None:
        self.boundary_calls += 1
        self.boundary_seconds += time.perf_counter() - started

    def _prepare(self, function, root_ids: list[int]):
        roots = self._root_array(root_ids)
        count = len(root_ids)
        encoded = np.empty((count, 8, self.size, self.size), dtype=np.float32)
        offsets = np.empty(count + 1, dtype=np.int32)
        actions = np.empty(count * self.action_size, dtype=np.int32)
        started = time.perf_counter()
        evaluation_count = function(
            self.handle,
            roots.ctypes.data_as(_INT_POINTER),
            count,
            encoded.ctypes.data_as(_FLOAT_POINTER),
            encoded.size,
            offsets.ctypes.data_as(_INT_POINTER),
            actions.ctypes.data_as(_INT_POINTER),
            len(actions),
        )
        self._record_boundary(started)
        if evaluation_count < 0:
            raise RuntimeError(f"native search preparation failed: {evaluation_count}")
        action_count = int(offsets[evaluation_count])
        return (
            encoded[:evaluation_count],
            offsets[:evaluation_count + 1],
            actions[:action_count],
        )

    def _evaluate_and_expand(self, evaluator, prepared) -> None:
        encoded, offsets, actions = prepared
        policies, values = evaluator.evaluate_encoded_batch_arrays(encoded)
        policy_array = np.ascontiguousarray(policies, dtype=np.float32)
        value_array = np.ascontiguousarray(values, dtype=np.float32)
        started = time.perf_counter()
        result = self.library.bz_search_expand_backup(
            self.handle,
            policy_array.ctypes.data_as(_FLOAT_POINTER),
            value_array.ctypes.data_as(_FLOAT_POINTER),
            len(values),
            offsets.ctypes.data_as(_INT_POINTER),
            actions.ctypes.data_as(_INT_POINTER),
        )
        self._record_boundary(started)
        if result != 0:
            raise RuntimeError(f"native search expansion failed: {result}")

    def _ensure_expanded(self, root_ids: list[int], evaluator) -> None:
        prepared = self._prepare(self.library.bz_search_prepare_roots, root_ids)
        self._evaluate_and_expand(evaluator, prepared)

    def _add_noise(
        self,
        root_ids: list[int],
        rng: random.Random,
        alpha: float,
        fraction: float,
    ) -> None:
        roots = self._root_array(root_ids)
        offsets = np.empty(len(root_ids) + 1, dtype=np.int32)
        actions = np.empty(len(root_ids) * self.action_size, dtype=np.int32)
        started = time.perf_counter()
        count = self.library.bz_search_root_actions(
            self.handle,
            roots.ctypes.data_as(_INT_POINTER),
            len(root_ids),
            offsets.ctypes.data_as(_INT_POINTER),
            actions.ctypes.data_as(_INT_POINTER),
            len(actions),
        )
        self._record_boundary(started)
        if count < 0:
            raise RuntimeError(f"native root-action buffer requires {-count} entries")
        noise = segmented_dirichlet_noise(rng, np.diff(offsets), alpha)
        started = time.perf_counter()
        result = self.library.bz_search_add_noise(
            self.handle,
            roots.ctypes.data_as(_INT_POINTER),
            len(root_ids),
            offsets.ctypes.data_as(_INT_POINTER),
            noise.ctypes.data_as(_DOUBLE_POINTER),
            fraction,
        )
        self._record_boundary(started)
        if result != 0:
            raise RuntimeError(f"native root-noise update failed: {result}")

    def search_roots(
        self,
        root_ids: list[int],
        evaluator,
        simulations: int,
        temperature: float,
        add_noise: bool,
        rng: random.Random,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.25,
    ) -> list[SearchResult]:
        self._ensure_expanded(root_ids, evaluator)
        if add_noise:
            self._add_noise(
                root_ids, rng, dirichlet_alpha, noise_fraction
            )
        for _ in range(simulations):
            prepared = self._prepare(
                self.library.bz_search_descend_prepare, root_ids
            )
            self._evaluate_and_expand(evaluator, prepared)
        roots = self._root_array(root_ids)
        shape = (len(root_ids), self.action_size)
        policies = np.empty(shape, dtype=np.float64)
        visits = np.empty(shape, dtype=np.int32)
        priors = np.empty(shape, dtype=np.float64)
        started = time.perf_counter()
        result = self.library.bz_search_results(
            self.handle,
            roots.ctypes.data_as(_INT_POINTER),
            len(root_ids),
            temperature,
            policies.ctypes.data_as(_DOUBLE_POINTER),
            visits.ctypes.data_as(_INT_POINTER),
            priors.ctypes.data_as(_DOUBLE_POINTER),
            policies.size,
        )
        self._record_boundary(started)
        if result != policies.size:
            raise RuntimeError(f"native search results failed: {result}")
        return [
            SearchResult(policy.tolist(), visit.tolist(), prior.tolist())
            for policy, visit, prior in zip(policies, visits, priors)
        ]

    def advance(
        self, root_ids: list[int], actions: list[int]
    ) -> tuple[list[int], list[int]]:
        if len(root_ids) != len(actions):
            raise ValueError("roots and actions must have equal length")
        roots = self._root_array(root_ids)
        action_array = np.asarray(actions, dtype=np.int32)
        advanced = np.empty(len(roots), dtype=np.int32)
        reused = np.empty(len(roots), dtype=np.int32)
        started = time.perf_counter()
        result = self.library.bz_search_advance(
            self.handle,
            roots.ctypes.data_as(_INT_POINTER),
            action_array.ctypes.data_as(_INT_POINTER),
            len(roots),
            advanced.ctypes.data_as(_INT_POINTER),
            reused.ctypes.data_as(_INT_POINTER),
        )
        self._record_boundary(started)
        if result != 0:
            raise RuntimeError(f"native search advance failed: {result}")
        return advanced.tolist(), reused.tolist()

    @staticmethod
    def _walls(mask: int, size: int) -> frozenset[tuple[int, int]]:
        width = size - 1
        return frozenset(
            (index // width, index % width)
            for index in range(width * width)
            if mask & (1 << index)
        )

    def states(self, node_ids: list[int]) -> list[GameState]:
        ids = self._root_array(node_ids)
        integer_arrays = [np.empty(len(ids), dtype=np.int32) for _ in range(6)]
        horizontal = np.empty(len(ids), dtype=np.uint64)
        vertical = np.empty(len(ids), dtype=np.uint64)
        started = time.perf_counter()
        result = self.library.bz_search_get_states(
            self.handle,
            ids.ctypes.data_as(_INT_POINTER),
            len(ids),
            integer_arrays[0].ctypes.data_as(_INT_POINTER),
            integer_arrays[1].ctypes.data_as(_INT_POINTER),
            horizontal.ctypes.data_as(_UINT64_POINTER),
            vertical.ctypes.data_as(_UINT64_POINTER),
            integer_arrays[2].ctypes.data_as(_INT_POINTER),
            integer_arrays[3].ctypes.data_as(_INT_POINTER),
            integer_arrays[4].ctypes.data_as(_INT_POINTER),
            integer_arrays[5].ctypes.data_as(_INT_POINTER),
        )
        self._record_boundary(started)
        if result != 0:
            raise RuntimeError(f"native state export failed: {result}")
        states = []
        for index in range(len(ids)):
            winner = int(integer_arrays[5][index])
            states.append(GameState(
                size=self.size,
                pawns=(
                    divmod(int(integer_arrays[0][index]), self.size),
                    divmod(int(integer_arrays[1][index]), self.size),
                ),
                horizontal_walls=self._walls(int(horizontal[index]), self.size),
                vertical_walls=self._walls(int(vertical[index]), self.size),
                walls_remaining=(
                    int(integer_arrays[2][index]), int(integer_arrays[3][index])
                ),
                turn=int(integer_arrays[4][index]),
                winner=None if winner < 0 else winner,
            ))
        return states

    @property
    def metrics(self) -> NativeSearchMetrics:
        integers = np.empty(3, dtype=np.int64)
        seconds = np.empty(5, dtype=np.float64)
        started = time.perf_counter()
        result = self.library.bz_search_stats(
            self.handle,
            integers.ctypes.data_as(_INT64_POINTER),
            seconds.ctypes.data_as(_DOUBLE_POINTER),
        )
        self._record_boundary(started)
        if result != 0:
            raise RuntimeError(f"native search metrics failed: {result}")
        return NativeSearchMetrics(
            nodes=int(integers[0]),
            expansions=int(integers[1]),
            descents=int(integers[2]),
            traversal_seconds=float(seconds[0]),
            preparation_seconds=float(seconds[1]),
            expansion_seconds=float(seconds[2]),
            encoding_seconds=float(seconds[3]),
            legality_seconds=float(seconds[4]),
            boundary_calls=self.boundary_calls,
            boundary_seconds=self.boundary_seconds,
        )

    def close(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.library.bz_search_destroy(handle)
            self.handle = None

    def __del__(self) -> None:
        self.close()
