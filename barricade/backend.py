"""Optional native rules backend with a tested Python fallback."""

from __future__ import annotations

import ctypes
from pathlib import Path
import time
from typing import Protocol

import numpy as np

from .state import GameState


class RulesBackend(Protocol):
    name: str

    def legal_actions(self, state: GameState) -> list[int]: ...

    def has_path_with_extra_wall(
        self, state: GameState, player: int, orientation: str | None = None,
        row: int = 0, col: int = 0,
    ) -> bool: ...


class PythonRulesBackend:
    name = "python"

    def __init__(self, collect_metrics: bool = False) -> None:
        self.collect_metrics = collect_metrics
        self.boundary_calls = 0
        self.boundary_seconds = 0.0
        self.legal_action_calls = 0
        self.legal_action_seconds = 0.0
        self.path_validation_calls = 0
        self.path_validation_seconds = 0.0
        self.encoding_calls = 0
        self.encoding_seconds = 0.0
        self.batch_preparation_calls = 0
        self.batch_preparation_seconds = 0.0

    def _record(self, operation: str, started: float) -> None:
        if not self.collect_metrics:
            return
        elapsed = time.perf_counter() - started
        if operation == "legal":
            self.legal_action_calls += 1
            self.legal_action_seconds += elapsed
        elif operation == "path":
            self.path_validation_calls += 1
            self.path_validation_seconds += elapsed
        elif operation == "encoding":
            self.encoding_calls += 1
            self.encoding_seconds += elapsed

    def legal_actions(self, state: GameState) -> list[int]:
        started = time.perf_counter()
        result = state.legal_actions()
        self._record("legal", started)
        return result

    def legal_actions_batch(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        started = time.perf_counter()
        batches = [state.legal_actions() for state in states]
        offsets = np.zeros(len(states) + 1, dtype=np.int32)
        if batches:
            offsets[1:] = np.cumsum([len(actions) for actions in batches], dtype=np.int32)
        actions = np.asarray(
            [action for batch in batches for action in batch], dtype=np.int32
        )
        self._record("legal", started)
        return offsets, actions

    def has_path_with_extra_wall(
        self, state: GameState, player: int, orientation: str | None = None,
        row: int = 0, col: int = 0,
    ) -> bool:
        started = time.perf_counter()
        result = state._has_path_with_extra_wall(player, orientation, row, col)
        self._record("path", started)
        return result

    def encode_state(self, state: GameState) -> list[list[list[float]]]:
        from .encoding import encode_state

        started = time.perf_counter()
        result = encode_state(state)
        self._record("encoding", started)
        return result

    def encode_batch(self, states: list[GameState]) -> np.ndarray:
        from .encoding import encode_state

        started = time.perf_counter()
        result = np.ascontiguousarray(
            [encode_state(state) for state in states], dtype=np.float32
        )
        self._record("encoding", started)
        return result

    def encode_canonical_batch(self, states: list[GameState]) -> np.ndarray:
        return self.encode_batch([state.canonical() for state in states])

    def prepare_batch(
        self, states: list[GameState]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoded = self.encode_canonical_batch(states)
        offsets, actions = self.legal_actions_batch(states)
        return encoded, offsets, actions


def _wall_mask(walls: frozenset[tuple[int, int]], size: int) -> int:
    width = size - 1
    return sum(1 << (row * width + col) for row, col in walls)


class NativeRulesBackend:
    name = "native"

    def __init__(
        self, library: str | Path | None = None, collect_metrics: bool = False
    ) -> None:
        path = Path(library) if library else Path(__file__).parents[1] / "native" / "libbarricade_rules.so"
        self.library = ctypes.CDLL(str(path))
        self.collect_metrics = collect_metrics
        self.boundary_calls = 0
        self.boundary_seconds = 0.0
        self.legal_action_calls = 0
        self.legal_action_seconds = 0.0
        self.path_validation_calls = 0
        self.path_validation_seconds = 0.0
        self.encoding_calls = 0
        self.encoding_seconds = 0.0
        self.batch_preparation_calls = 0
        self.batch_preparation_seconds = 0.0
        self.library.bz_has_path.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.library.bz_has_path.restype = ctypes.c_int
        self.library.bz_legal_actions.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ]
        self.library.bz_legal_actions.restype = ctypes.c_int
        int_pointer = ctypes.POINTER(ctypes.c_int)
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        self.library.bz_legal_actions_batch.argtypes = [
            ctypes.c_int, ctypes.c_int,
            int_pointer, int_pointer, uint64_pointer, uint64_pointer,
            int_pointer, int_pointer, int_pointer, int_pointer,
            int_pointer, int_pointer, ctypes.c_int,
        ]
        self.library.bz_legal_actions_batch.restype = ctypes.c_int
        self.library.bz_encode_state.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ]
        self.library.bz_encode_state.restype = ctypes.c_int
        self.library.bz_encode_state_f32.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        self.library.bz_encode_state_f32.restype = ctypes.c_int
        self.library.bz_encode_state_batch_f32.argtypes = [
            ctypes.c_int, ctypes.c_int,
            int_pointer, int_pointer, uint64_pointer, uint64_pointer,
            int_pointer, int_pointer, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        self.library.bz_encode_state_batch_f32.restype = ctypes.c_int
        self.library.bz_encode_canonical_batch_f32.argtypes = [
            ctypes.c_int, ctypes.c_int,
            int_pointer, int_pointer, uint64_pointer, uint64_pointer,
            int_pointer, int_pointer, int_pointer,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        self.library.bz_encode_canonical_batch_f32.restype = ctypes.c_int
        self.library.bz_prepare_batch_f32.argtypes = [
            ctypes.c_int, ctypes.c_int,
            int_pointer, int_pointer, uint64_pointer, uint64_pointer,
            int_pointer, int_pointer, int_pointer, int_pointer,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            int_pointer, int_pointer, ctypes.c_int,
        ]
        self.library.bz_prepare_batch_f32.restype = ctypes.c_int

    def _record(self, operation: str, started: float) -> None:
        if not self.collect_metrics:
            return
        elapsed = time.perf_counter() - started
        self.boundary_calls += 1
        self.boundary_seconds += elapsed
        if operation == "legal":
            self.legal_action_calls += 1
            self.legal_action_seconds += elapsed
        elif operation == "path":
            self.path_validation_calls += 1
            self.path_validation_seconds += elapsed
        elif operation == "encoding":
            self.encoding_calls += 1
            self.encoding_seconds += elapsed

    @staticmethod
    def _arguments(state: GameState) -> tuple[int, int, int, int, int]:
        if state.size > 9:
            raise ValueError("native backend currently supports board sizes up to 9")
        return (
            state.size,
            state.pawns[0][0] * state.size + state.pawns[0][1],
            state.pawns[1][0] * state.size + state.pawns[1][1],
            _wall_mask(state.horizontal_walls, state.size),
            _wall_mask(state.vertical_walls, state.size),
        )

    def legal_actions(self, state: GameState) -> list[int]:
        started = time.perf_counter()
        capacity = state.action_size
        output = (ctypes.c_int * capacity)()
        winner = -1 if state.winner is None else state.winner
        count = self.library.bz_legal_actions(
            *self._arguments(state), *state.walls_remaining, state.turn, winner,
            output, capacity,
        )
        if count < 0:
            raise RuntimeError(f"native action buffer requires {-count} entries")
        result = list(output[:count])
        self._record("legal", started)
        return result

    def legal_actions_batch(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            return np.zeros(1, dtype=np.int32), np.empty(0, dtype=np.int32)
        started = time.perf_counter()
        size = states[0].size
        if any(state.size != size for state in states):
            raise ValueError("all states in a legal-action batch must share board size")
        arguments = [self._arguments(state) for state in states]
        p0 = np.asarray([item[1] for item in arguments], dtype=np.int32)
        p1 = np.asarray([item[2] for item in arguments], dtype=np.int32)
        horizontal = np.asarray([item[3] for item in arguments], dtype=np.uint64)
        vertical = np.asarray([item[4] for item in arguments], dtype=np.uint64)
        w0 = np.asarray([state.walls_remaining[0] for state in states], dtype=np.int32)
        w1 = np.asarray([state.walls_remaining[1] for state in states], dtype=np.int32)
        turns = np.asarray([state.turn for state in states], dtype=np.int32)
        winners = np.asarray(
            [-1 if state.winner is None else state.winner for state in states], dtype=np.int32
        )
        offsets = np.empty(len(states) + 1, dtype=np.int32)
        capacity = len(states) * states[0].action_size
        actions = np.empty(capacity, dtype=np.int32)

        int_pointer = ctypes.POINTER(ctypes.c_int)
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        count = self.library.bz_legal_actions_batch(
            size, len(states),
            p0.ctypes.data_as(int_pointer), p1.ctypes.data_as(int_pointer),
            horizontal.ctypes.data_as(uint64_pointer), vertical.ctypes.data_as(uint64_pointer),
            w0.ctypes.data_as(int_pointer), w1.ctypes.data_as(int_pointer),
            turns.ctypes.data_as(int_pointer), winners.ctypes.data_as(int_pointer),
            offsets.ctypes.data_as(int_pointer), actions.ctypes.data_as(int_pointer), capacity,
        )
        if count < 0:
            raise RuntimeError(f"native batch action buffer requires {-count} entries")
        result = (offsets, actions[:count])
        self._record("legal", started)
        return result

    def has_path_with_extra_wall(
        self, state: GameState, player: int, orientation: str | None = None,
        row: int = 0, col: int = 0,
    ) -> bool:
        started = time.perf_counter()
        extra = 0 if orientation is None else (1 if orientation == "H" else 2)
        result = bool(self.library.bz_has_path(
            *self._arguments(state), player, extra, row, col
        ))
        self._record("path", started)
        return result

    def encode_state(self, state: GameState) -> list[list[list[float]]]:
        started = time.perf_counter()
        capacity = 8 * state.size * state.size
        output = (ctypes.c_double * capacity)()
        count = self.library.bz_encode_state(
            *self._arguments(state), *state.walls_remaining, output, capacity
        )
        if count != capacity:
            raise RuntimeError(f"native encoder returned {count}, expected {capacity}")
        size = state.size
        cells = size * size
        flat = list(output)
        result = [
            [flat[plane * cells + row * size:plane * cells + (row + 1) * size]
             for row in range(size)]
            for plane in range(8)
        ]
        self._record("encoding", started)
        return result

    def encode_batch(self, states: list[GameState]) -> np.ndarray:
        if not states:
            return np.empty((0, 8, 0, 0), dtype=np.float32)
        started = time.perf_counter()
        size = states[0].size
        if any(state.size != size for state in states):
            raise ValueError("all states in an encoding batch must share board size")
        output = np.empty((len(states), 8, size, size), dtype=np.float32)
        per_state = 8 * size * size
        arguments = [self._arguments(state) for state in states]
        p0 = np.asarray([item[1] for item in arguments], dtype=np.int32)
        p1 = np.asarray([item[2] for item in arguments], dtype=np.int32)
        horizontal = np.asarray([item[3] for item in arguments], dtype=np.uint64)
        vertical = np.asarray([item[4] for item in arguments], dtype=np.uint64)
        w0 = np.asarray([state.walls_remaining[0] for state in states], dtype=np.int32)
        w1 = np.asarray([state.walls_remaining[1] for state in states], dtype=np.int32)
        int_pointer = ctypes.POINTER(ctypes.c_int)
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        expected = len(states) * per_state
        count = self.library.bz_encode_state_batch_f32(
            size, len(states),
            p0.ctypes.data_as(int_pointer), p1.ctypes.data_as(int_pointer),
            horizontal.ctypes.data_as(uint64_pointer), vertical.ctypes.data_as(uint64_pointer),
            w0.ctypes.data_as(int_pointer), w1.ctypes.data_as(int_pointer),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), expected,
        )
        if count != expected:
            raise RuntimeError(f"native batch encoder returned {count}, expected {expected}")
        self._record("encoding", started)
        return output

    def encode_canonical_batch(self, states: list[GameState]) -> np.ndarray:
        if not states:
            return np.empty((0, 8, 0, 0), dtype=np.float32)
        started = time.perf_counter()
        size = states[0].size
        if any(state.size != size for state in states):
            raise ValueError("all states in an encoding batch must share board size")
        output = np.empty((len(states), 8, size, size), dtype=np.float32)
        arguments = [self._arguments(state) for state in states]
        p0 = np.asarray([item[1] for item in arguments], dtype=np.int32)
        p1 = np.asarray([item[2] for item in arguments], dtype=np.int32)
        horizontal = np.asarray([item[3] for item in arguments], dtype=np.uint64)
        vertical = np.asarray([item[4] for item in arguments], dtype=np.uint64)
        w0 = np.asarray([state.walls_remaining[0] for state in states], dtype=np.int32)
        w1 = np.asarray([state.walls_remaining[1] for state in states], dtype=np.int32)
        turns = np.asarray([state.turn for state in states], dtype=np.int32)
        int_pointer = ctypes.POINTER(ctypes.c_int)
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        expected = len(states) * 8 * size * size
        count = self.library.bz_encode_canonical_batch_f32(
            size, len(states),
            p0.ctypes.data_as(int_pointer), p1.ctypes.data_as(int_pointer),
            horizontal.ctypes.data_as(uint64_pointer), vertical.ctypes.data_as(uint64_pointer),
            w0.ctypes.data_as(int_pointer), w1.ctypes.data_as(int_pointer),
            turns.ctypes.data_as(int_pointer),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), expected,
        )
        if count != expected:
            raise RuntimeError(
                f"native canonical batch encoder returned {count}, expected {expected}"
            )
        self._record("encoding", started)
        return output

    def prepare_batch(
        self, states: list[GameState]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not states:
            return (
                np.empty((0, 8, 0, 0), dtype=np.float32),
                np.zeros(1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
        started = time.perf_counter()
        size = states[0].size
        if any(state.size != size for state in states):
            raise ValueError("all states in a prepared batch must share board size")
        encoded = np.empty((len(states), 8, size, size), dtype=np.float32)
        arguments = [self._arguments(state) for state in states]
        p0 = np.asarray([item[1] for item in arguments], dtype=np.int32)
        p1 = np.asarray([item[2] for item in arguments], dtype=np.int32)
        horizontal = np.asarray([item[3] for item in arguments], dtype=np.uint64)
        vertical = np.asarray([item[4] for item in arguments], dtype=np.uint64)
        w0 = np.asarray([state.walls_remaining[0] for state in states], dtype=np.int32)
        w1 = np.asarray([state.walls_remaining[1] for state in states], dtype=np.int32)
        turns = np.asarray([state.turn for state in states], dtype=np.int32)
        winners = np.asarray(
            [-1 if state.winner is None else state.winner for state in states],
            dtype=np.int32,
        )
        offsets = np.empty(len(states) + 1, dtype=np.int32)
        actions = np.empty(len(states) * states[0].action_size, dtype=np.int32)
        int_pointer = ctypes.POINTER(ctypes.c_int)
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        expected = len(states) * 8 * size * size
        count = self.library.bz_prepare_batch_f32(
            size, len(states),
            p0.ctypes.data_as(int_pointer), p1.ctypes.data_as(int_pointer),
            horizontal.ctypes.data_as(uint64_pointer), vertical.ctypes.data_as(uint64_pointer),
            w0.ctypes.data_as(int_pointer), w1.ctypes.data_as(int_pointer),
            turns.ctypes.data_as(int_pointer), winners.ctypes.data_as(int_pointer),
            encoded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), expected,
            offsets.ctypes.data_as(int_pointer), actions.ctypes.data_as(int_pointer),
            len(actions),
        )
        if count < 0:
            raise RuntimeError(f"native prepared action buffer requires {-count} entries")
        if self.collect_metrics:
            elapsed = time.perf_counter() - started
            self.boundary_calls += 1
            self.boundary_seconds += elapsed
            self.batch_preparation_calls += 1
            self.batch_preparation_seconds += elapsed
        return encoded, offsets, actions[:count]


def load_rules_backend(
    prefer_native: bool = True, collect_metrics: bool = False
) -> RulesBackend:
    if prefer_native:
        try:
            return NativeRulesBackend(collect_metrics=collect_metrics)
        except OSError:
            pass
    return PythonRulesBackend(collect_metrics=collect_metrics)
