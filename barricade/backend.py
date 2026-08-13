"""Optional native rules backend with a tested Python fallback."""

from __future__ import annotations

import ctypes
from pathlib import Path
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

    def legal_actions(self, state: GameState) -> list[int]:
        return state.legal_actions()

    def has_path_with_extra_wall(
        self, state: GameState, player: int, orientation: str | None = None,
        row: int = 0, col: int = 0,
    ) -> bool:
        return state._has_path_with_extra_wall(player, orientation, row, col)

    def encode_state(self, state: GameState) -> list[list[list[float]]]:
        from .encoding import encode_state

        return encode_state(state)

    def encode_batch(self, states: list[GameState]) -> np.ndarray:
        return np.ascontiguousarray(
            [self.encode_state(state) for state in states], dtype=np.float32
        )


def _wall_mask(walls: frozenset[tuple[int, int]], size: int) -> int:
    width = size - 1
    return sum(1 << (row * width + col) for row, col in walls)


class NativeRulesBackend:
    name = "native"

    def __init__(self, library: str | Path | None = None) -> None:
        path = Path(library) if library else Path(__file__).parents[1] / "native" / "libbarricade_rules.so"
        self.library = ctypes.CDLL(str(path))
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
        capacity = state.action_size
        output = (ctypes.c_int * capacity)()
        winner = -1 if state.winner is None else state.winner
        count = self.library.bz_legal_actions(
            *self._arguments(state), *state.walls_remaining, state.turn, winner,
            output, capacity,
        )
        if count < 0:
            raise RuntimeError(f"native action buffer requires {-count} entries")
        return list(output[:count])

    def has_path_with_extra_wall(
        self, state: GameState, player: int, orientation: str | None = None,
        row: int = 0, col: int = 0,
    ) -> bool:
        extra = 0 if orientation is None else (1 if orientation == "H" else 2)
        return bool(self.library.bz_has_path(
            *self._arguments(state), player, extra, row, col
        ))

    def encode_state(self, state: GameState) -> list[list[list[float]]]:
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
        return [
            [flat[plane * cells + row * size:plane * cells + (row + 1) * size]
             for row in range(size)]
            for plane in range(8)
        ]

    def encode_batch(self, states: list[GameState]) -> np.ndarray:
        if not states:
            return np.empty((0, 8, 0, 0), dtype=np.float32)
        size = states[0].size
        if any(state.size != size for state in states):
            raise ValueError("all states in an encoding batch must share board size")
        output = np.empty((len(states), 8, size, size), dtype=np.float32)
        per_state = 8 * size * size
        for index, state in enumerate(states):
            pointer = output[index].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            count = self.library.bz_encode_state_f32(
                *self._arguments(state), *state.walls_remaining, pointer, per_state
            )
            if count != per_state:
                raise RuntimeError(f"native encoder returned {count}, expected {per_state}")
        return output


def load_rules_backend(prefer_native: bool = True) -> RulesBackend:
    if prefer_native:
        try:
            return NativeRulesBackend()
        except OSError:
            pass
    return PythonRulesBackend()
