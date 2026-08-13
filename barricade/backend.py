"""Optional native rules backend with a tested Python fallback."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Protocol

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


def load_rules_backend(prefer_native: bool = True) -> RulesBackend:
    if prefer_native:
        try:
            return NativeRulesBackend()
        except OSError:
            pass
    return PythonRulesBackend()
