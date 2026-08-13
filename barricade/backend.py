"""Optional native rules backend with a tested Python fallback.

The future extension should expose ``legal_actions(state)`` and
``has_path_with_extra_wall(state, player, orientation, row, col)``. Keeping this
boundary state-oriented lets self-play/MCTS remain Python while moving the
9x9 wall/path hot loop to compiled code.
"""

from __future__ import annotations

from typing import Protocol

from .state import GameState


class RulesBackend(Protocol):
    name: str

    def legal_actions(self, state: GameState) -> list[int]: ...

    def has_path_with_extra_wall(
        self,
        state: GameState,
        player: int,
        orientation: str | None = None,
        row: int = 0,
        col: int = 0,
    ) -> bool: ...


class PythonRulesBackend:
    name = "python"

    def legal_actions(self, state: GameState) -> list[int]:
        return state.legal_actions()

    def has_path_with_extra_wall(
        self,
        state: GameState,
        player: int,
        orientation: str | None = None,
        row: int = 0,
        col: int = 0,
    ) -> bool:
        return state._has_path_with_extra_wall(player, orientation, row, col)


def load_rules_backend(prefer_native: bool = True) -> RulesBackend:
    if prefer_native:
        try:
            from barricade_native import NativeRulesBackend

            return NativeRulesBackend()
        except ImportError:
            pass
    return PythonRulesBackend()
