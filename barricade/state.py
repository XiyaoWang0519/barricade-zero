"""Immutable, configurable Barricade/Quoridor game state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
import json

from .actions import (
    DELTA_ACTION,
    MOVE_DELTAS,
    action_size,
    decode_wall_action,
    encode_wall_action,
)

Cell = tuple[int, int]
Wall = tuple[int, int]
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


class IllegalAction(ValueError):
    pass


@dataclass(frozen=True)
class GameState:
    size: int = 9
    pawns: tuple[Cell, Cell] = ((8, 4), (0, 4))
    horizontal_walls: frozenset[Wall] = frozenset()
    vertical_walls: frozenset[Wall] = frozenset()
    walls_remaining: tuple[int, int] = (10, 10)
    turn: int = 0
    winner: int | None = None

    def __post_init__(self) -> None:
        if self.size < 3:
            raise ValueError("board size must be at least 3")
        if self.turn not in (0, 1):
            raise ValueError("turn must be 0 or 1")
        if len(set(self.pawns)) != 2:
            raise ValueError("pawns cannot share a square")
        if any(not self.in_bounds(cell) for cell in self.pawns):
            raise ValueError("pawn outside board")
        if any(count < 0 for count in self.walls_remaining):
            raise ValueError("negative wall inventory")
        for row, col in self.horizontal_walls | self.vertical_walls:
            if not (0 <= row < self.size - 1 and 0 <= col < self.size - 1):
                raise ValueError("wall anchor outside board")

    @classmethod
    def initial(cls, size: int = 9, walls_per_player: int = 10) -> GameState:
        middle = size // 2
        return cls(
            size=size,
            pawns=((size - 1, middle), (0, middle)),
            walls_remaining=(walls_per_player, walls_per_player),
        )

    @property
    def action_size(self) -> int:
        return action_size(self.size)

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def edge_blocked(self, first: Cell, second: Cell) -> bool:
        row1, col1 = first
        row2, col2 = second
        if abs(row1 - row2) + abs(col1 - col2) != 1:
            raise ValueError("edge endpoints must be adjacent")
        if row1 != row2:
            upper = min(row1, row2)
            col = col1
            return (upper, col) in self.horizontal_walls or (upper, col - 1) in self.horizontal_walls
        left = min(col1, col2)
        row = row1
        return (row, left) in self.vertical_walls or (row - 1, left) in self.vertical_walls

    def neighbors(self, cell: Cell) -> list[Cell]:
        result = []
        for dr, dc in ORTHOGONAL:
            nxt = (cell[0] + dr, cell[1] + dc)
            if self.in_bounds(nxt) and not self.edge_blocked(cell, nxt):
                result.append(nxt)
        return result

    @cached_property
    def _blocked_edge_masks(self) -> tuple[int, int]:
        """Bitsets for row-crossing and column-crossing edges."""
        horizontal = 0
        vertical = 0
        for row, col in self.horizontal_walls:
            horizontal |= 1 << (row * self.size + col)
            horizontal |= 1 << (row * self.size + col + 1)
        for row, col in self.vertical_walls:
            vertical |= 1 << (row * (self.size - 1) + col)
            vertical |= 1 << ((row + 1) * (self.size - 1) + col)
        return horizontal, vertical

    def _has_path_with_extra_wall(
        self,
        player: int,
        orientation: str | None = None,
        row: int = 0,
        col: int = 0,
    ) -> bool:
        horizontal, vertical = self._blocked_edge_masks
        if orientation == "H":
            horizontal |= 1 << (row * self.size + col)
            horizontal |= 1 << (row * self.size + col + 1)
        elif orientation == "V":
            vertical |= 1 << (row * (self.size - 1) + col)
            vertical |= 1 << ((row + 1) * (self.size - 1) + col)

        size = self.size
        goal_row = 0 if player == 0 else size - 1
        start_row, start_col = self.pawns[player]
        start = start_row * size + start_col
        queue = deque([start])
        seen = 1 << start
        while queue:
            cell = queue.popleft()
            current_row, current_col = divmod(cell, size)
            if current_row == goal_row:
                return True
            candidates = []
            if current_row > 0 and not horizontal & (1 << ((current_row - 1) * size + current_col)):
                candidates.append(cell - size)
            if current_row + 1 < size and not horizontal & (1 << (current_row * size + current_col)):
                candidates.append(cell + size)
            if current_col > 0 and not vertical & (1 << (current_row * (size - 1) + current_col - 1)):
                candidates.append(cell - 1)
            if current_col + 1 < size and not vertical & (1 << (current_row * (size - 1) + current_col)):
                candidates.append(cell + 1)
            for nxt in candidates:
                bit = 1 << nxt
                if not seen & bit:
                    seen |= bit
                    queue.append(nxt)
        return False

    def has_path(self, player: int) -> bool:
        return self._has_path_with_extra_wall(player)

    def shortest_path_distance(self, player: int) -> int | None:
        goal_row = 0 if player == 0 else self.size - 1
        queue = deque([(self.pawns[player], 0)])
        seen = {self.pawns[player]}
        while queue:
            cell, distance = queue.popleft()
            if cell[0] == goal_row:
                return distance
            for nxt in self.neighbors(cell):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        return None

    def legal_pawn_destinations(self) -> dict[int, Cell]:
        me = self.pawns[self.turn]
        opponent = self.pawns[1 - self.turn]
        destinations: dict[int, Cell] = {}
        for dr, dc in ORTHOGONAL:
            adjacent = (me[0] + dr, me[1] + dc)
            if not self.in_bounds(adjacent) or self.edge_blocked(me, adjacent):
                continue
            action = DELTA_ACTION[(dr, dc)]
            if adjacent != opponent:
                destinations[action] = adjacent
                continue

            behind = (opponent[0] + dr, opponent[1] + dc)
            if self.in_bounds(behind) and not self.edge_blocked(opponent, behind):
                destinations[action] = behind
                continue

            for side_dr, side_dc in ((dc, dr), (-dc, -dr)):
                diagonal = (opponent[0] + side_dr, opponent[1] + side_dc)
                if self.in_bounds(diagonal) and not self.edge_blocked(opponent, diagonal):
                    delta = (dr + side_dr, dc + side_dc)
                    destinations[DELTA_ACTION[delta]] = diagonal
        return destinations

    def legal_pawn_actions(self) -> list[int]:
        return sorted(self.legal_pawn_destinations())

    def _wall_geometry_legal(self, orientation: str, row: int, col: int) -> bool:
        wall = (row, col)
        if wall in self.horizontal_walls or wall in self.vertical_walls:
            return False
        if orientation == "H":
            if (row, col - 1) in self.horizontal_walls or (row, col + 1) in self.horizontal_walls:
                return False
            return wall not in self.vertical_walls
        if (row - 1, col) in self.vertical_walls or (row + 1, col) in self.vertical_walls:
            return False
        return wall not in self.horizontal_walls

    def _with_wall(self, orientation: str, row: int, col: int) -> GameState:
        horizontal = self.horizontal_walls
        vertical = self.vertical_walls
        if orientation == "H":
            horizontal = horizontal | {(row, col)}
        else:
            vertical = vertical | {(row, col)}
        return GameState(
            size=self.size,
            pawns=self.pawns,
            horizontal_walls=frozenset(horizontal),
            vertical_walls=frozenset(vertical),
            walls_remaining=self.walls_remaining,
            turn=self.turn,
            winner=self.winner,
        )

    @cached_property
    def _legal_actions_tuple(self) -> tuple[int, ...]:
        if self.is_terminal():
            return ()
        actions = self.legal_pawn_actions()
        if self.walls_remaining[self.turn] == 0:
            return tuple(actions)
        for orientation in ("H", "V"):
            for row in range(self.size - 1):
                for col in range(self.size - 1):
                    if not self._wall_geometry_legal(orientation, row, col):
                        continue
                    if self._has_path_with_extra_wall(
                        0, orientation, row, col
                    ) and self._has_path_with_extra_wall(1, orientation, row, col):
                        actions.append(encode_wall_action(orientation, row, col, self.size))
        return tuple(actions)

    def legal_actions(self) -> list[int]:
        return list(self._legal_actions_tuple)

    def apply_action(self, action: int) -> GameState:
        if action not in self._legal_actions_tuple:
            raise IllegalAction(f"illegal action: {action}")
        return self.apply_known_legal_action(action)

    def apply_known_legal_action(self, action: int) -> GameState:
        """Apply an action already validated by the current search backend."""
        pawns = self.pawns
        horizontal = self.horizontal_walls
        vertical = self.vertical_walls
        walls = self.walls_remaining
        winner = None
        if action < 8:
            destination = self.legal_pawn_destinations()[action]
            mutable_pawns = list(pawns)
            mutable_pawns[self.turn] = destination
            pawns = tuple(mutable_pawns)
            goal_row = 0 if self.turn == 0 else self.size - 1
            if destination[0] == goal_row:
                winner = self.turn
        else:
            orientation, row, col = decode_wall_action(action, self.size)
            if orientation == "H":
                horizontal = horizontal | {(row, col)}
            else:
                vertical = vertical | {(row, col)}
            mutable_walls = list(walls)
            mutable_walls[self.turn] -= 1
            walls = tuple(mutable_walls)
        return GameState(
            size=self.size,
            pawns=pawns,
            horizontal_walls=frozenset(horizontal),
            vertical_walls=frozenset(vertical),
            walls_remaining=walls,
            turn=1 - self.turn,
            winner=winner,
        )

    def is_terminal(self) -> bool:
        return self.winner is not None

    def outcome_for_current_player(self) -> float:
        if not self.is_terminal():
            raise ValueError("game is not terminal")
        return 1.0 if self.winner == self.turn else -1.0

    def canonical(self) -> GameState:
        if self.turn == 0:
            return self

        def rotate_cell(cell: Cell) -> Cell:
            return self.size - 1 - cell[0], self.size - 1 - cell[1]

        def rotate_wall(wall: Wall) -> Wall:
            return self.size - 2 - wall[0], self.size - 2 - wall[1]

        winner = None if self.winner is None else 1 - self.winner
        return GameState(
            size=self.size,
            pawns=(rotate_cell(self.pawns[1]), rotate_cell(self.pawns[0])),
            horizontal_walls=frozenset(rotate_wall(wall) for wall in self.horizontal_walls),
            vertical_walls=frozenset(rotate_wall(wall) for wall in self.vertical_walls),
            walls_remaining=(self.walls_remaining[1], self.walls_remaining[0]),
            turn=0,
            winner=winner,
        )

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "pawns": [list(cell) for cell in self.pawns],
            "horizontal_walls": [list(wall) for wall in sorted(self.horizontal_walls)],
            "vertical_walls": [list(wall) for wall in sorted(self.vertical_walls)],
            "walls_remaining": list(self.walls_remaining),
            "turn": self.turn,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GameState:
        return cls(
            size=data["size"],
            pawns=tuple(tuple(cell) for cell in data["pawns"]),
            horizontal_walls=frozenset(tuple(wall) for wall in data["horizontal_walls"]),
            vertical_walls=frozenset(tuple(wall) for wall in data["vertical_walls"]),
            walls_remaining=tuple(data["walls_remaining"]),
            turn=data["turn"],
            winner=data.get("winner"),
        )

    def canonical_key(self) -> bytes:
        canonical = self.canonical()
        return json.dumps(canonical.to_dict(), sort_keys=True, separators=(",", ":")).encode()
