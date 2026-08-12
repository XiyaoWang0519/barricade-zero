"""Dependency-free neural input and policy transformations."""

from __future__ import annotations

from collections import deque
from typing import Sequence

from .actions import rotate_action
from .state import GameState


def _zeros(size: int) -> list[list[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def _constant(size: int, value: float) -> list[list[float]]:
    return [[value for _ in range(size)] for _ in range(size)]


def _distance_map(state: GameState, player: int) -> list[list[float]]:
    goal_row = 0 if player == 0 else state.size - 1
    distances = [[-1 for _ in range(state.size)] for _ in range(state.size)]
    queue = deque()
    for col in range(state.size):
        distances[goal_row][col] = 0
        queue.append((goal_row, col))
    while queue:
        cell = queue.popleft()
        for nxt in state.neighbors(cell):
            if distances[nxt[0]][nxt[1]] == -1:
                distances[nxt[0]][nxt[1]] = distances[cell[0]][cell[1]] + 1
                queue.append(nxt)
    scale = max(1, state.size * state.size - 1)
    return [[max(0, value) / scale for value in row] for row in distances]


def encode_state(state: GameState, max_walls: int | None = None) -> list[list[list[float]]]:
    state = state.canonical()
    max_walls = max_walls if max_walls is not None else max(state.walls_remaining, default=1)
    max_walls = max(1, max_walls)
    planes = [_zeros(state.size) for _ in range(4)]
    planes[0][state.pawns[0][0]][state.pawns[0][1]] = 1.0
    planes[1][state.pawns[1][0]][state.pawns[1][1]] = 1.0
    for row, col in state.horizontal_walls:
        planes[2][row][col] = 1.0
    for row, col in state.vertical_walls:
        planes[3][row][col] = 1.0
    planes.append(_constant(state.size, state.walls_remaining[0] / max_walls))
    planes.append(_constant(state.size, state.walls_remaining[1] / max_walls))
    planes.append(_distance_map(state, 0))
    planes.append(_distance_map(state, 1))
    return planes


def legal_action_mask(state: GameState) -> list[bool]:
    mask = [False] * state.action_size
    for action in state.legal_actions():
        mask[action] = True
    return mask


def rotate_policy(policy: Sequence[float], size: int) -> list[float]:
    if len(policy) != 8 + 2 * (size - 1) ** 2:
        raise ValueError("policy length does not match board size")
    rotated = [0.0] * len(policy)
    for action, value in enumerate(policy):
        rotated[rotate_action(action, size)] = value
    return rotated
