"""Fixed action encoding for Barricade/Quoridor."""

UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = range(8)

MOVE_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP_LEFT: (-1, -1),
    UP_RIGHT: (-1, 1),
    DOWN_LEFT: (1, -1),
    DOWN_RIGHT: (1, 1),
}
DELTA_ACTION = {delta: action for action, delta in MOVE_DELTAS.items()}
ROTATED_MOVE = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
    UP_LEFT: DOWN_RIGHT,
    UP_RIGHT: DOWN_LEFT,
    DOWN_LEFT: UP_RIGHT,
    DOWN_RIGHT: UP_LEFT,
}


def action_size(size: int) -> int:
    return 8 + 2 * (size - 1) ** 2


def encode_wall_action(orientation: str, row: int, col: int, size: int) -> int:
    width = size - 1
    if orientation not in {"H", "V"}:
        raise ValueError("orientation must be H or V")
    if not (0 <= row < width and 0 <= col < width):
        raise ValueError("wall anchor is outside the board")
    offset = 0 if orientation == "H" else width * width
    return 8 + offset + row * width + col


def decode_wall_action(action: int, size: int) -> tuple[str, int, int]:
    if not 8 <= action < action_size(size):
        raise ValueError("not a wall action")
    width = size - 1
    index = action - 8
    orientation = "H" if index < width * width else "V"
    index %= width * width
    return orientation, index // width, index % width


def rotate_action(action: int, size: int) -> int:
    if action < 8:
        if action < 0:
            raise ValueError("invalid action")
        return ROTATED_MOVE[action]
    orientation, row, col = decode_wall_action(action, size)
    return encode_wall_action(orientation, size - 2 - row, size - 2 - col, size)
