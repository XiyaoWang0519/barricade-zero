"""JSON views of Barricade positions for the gameplay UI."""

from __future__ import annotations

from barricade.actions import decode_wall_action
from barricade.state import GameState

MOVE_NAMES = ("up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right")
MOVE_GLYPHS = ("↑", "↓", "←", "→", "↖", "↗", "↙", "↘")


def describe_action(state: GameState, action: int) -> str:
    if action < 8:
        destinations = state.legal_pawn_destinations()
        glyph = MOVE_GLYPHS[action] if 0 <= action < 8 else str(action)
        if action in destinations:
            row, col = destinations[action]
            return f"pawn {glyph} → ({row},{col})"
        return f"pawn {glyph}"
    orientation, row, col = decode_wall_action(action, state.size)
    return f"{orientation} wall @ ({row},{col})"


def _analysis_lookup(analysis: dict | None) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    if not analysis:
        return lookup
    for item in analysis.get("actions") or []:
        lookup[int(item["action"])] = item
    return lookup


def serialize_state(state: GameState, analysis: dict | None = None) -> dict:
    stats = _analysis_lookup(analysis)
    pawns = []
    if state.is_terminal():
        return {
            "size": state.size,
            "pawns": [list(cell) for cell in state.pawns],
            "horizontal_walls": [list(wall) for wall in sorted(state.horizontal_walls)],
            "vertical_walls": [list(wall) for wall in sorted(state.vertical_walls)],
            "walls_remaining": list(state.walls_remaining),
            "turn": state.turn,
            "winner": state.winner,
            "terminal": True,
            "distances": [state.shortest_path_distance(0), state.shortest_path_distance(1)],
            "legal": {"pawns": [], "walls": []},
        }
    for action, (row, col) in state.legal_pawn_destinations().items():
        item = {
            "action": action,
            "row": row,
            "col": col,
            "name": MOVE_NAMES[action],
        }
        if action in stats:
            item.update({key: stats[action][key] for key in ("prior", "visits", "policy") if key in stats[action]})
        pawns.append(item)
    walls = []
    for action in state.legal_actions():
        if action < 8:
            continue
        orientation, row, col = decode_wall_action(action, state.size)
        item = {
            "action": action,
            "orientation": orientation,
            "row": row,
            "col": col,
        }
        if action in stats:
            item.update({key: stats[action][key] for key in ("prior", "visits", "policy") if key in stats[action]})
        walls.append(item)
    return {
        "size": state.size,
        "pawns": [list(cell) for cell in state.pawns],
        "horizontal_walls": [list(wall) for wall in sorted(state.horizontal_walls)],
        "vertical_walls": [list(wall) for wall in sorted(state.vertical_walls)],
        "walls_remaining": list(state.walls_remaining),
        "turn": state.turn,
        "winner": state.winner,
        "terminal": state.is_terminal(),
        "distances": [state.shortest_path_distance(0), state.shortest_path_distance(1)],
        "legal": {"pawns": pawns, "walls": walls},
    }
