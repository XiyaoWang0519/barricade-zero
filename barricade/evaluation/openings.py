"""Versioned fixed-opening suites for reproducible paired matches."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from barricade.state import GameState


FORMAT_VERSION = 1


def _canonical_json(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class Opening:
    opening_id: str
    state: GameState
    plies: int

    def to_dict(self) -> dict:
        return {
            "id": self.opening_id,
            "plies": self.plies,
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Opening":
        return cls(
            opening_id=str(data["id"]),
            plies=int(data["plies"]),
            state=GameState.from_dict(data["state"]),
        )


@dataclass(frozen=True)
class OpeningSuite:
    board_size: int
    walls_per_player: int
    seed: int
    min_plies: int
    max_plies: int
    openings: tuple[Opening, ...]
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"unsupported opening format version: {self.format_version}")
        if not self.openings:
            raise ValueError("opening suite cannot be empty")
        ids = set()
        keys = set()
        for opening in self.openings:
            state = opening.state
            if opening.opening_id in ids:
                raise ValueError(f"duplicate opening id: {opening.opening_id}")
            if state.canonical_key() in keys:
                raise ValueError(f"duplicate canonical opening: {opening.opening_id}")
            if state.size != self.board_size:
                raise ValueError("opening board size does not match suite")
            if state.is_terminal() or not state.legal_actions():
                raise ValueError(f"opening must be playable: {opening.opening_id}")
            if not state.has_path(0) or not state.has_path(1):
                raise ValueError(f"opening has a blocked player: {opening.opening_id}")
            ids.add(opening.opening_id)
            keys.add(state.canonical_key())

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "board_size": self.board_size,
            "walls_per_player": self.walls_per_player,
            "seed": self.seed,
            "min_plies": self.min_plies,
            "max_plies": self.max_plies,
            "openings": [opening.to_dict() for opening in self.openings],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpeningSuite":
        return cls(
            format_version=int(data["format_version"]),
            board_size=int(data["board_size"]),
            walls_per_player=int(data["walls_per_player"]),
            seed=int(data["seed"]),
            min_plies=int(data["min_plies"]),
            max_plies=int(data["max_plies"]),
            openings=tuple(Opening.from_dict(item) for item in data["openings"]),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


def generate_opening_suite(
    *,
    count: int,
    board_size: int,
    walls_per_player: int,
    min_plies: int,
    max_plies: int,
    seed: int,
    maximum_attempts: int | None = None,
) -> OpeningSuite:
    if count <= 0:
        raise ValueError("count must be positive")
    if min_plies < 0 or max_plies < min_plies:
        raise ValueError("invalid opening ply range")
    rng = random.Random(seed)
    attempts_limit = maximum_attempts or max(1_000, count * 100)
    openings = []
    seen = set()
    attempts = 0
    while len(openings) < count and attempts < attempts_limit:
        attempts += 1
        state = GameState.initial(board_size, walls_per_player)
        target_plies = rng.randint(min_plies, max_plies)
        actual_plies = 0
        for _ in range(target_plies):
            if state.is_terminal():
                break
            state = state.apply_known_legal_action(rng.choice(state.legal_actions()))
            actual_plies += 1
        if state.is_terminal():
            continue
        key = state.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        openings.append(
            Opening(
                opening_id=f"opening_{len(openings):04d}",
                state=state,
                plies=actual_plies,
            )
        )
    if len(openings) != count:
        raise RuntimeError(
            f"generated only {len(openings)} unique openings after {attempts} attempts"
        )
    return OpeningSuite(
        board_size=board_size,
        walls_per_player=walls_per_player,
        seed=seed,
        min_plies=min_plies,
        max_plies=max_plies,
        openings=tuple(openings),
    )


def save_opening_suite(path: str | Path, suite: OpeningSuite) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(suite.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_opening_suite(path: str | Path) -> OpeningSuite:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("opening suite must be a JSON object")
    return OpeningSuite.from_dict(data)
