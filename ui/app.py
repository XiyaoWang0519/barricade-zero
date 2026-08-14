"""Gameplay sessions, agents, and checkpoint testing for the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from pathlib import Path
import threading
import uuid

from barricade.actions import decode_wall_action
from barricade.agents import RandomAgent, ShortestPathAgent
from barricade.mcts import MCTS, SearchResult, UniformEvaluator
from barricade.state import GameState, IllegalAction

from .checkpoints import list_checkpoints, load_evaluator
from .view import describe_action, serialize_state

PLAYER_TYPES = ("human", "random", "shortest_path", "uniform_mcts", "checkpoint")


class UiError(ValueError):
    pass


@dataclass
class PlayerSpec:
    type: str
    path: str | None = None

    def to_dict(self) -> dict:
        payload = {"type": self.type}
        if self.path:
            payload["path"] = self.path
        return payload


@dataclass
class Ply:
    action: int
    player: int
    description: str
    value: float | None = None
    dest: list[int] | None = None
    wall: dict | None = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "player": self.player,
            "description": self.description,
            "value": self.value,
            "dest": self.dest,
            "wall": self.wall,
        }


@dataclass
class GameSession:
    id: str
    board_size: int
    walls_per_player: int
    players: tuple[PlayerSpec, PlayerSpec]
    simulations: int
    seed: int
    history: list[GameState] = field(default_factory=list)
    plies: list[Ply] = field(default_factory=list)
    analysis: dict | None = None
    rng: random.Random = field(default_factory=random.Random)

    @property
    def state(self) -> GameState:
        return self.history[-1]


def _make_ply(state: GameState, action: int, value: float | None = None) -> Ply:
    dest = None
    wall = None
    if action < 8:
        dest = list(state.legal_pawn_destinations()[action])
    else:
        orientation, row, col = decode_wall_action(action, state.size)
        wall = {"orientation": orientation, "row": row, "col": col}
    return Ply(
        action,
        state.turn,
        describe_action(state, action),
        value,
        dest,
        wall,
    )


def _parse_player(raw: dict | None, label: str) -> PlayerSpec:
    raw = raw or {"type": "human"}
    kind = raw.get("type", "human")
    if kind not in PLAYER_TYPES:
        raise UiError(f"{label} type must be one of {', '.join(PLAYER_TYPES)}")
    path = raw.get("path")
    if kind == "checkpoint" and not path:
        raise UiError(f"{label} checkpoint path is required")
    return PlayerSpec(kind, str(path) if path else None)


def _top_actions(state: GameState, result: SearchResult, limit: int = 8) -> list[dict]:
    ranked = sorted(
        (
            (action, result.visits[action], result.policy[action], result.root_priors[action])
            for action in range(len(result.visits))
            if result.visits[action] or result.policy[action] or result.root_priors[action]
        ),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    top = []
    for action, visits, policy, prior in ranked[:limit]:
        top.append(
            {
                "action": action,
                "description": describe_action(state, action),
                "visits": visits,
                "policy": policy,
                "prior": prior,
            }
        )
    return top


def _analysis_from_search(
    state: GameState,
    result: SearchResult,
    value: float | None,
    source: str,
    checkpoint: str | None,
    simulations: int,
) -> dict:
    actions = []
    for action in state.legal_actions():
        item = {
            "action": action,
            "prior": result.root_priors[action],
            "visits": result.visits[action],
            "policy": result.policy[action],
        }
        actions.append(item)
    return {
        "source": source,
        "value": None if value is None else float(value),
        "checkpoint": checkpoint,
        "simulations": simulations,
        "top": _top_actions(state, result, 8),
        "actions": actions,
    }


def _analysis_from_policy(
    state: GameState,
    policy: list[float],
    value: float,
    checkpoint: str | None,
) -> dict:
    actions = []
    ranked = []
    for action in state.legal_actions():
        prior = float(policy[action])
        actions.append({"action": action, "prior": prior, "policy": prior, "visits": 0})
        ranked.append((action, prior))
    ranked.sort(key=lambda item: item[1], reverse=True)
    top = [
        {
            "action": action,
            "description": describe_action(state, action),
            "visits": 0,
            "policy": prior,
            "prior": prior,
        }
        for action, prior in ranked[:8]
    ]
    return {
        "source": "network",
        "value": float(value),
        "checkpoint": checkpoint,
        "simulations": 0,
        "top": top,
        "actions": actions,
    }


class UiApp:
    def __init__(self, checkpoint_dir: str | Path = "checkpoints", device: str = "cpu") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._games: dict[str, GameSession] = {}
        self._evaluators: dict[str, tuple] = {}
        self._lock = threading.RLock()

    def list_checkpoints(self) -> list[dict]:
        return list_checkpoints(self.checkpoint_dir)

    def _resolve_checkpoint(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_file():
            candidate = self.checkpoint_dir / path
        if not candidate.is_file():
            raise UiError(f"checkpoint not found: {path}")
        return candidate.resolve()

    def _evaluator(self, path: str):
        resolved = str(self._resolve_checkpoint(path))
        cached = self._evaluators.get(resolved)
        if cached is None:
            evaluator, info = load_evaluator(resolved, self.device)
            cached = (evaluator, info)
            self._evaluators[resolved] = cached
        return cached

    def _checkpoint_info(self, spec: PlayerSpec) -> dict | None:
        if spec.type != "checkpoint":
            return None
        _, info = self._evaluator(spec.path)
        return info

    def create_game(
        self,
        board_size: int = 9,
        walls_per_player: int = 10,
        player0: dict | None = None,
        player1: dict | None = None,
        simulations: int = 32,
        seed: int = 1,
    ) -> dict:
        if board_size < 3 or board_size % 2 == 0:
            raise UiError("board size must be an odd integer >= 3")
        if walls_per_player < 0:
            raise UiError("walls per player cannot be negative")
        if simulations <= 0:
            raise UiError("simulations must be positive")
        players = (_parse_player(player0, "player0"), _parse_player(player1, "player1"))
        for spec in players:
            info = self._checkpoint_info(spec)
            if info and info["board_size"] != board_size:
                raise UiError(
                    f"checkpoint {spec.path} was trained for {info['board_size']}x{info['board_size']}, "
                    f"not {board_size}x{board_size}"
                )
        session = GameSession(
            id=uuid.uuid4().hex[:8],
            board_size=board_size,
            walls_per_player=walls_per_player,
            players=players,
            simulations=simulations,
            seed=seed,
            history=[GameState.initial(board_size, walls_per_player)],
            rng=random.Random(seed),
        )
        with self._lock:
            self._games[session.id] = session
        return self._payload(session)

    def _session(self, game_id: str) -> GameSession:
        with self._lock:
            session = self._games.get(game_id)
        if session is None:
            raise UiError(f"unknown game: {game_id}")
        return session

    def get_game(self, game_id: str, ply: int | None = None) -> dict:
        return self._payload(self._session(game_id), ply)

    def apply_move(self, game_id: str, action: int) -> dict:
        session = self._session(game_id)
        with self._lock:
            if session.players[session.state.turn].type != "human":
                raise UiError("it is not a human turn")
            if session.state.is_terminal():
                raise UiError("game is already over")
            try:
                successor = session.state.apply_action(int(action))
            except IllegalAction as error:
                raise UiError(str(error)) from error
            session.plies.append(_make_ply(session.state, int(action)))
            session.history.append(successor)
            session.analysis = None
            return self._payload(session)

    def step(self, game_id: str) -> dict:
        session = self._session(game_id)
        with self._lock:
            spec = session.players[session.state.turn]
            if spec.type == "human":
                raise UiError("current player is human; send a move instead")
            if session.state.is_terminal():
                raise UiError("game is already over")
            action, analysis = self._choose(session, spec)
            ply = _make_ply(
                session.state,
                action,
                None if analysis is None else analysis.get("value"),
            )
            successor = session.state.apply_known_legal_action(action)
            session.plies.append(ply)
            session.history.append(successor)
            session.analysis = analysis
            return self._payload(session)

    def undo(self, game_id: str) -> dict:
        session = self._session(game_id)
        with self._lock:
            if len(session.history) <= 1:
                raise UiError("nothing to undo")
            session.history.pop()
            session.plies.pop()
            session.analysis = None
            return self._payload(session)

    def reset(self, game_id: str) -> dict:
        session = self._session(game_id)
        with self._lock:
            session.history = [GameState.initial(session.board_size, session.walls_per_player)]
            session.plies = []
            session.analysis = None
            session.rng = random.Random(session.seed)
            return self._payload(session)

    def inspect(
        self,
        game_id: str,
        checkpoint: str | None = None,
        simulations: int = 0,
        ply: int | None = None,
    ) -> dict:
        session = self._session(game_id)
        with self._lock:
            state = self._state_at(session, ply)
            if checkpoint:
                evaluator, info = self._evaluator(checkpoint)
                if info["board_size"] != state.size:
                    raise UiError(
                        f"checkpoint {checkpoint} was trained for {info['board_size']}x{info['board_size']}"
                    )
                display = info.get("path") or checkpoint
                if simulations > 0:
                    search = MCTS(
                        evaluator,
                        simulations=simulations,
                        rng=random.Random(session.rng.getrandbits(64)),
                    )
                    result = search.search(state, temperature=0, add_noise=False)
                    _, value = evaluator.evaluate(state)
                    analysis = _analysis_from_search(
                        state, result, value, "mcts", display, simulations
                    )
                else:
                    policy, value = evaluator.evaluate(state)
                    analysis = _analysis_from_policy(state, policy, value, display)
            else:
                raise UiError("checkpoint is required to inspect a position")
            live = ply is None or ply == len(session.history) - 1
            if live:
                session.analysis = analysis
            return self._payload(session, ply, analysis)

    def _choose(self, session: GameSession, spec: PlayerSpec) -> tuple[int, dict | None]:
        state = session.state
        if spec.type == "random":
            return RandomAgent(session.rng).choose_action(state), None
        if spec.type == "shortest_path":
            return ShortestPathAgent().choose_action(state), None
        if spec.type == "uniform_mcts":
            search = MCTS(
                UniformEvaluator(),
                simulations=session.simulations,
                rng=random.Random(session.rng.getrandbits(64)),
            )
            result = search.search(state, temperature=0, add_noise=False)
            analysis = _analysis_from_search(
                state, result, 0.0, "mcts", None, session.simulations
            )
            return result.best_action, analysis
        evaluator, info = self._evaluator(spec.path)
        search = MCTS(
            evaluator,
            simulations=session.simulations,
            rng=random.Random(session.rng.getrandbits(64)),
        )
        result = search.search(state, temperature=0, add_noise=False)
        _, value = evaluator.evaluate(state)
        analysis = _analysis_from_search(
            state, result, value, "mcts", info.get("path") or spec.path, session.simulations
        )
        return result.best_action, analysis

    def _state_at(self, session: GameSession, ply: int | None) -> GameState:
        if ply is None:
            return session.state
        if ply < 0 or ply >= len(session.history):
            raise UiError("ply is out of range")
        return session.history[ply]

    def _payload(self, session: GameSession, ply: int | None = None, analysis: dict | None = None) -> dict:
        viewing = len(session.history) - 1 if ply is None else ply
        if viewing < 0 or viewing >= len(session.history):
            raise UiError("ply is out of range")
        state = session.history[viewing]
        live = viewing == len(session.history) - 1
        if analysis is None:
            analysis = session.analysis if live else None
        return {
            "id": session.id,
            "board_size": session.board_size,
            "walls_per_player": session.walls_per_player,
            "simulations": session.simulations,
            "seed": session.seed,
            "players": [player.to_dict() for player in session.players],
            "ply_count": len(session.plies),
            "viewing_ply": viewing,
            "live": live,
            "plies": [item.to_dict() for item in session.plies],
            "analysis": analysis,
            "state": serialize_state(state, analysis),
        }
