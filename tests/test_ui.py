import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from barricade.actions import UP, encode_wall_action
from barricade.state import GameState

try:
    import torch
except ImportError:
    torch = None


class ViewTests(unittest.TestCase):
    def test_serialize_initial_state_includes_pawns_and_legal_moves(self):
        from ui.view import serialize_state

        state = GameState.initial(size=5, walls_per_player=2)
        payload = serialize_state(state)
        self.assertEqual(payload["size"], 5)
        self.assertEqual(payload["pawns"], [[4, 2], [0, 2]])
        self.assertEqual(payload["turn"], 0)
        self.assertFalse(payload["terminal"])
        self.assertEqual(payload["distances"], [4, 4])
        pawn_actions = {move["action"] for move in payload["legal"]["pawns"]}
        self.assertIn(UP, pawn_actions)
        self.assertTrue(payload["legal"]["walls"])
        wall = payload["legal"]["walls"][0]
        self.assertIn(wall["orientation"], ("H", "V"))

    def test_describe_pawn_and_wall_actions(self):
        from ui.view import describe_action

        state = GameState.initial(size=5, walls_per_player=2)
        self.assertIn("pawn", describe_action(state, UP))
        wall = encode_wall_action("H", 2, 1, 5)
        self.assertEqual(describe_action(state, wall), "H wall @ (2,1)")


class SessionTests(unittest.TestCase):
    def setUp(self):
        from ui.app import UiApp

        self.app = UiApp(checkpoint_dir=tempfile.mkdtemp())

    def test_human_move_switches_turn_and_records_ply(self):
        game = self.app.create_game(
            board_size=5,
            walls_per_player=2,
            player0={"type": "human"},
            player1={"type": "human"},
        )
        moved = self.app.apply_move(game["id"], UP)
        self.assertEqual(moved["state"]["turn"], 1)
        self.assertEqual(moved["state"]["pawns"][0], [3, 2])
        self.assertEqual(len(moved["plies"]), 1)
        self.assertEqual(moved["plies"][0]["player"], 0)

    def test_illegal_human_move_is_rejected(self):
        from ui.app import UiError

        game = self.app.create_game(
            board_size=5,
            walls_per_player=0,
            player0={"type": "human"},
            player1={"type": "human"},
        )
        with self.assertRaises(UiError):
            self.app.apply_move(game["id"], 1)

    def test_human_turn_cannot_be_stepped(self):
        from ui.app import UiError

        game = self.app.create_game(
            board_size=5,
            walls_per_player=0,
            player0={"type": "human"},
            player1={"type": "shortest_path"},
        )
        with self.assertRaises(UiError):
            self.app.step(game["id"])

    def test_shortest_path_agents_can_finish_a_game(self):
        game = self.app.create_game(
            board_size=5,
            walls_per_player=0,
            player0={"type": "shortest_path"},
            player1={"type": "shortest_path"},
        )
        for _ in range(20):
            game = self.app.step(game["id"])
            if game["state"]["terminal"]:
                break
        self.assertTrue(game["state"]["terminal"])
        self.assertIn(game["state"]["winner"], (0, 1))

    def test_undo_restores_previous_position(self):
        game = self.app.create_game(
            board_size=5,
            walls_per_player=0,
            player0={"type": "human"},
            player1={"type": "human"},
        )
        self.app.apply_move(game["id"], UP)
        undone = self.app.undo(game["id"])
        self.assertEqual(undone["state"]["pawns"][0], [4, 2])
        self.assertEqual(undone["plies"], [])

    def test_history_ply_is_view_only(self):
        game = self.app.create_game(
            board_size=5,
            walls_per_player=0,
            player0={"type": "human"},
            player1={"type": "human"},
        )
        self.app.apply_move(game["id"], UP)
        viewed = self.app.get_game(game["id"], ply=0)
        self.assertEqual(viewed["viewing_ply"], 0)
        self.assertFalse(viewed["live"])
        self.assertEqual(viewed["state"]["pawns"][0], [4, 2])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTests(unittest.TestCase):
    def test_infer_network_spec_from_state_dict(self):
        from neural.model import PolicyValueNetwork
        from ui.checkpoints import infer_network_spec

        model = PolicyValueNetwork(board_size=5, channels=8, residual_blocks=1)
        spec = infer_network_spec(model.state_dict())
        self.assertEqual(spec["board_size"], 5)
        self.assertEqual(spec["channels"], 8)
        self.assertEqual(spec["residual_blocks"], 1)

    def test_list_and_play_a_tiny_checkpoint(self):
        from neural.model import PolicyValueNetwork
        from training.checkpoint import save_checkpoint
        from training.learner import Learner
        from ui.app import UiApp

        model = PolicyValueNetwork(board_size=5, channels=8, residual_blocks=1)
        learner = Learner(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_001.pt"
            save_checkpoint(
                path,
                model,
                learner.optimizer,
                generation=1,
                metadata={"config": {"board_size": 5, "walls_per_player": 2, "channels": 8}},
            )
            app = UiApp(checkpoint_dir=directory)
            listed = app.list_checkpoints()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["generation"], 1)
            self.assertEqual(listed[0]["board_size"], 5)
            game = app.create_game(
                board_size=5,
                walls_per_player=0,
                player0={"type": "checkpoint", "path": "generation_001.pt"},
                player1={"type": "shortest_path"},
                simulations=2,
            )
            stepped = app.step(game["id"])
            self.assertEqual(len(stepped["plies"]), 1)
            self.assertIsNotNone(stepped["analysis"]["value"])
            inspected = app.inspect(game["id"], checkpoint="generation_001.pt")
            self.assertEqual(inspected["analysis"]["source"], "network")
            self.assertTrue(inspected["state"]["legal"]["pawns"])

    def test_checkpoint_board_size_mismatch_is_rejected(self):
        from neural.model import PolicyValueNetwork
        from training.checkpoint import save_checkpoint
        from training.learner import Learner
        from ui.app import UiApp, UiError

        model = PolicyValueNetwork(board_size=5, channels=8, residual_blocks=1)
        learner = Learner(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            save_checkpoint(path, model, learner.optimizer, generation=0)
            app = UiApp(checkpoint_dir=directory)
            with self.assertRaises(UiError):
                app.create_game(
                    board_size=9,
                    walls_per_player=10,
                    player0={"type": "checkpoint", "path": "tiny.pt"},
                    player1={"type": "human"},
                )


class ServerTests(unittest.TestCase):
    def setUp(self):
        from ui.app import UiApp
        from ui.server import make_handler

        self.directory = tempfile.TemporaryDirectory()
        self.app = UiApp(checkpoint_dir=self.directory.name)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.directory.cleanup()

    def _json(self, path, payload=None, method=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method or ("POST" if payload is not None else "GET"),
            headers={"Content-Type": "application/json"} if payload is not None else {},
        )
        try:
            with urllib.request.urlopen(request) as response:
                body = response.read()
                return response.status, json.loads(body) if body else None
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_index_page_is_served(self):
        with urllib.request.urlopen(self.base + "/") as response:
            html = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn("Barricade Zero", html)

    def test_create_and_fetch_game_over_http(self):
        status, created = self._json(
            "/api/games",
            {
                "board_size": 5,
                "walls_per_player": 0,
                "player0": {"type": "shortest_path"},
                "player1": {"type": "human"},
                "simulations": 8,
            },
        )
        self.assertEqual(status, 200)
        status, stepped = self._json(f"/api/games/{created['id']}/step", {})
        self.assertEqual(status, 200)
        self.assertEqual(stepped["state"]["turn"], 1)
        status, fetched = self._json(f"/api/games/{created['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["id"], created["id"])


if __name__ == "__main__":
    unittest.main()
