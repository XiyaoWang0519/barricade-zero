"""Local HTTP server for the Barricade Zero gameplay UI."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import webbrowser

from .app import PLAYER_TYPES, UiApp, UiError

STATIC_DIR = Path(__file__).parent / "static"
MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def make_handler(app: UiApp, static_dir: Path = STATIC_DIR):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            print(f"[ui] {self.address_string()} {format % args}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api("GET", parsed)
                return
            self._serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api("POST", parsed)
                return
            self._send_json(404, {"error": "not found"})

        def _serve_static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            target = (static_dir / relative).resolve()
            if static_dir.resolve() not in target.parents and target != static_dir.resolve():
                self._send_json(403, {"error": "forbidden"})
                return
            if not target.is_file():
                self.send_response(404)
                self.end_headers()
                return
            data = target.read_bytes()
            content_type = MIME_TYPES.get(target.suffix, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            if length > 1_000_000:
                raise UiError("request too large")
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as error:
                raise UiError("invalid JSON") from error
            if not isinstance(payload, dict):
                raise UiError("JSON body must be an object")
            return payload

        def _handle_api(self, method: str, parsed) -> None:
            try:
                payload = self._dispatch(method, parsed)
            except UiError as error:
                message = str(error)
                status = 404 if message in {"not found"} or message.startswith("unknown game") else 400
                self._send_json(status, {"error": message})
                return
            except FileNotFoundError as error:
                self._send_json(404, {"error": str(error)})
                return
            except Exception as error:
                self._send_json(500, {"error": str(error)})
                return
            self._send_json(200, payload)

        def _dispatch(self, method: str, parsed) -> dict:
            path = parsed.path.rstrip("/") or "/"
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            if method == "GET" and path == "/api/checkpoints":
                return {"checkpoints": app.list_checkpoints()}
            if method == "GET" and path == "/api/meta":
                return {"player_types": list(PLAYER_TYPES)}
            if method == "POST" and path == "/api/games":
                body = self._read_json()
                return app.create_game(
                    board_size=int(body.get("board_size", 9)),
                    walls_per_player=int(body.get("walls_per_player", 10)),
                    player0=body.get("player0"),
                    player1=body.get("player1"),
                    simulations=int(body.get("simulations", 32)),
                    seed=int(body.get("seed", 1)),
                )
            parts = path.split("/")
            if len(parts) >= 4 and parts[1] == "api" and parts[2] == "games":
                game_id = parts[3]
                action = parts[4] if len(parts) > 4 else None
                if method == "GET" and action is None:
                    ply = int(query["ply"]) if "ply" in query else None
                    return app.get_game(game_id, ply)
                body = self._read_json() if method == "POST" else {}
                if method == "POST" and action == "move":
                    if "action" not in body:
                        raise UiError("action is required")
                    return app.apply_move(game_id, int(body["action"]))
                if method == "POST" and action == "step":
                    return app.step(game_id)
                if method == "POST" and action == "undo":
                    return app.undo(game_id)
                if method == "POST" and action == "reset":
                    return app.reset(game_id)
                if method == "POST" and action == "inspect":
                    ply = body.get("ply")
                    return app.inspect(
                        game_id,
                        checkpoint=body.get("checkpoint"),
                        simulations=int(body.get("simulations", 0)),
                        ply=None if ply is None else int(ply),
                    )
            raise UiError("not found")

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(host: str, port: int, app: UiApp, open_browser: bool = False) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{httpd.server_address[1]}"
    print(f"Barricade Zero UI at {url}")
    if open_browser:
        webbrowser.open(url)
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the Barricade Zero gameplay UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--open", action="store_true", help="open the UI in a browser")
    args = parser.parse_args(argv)
    app = UiApp(checkpoint_dir=args.checkpoint_dir, device=args.device)
    serve(args.host, args.port, app, open_browser=args.open)


if __name__ == "__main__":
    main()
