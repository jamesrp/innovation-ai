"""Standard-library HTTP server for the Innovation hot-seat QA UI."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import cast
from urllib.parse import urlparse

from innovation_ai.innovation.errors import InnovationEngineError
from innovation_ai.innovation.logs import GameLogError
from innovation_ai.innovation.replay import ReplayRecordingError
from innovation_ai.innovation.serialization import JsonValue, SerializationError, canonical_json
from innovation_ai.web.session import SessionConflict, WebGameSession

LOGGER = logging.getLogger(__name__)
_STATIC_PACKAGE = "innovation_ai.web.static"
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
_MAX_REQUEST_BYTES = 64 * 1024


class QaRequestHandler(BaseHTTPRequestHandler):
    """Serve one shared process-local QA game and its static browser client."""

    server: QaHttpServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.server.session.snapshot())
            return
        if path == "/api/log":
            body = f"{self.server.session.game_log_json()}\n".encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="innovation-game-seed-{self.server.session.seed}.json"',
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        static = _STATIC_FILES.get(path)
        if static is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        filename, content_type = static
        body = resources.files(_STATIC_PACKAGE).joinpath(filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/new":
                if not isinstance(payload, dict) or set(payload) != {"seed"}:
                    raise ValueError("request must contain exactly one seed")
                seed = payload["seed"]
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError("seed must be an integer")
                response = self.server.session.new_game_and_snapshot(seed)
            elif path == "/api/action":
                if not isinstance(payload, dict) or set(payload) != {
                    "action",
                    "game_id",
                    "revision",
                }:
                    raise ValueError("request must contain action, game_id, and revision")
                game_id = payload["game_id"]
                revision = payload["revision"]
                if not isinstance(game_id, str):
                    raise ValueError("game_id must be a string")
                if isinstance(revision, bool) or not isinstance(revision, int):
                    raise ValueError("revision must be an integer")
                response = self.server.session.submit_and_snapshot(
                    payload["action"], game_id=game_id, revision=revision
                )
            elif path == "/api/undo":
                if not isinstance(payload, dict) or set(payload) != {"game_id", "revision"}:
                    raise ValueError("undo request must contain game_id and revision")
                game_id = payload["game_id"]
                revision = payload["revision"]
                if not isinstance(game_id, str):
                    raise ValueError("game_id must be a string")
                if isinstance(revision, bool) or not isinstance(revision, int):
                    raise ValueError("revision must be an integer")
                response = self.server.session.undo_and_snapshot(game_id=game_id, revision=revision)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._send_json(response)
        except SessionConflict as error:
            self._send_json(
                {"error": str(error), "current": self.server.session.snapshot()},
                HTTPStatus.CONFLICT,
            )
        except (
            GameLogError,
            SerializationError,
            ValueError,
        ) as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))
        except (InnovationEngineError, ReplayRecordingError):
            LOGGER.exception("engine failure while serving QA request")
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "the engine failed while applying this action; inspect the service log",
            )
        except json.JSONDecodeError as error:
            self._send_error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {error.msg}")

    def _read_json(self) -> object:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        length = int(raw_length)
        if not 0 <= length <= _MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        return json.loads(self.rfile.read(length))

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = canonical_json(cast(JsonValue, payload)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def log_message(self, format: str, *args: object) -> None:
        """Route request logs through the library logger instead of stderr."""

        LOGGER.info("%s - %s", self.address_string(), format % args)


class QaHttpServer(ThreadingHTTPServer):
    """HTTP server carrying the single shared in-memory game session."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], session: WebGameSession) -> None:
        self.session = session
        super().__init__(address, QaRequestHandler)


def serve(host: str, port: int, seed: int) -> None:
    """Run the QA UI until interrupted."""

    server = QaHttpServer((host, port), WebGameSession(seed))
    LOGGER.info("Innovation QA UI listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
