from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from innovation_ai.web.server import QaHttpServer
from innovation_ai.web.session import WebGameSession


@contextmanager
def _server() -> Iterator[str]:
    server = QaHttpServer(("127.0.0.1", 0), WebGameSession(seed=3))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.load(response)  # type: ignore[no-any-return]


def _post_json(url: str, payload: object) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)  # type: ignore[no-any-return]


def test_server_serves_browser_and_applies_revision_guarded_action() -> None:
    with _server() as base:
        with urllib.request.urlopen(f"{base}/") as response:
            assert response.status == 200
            assert b"Innovation QA Table" in response.read()

        initial = _get_json(f"{base}/api/state")
        decision = initial["decision"]
        assert isinstance(decision, dict)
        action = decision["legal_actions"][0]["payload"]
        request = {
            "action": action,
            "game_id": initial["game_id"],
            "revision": initial["revision"],
        }
        advanced = _post_json(f"{base}/api/action", request)
        assert advanced["revision"] == 1
        assert advanced["decision"]["chooser"] == "player-2"

        try:
            _post_json(f"{base}/api/action", request)
        except urllib.error.HTTPError as error:
            assert error.code == 409
            conflict = json.load(error)
            assert conflict["current"]["revision"] == 1
        else:  # pragma: no cover - assertion branch
            raise AssertionError("stale HTTP action was accepted")


def test_server_rejects_malformed_new_game_request() -> None:
    with _server() as base:
        try:
            _post_json(f"{base}/api/new", {"seed": "not-an-integer"})
        except urllib.error.HTTPError as error:
            assert error.code == 400
            assert json.load(error)["error"] == "seed must be an integer"
        else:  # pragma: no cover - assertion branch
            raise AssertionError("malformed seed was accepted")
