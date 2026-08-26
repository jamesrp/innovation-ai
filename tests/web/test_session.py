from __future__ import annotations

from collections.abc import Mapping

from innovation_ai.innovation.logs import loads_game_log
from innovation_ai.innovation.replay import replay_game_log
from innovation_ai.web.session import SessionConflict, WebGameSession


def _first_action(snapshot: Mapping[str, object]) -> object:
    decision = snapshot["decision"]
    assert isinstance(decision, dict)
    actions = decision["legal_actions"]
    assert isinstance(actions, list)
    first = actions[0]
    assert isinstance(first, dict)
    return first["payload"]


def test_hotseat_session_advances_setup_without_leaking_the_other_hand() -> None:
    session = WebGameSession(seed=17)
    initial = session.snapshot()

    assert initial["phase"] == "starting-melds"
    assert initial["pending_decision_count"] == 2
    decision = initial["decision"]
    assert isinstance(decision, dict)
    assert decision["chooser"] == "player-1"

    session.submit_payload(_first_action(initial))
    second = session.snapshot()
    second_decision = second["decision"]
    assert isinstance(second_decision, dict)
    assert second_decision["chooser"] == "player-2"
    observation = second["observation"]
    assert isinstance(observation, dict)
    players = observation["players"]
    assert isinstance(players, list)
    player_one = players[0]
    assert isinstance(player_one, dict)
    hand = player_one["hand"]
    assert isinstance(hand, dict)
    assert hand["known_cards"] == []
    assert "Domestication" not in str(second["history"])
    assert "Pottery" not in str(second["history"])
    assert second["history"] == [
        {"number": 1, "label": "Secret starting meld selected", "kind": "choose-starting-meld"}
    ]

    session.submit_payload(_first_action(second))
    playing = session.snapshot()
    assert playing["phase"] == "play"
    assert playing["transition_count"] == 2


def test_undo_rebuilds_an_identical_replayable_boundary() -> None:
    session = WebGameSession(seed=29)
    first = session.snapshot()
    session.submit_payload(_first_action(first))
    expected = session.snapshot()
    session.submit_payload(_first_action(expected))

    assert session.undo()
    restored = session.snapshot()
    assert restored["state_hash"] == expected["state_hash"]
    assert restored["decision"] == expected["decision"]
    assert restored["history"] == expected["history"]

    log = loads_game_log(session.game_log_json())
    replayed = replay_game_log(log)
    assert replayed.transitions_replayed == 1
    assert session.undo()
    assert not session.undo()


def test_revision_guard_rejects_delayed_browser_actions() -> None:
    session = WebGameSession(seed=5)
    initial = session.snapshot()
    action = _first_action(initial)
    game_id = initial["game_id"]
    revision = initial["revision"]
    assert isinstance(game_id, str)
    assert isinstance(revision, int)

    advanced = session.submit_and_snapshot(action, game_id=game_id, revision=revision)
    assert advanced["revision"] == revision + 1

    try:
        session.submit_and_snapshot(action, game_id=game_id, revision=revision)
    except SessionConflict as error:
        assert "changed" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("stale action was accepted")


def test_snapshot_exposes_card_reference_and_human_action_labels() -> None:
    session = WebGameSession(seed=0)
    snapshot = session.snapshot()
    cards = snapshot["cards"]
    assert isinstance(cards, dict)
    assert len(cards) == 105
    pottery = cards["pottery"]
    assert isinstance(pottery, dict)
    assert pottery["name"] == "Pottery"
    assert pottery["dogma"]

    decision = snapshot["decision"]
    assert isinstance(decision, dict)
    actions = decision["legal_actions"]
    assert isinstance(actions, list)
    assert all(
        isinstance(item, dict) and str(item["label"]).startswith("Start with ") for item in actions
    )
