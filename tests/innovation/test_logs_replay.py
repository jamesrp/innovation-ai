from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from innovation_ai.innovation.actions import DogmaAction
from innovation_ai.innovation.logs import (
    GameLog,
    GameLogError,
    ReplayOutcome,
    dumps_game_log,
    load_game_log,
    loads_game_log,
    save_game_log,
)
from innovation_ai.innovation.replay import (
    GameLogRecorder,
    ReplayCompatibilityError,
    ReplayDivergenceError,
    ReplayRecordingError,
    replay_game_log,
)
from innovation_ai.innovation.state import build_setup_state, state_hash


def _complete_log(seed: int = 1001) -> GameLog:
    recorder = GameLogRecorder(build_setup_state(seed))
    for _ in range(200):
        decisions = recorder.decisions()
        if not decisions:
            break
        recorder.submit(decisions[0].legal_actions[0])
    log = recorder.game_log()
    assert log.terminal_result is not None
    return log


def test_play_log_round_trip_replays_every_state_hash(tmp_path: Path) -> None:
    log = _complete_log()
    encoded = dumps_game_log(log)
    path = tmp_path / "game.json"
    save_game_log(log, path)

    assert path.read_text(encoding="utf-8") == encoded + "\n"
    loaded = load_game_log(path)
    assert loaded == log
    assert dumps_game_log(loaded) == encoded
    assert loaded.setup.seed == 1001
    assert loaded.setup.shuffled_piles
    assert loaded.transition_count == len(loaded.transitions) == 95
    assert all(entry.action in entry.decision.legal_actions for entry in loaded.transitions)

    result = replay_game_log(loaded)
    assert result.transitions_replayed == loaded.transition_count
    assert result.outcome is ReplayOutcome.TERMINAL
    assert result.terminal_result == loaded.terminal_result
    assert state_hash(result.state) == loaded.final_state_hash


def test_pending_effect_log_round_trips_for_later_wp4_resume() -> None:
    recorder = GameLogRecorder(build_setup_state(1002))
    recorder.submit(recorder.decisions()[0].legal_actions[0])
    recorder.submit(recorder.decisions()[0].legal_actions[0])
    turn_decision = recorder.decisions()[0]
    dogma = next(
        action for action in turn_decision.legal_actions if isinstance(action, DogmaAction)
    )
    recorder.submit(dogma)

    log = loads_game_log(dumps_game_log(recorder.game_log()))
    assert log.final_outcome is ReplayOutcome.EFFECT_RESOLUTION_PENDING
    result = replay_game_log(log)
    assert result.state.pending_effects
    assert result.outcome is ReplayOutcome.EFFECT_RESOLUTION_PENDING


def test_edited_action_or_hash_fails_loudly() -> None:
    payload = json.loads(dumps_game_log(_complete_log(1003)))
    payload["transitions"][0]["action"]["card_id"] = "not-a-real-card"
    edited_action = loads_game_log(json.dumps(payload))
    with pytest.raises(ReplayDivergenceError, match="not legal"):
        replay_game_log(edited_action)

    payload = json.loads(dumps_game_log(_complete_log(1004)))
    payload["transitions"][5]["state_hash"] = "sha256:" + "0" * 64
    edited_hash = loads_game_log(json.dumps(payload))
    with pytest.raises(ReplayDivergenceError, match="state hash differs"):
        replay_game_log(edited_hash)


def test_truncated_or_incompatible_log_fails_loudly() -> None:
    log = _complete_log(1005)
    encoded = dumps_game_log(log)
    with pytest.raises(GameLogError, match="invalid JSON"):
        loads_game_log(encoded[:-20])

    payload = json.loads(encoded)
    payload["transitions"].pop()
    with pytest.raises(GameLogError, match="transition count"):
        loads_game_log(json.dumps(payload))

    payload = json.loads(encoded)
    payload["engine_version"] = "999.0"
    incompatible = loads_game_log(json.dumps(payload))
    with pytest.raises(ReplayCompatibilityError, match="incompatible engine"):
        replay_game_log(incompatible)

    fingerprint = "sha256:" + "9" * 64
    incompatible_fingerprint = replace(
        incompatible,
        engine_version=log.engine_version,
        card_data_fingerprint=fingerprint,
        setup=replace(incompatible.setup, card_data_fingerprint=fingerprint),
    )
    with pytest.raises(ReplayCompatibilityError, match="card-data fingerprint"):
        replay_game_log(incompatible_fingerprint)


def test_replay_detects_edited_final_marker_and_recording_requires_setup() -> None:
    log = _complete_log(1006)
    with pytest.raises(ReplayDivergenceError, match="final state hash"):
        replay_game_log(replace(log, final_state_hash="sha256:" + "f" * 64))

    state = build_setup_state(1007)
    recorder = GameLogRecorder(state)
    recorder.submit(recorder.decisions()[0].legal_actions[0])
    with pytest.raises(ReplayRecordingError, match="setup boundary"):
        GameLogRecorder(recorder.state)
