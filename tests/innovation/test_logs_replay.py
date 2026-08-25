from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    effects_fingerprint,
    load_effect_programs,
    start_dogma,
)
from innovation_ai.innovation.logs import (
    GameLog,
    GameLogError,
    ReplayOutcome,
    dumps_game_log,
    load_game_log,
    loads_game_log,
    save_game_log,
)
from innovation_ai.innovation.protocol import current_decisions
from innovation_ai.innovation.replay import (
    GameLogRecorder,
    ReplayCompatibilityError,
    ReplayDivergenceError,
    ReplayRecordingError,
    replay_game_log,
)
from innovation_ai.innovation.serialization import dumps_state, loads_state
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    build_explicit_state,
    build_setup_state,
    state_hash,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId


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


def test_a_mid_dogma_log_round_trips_and_replays_to_the_same_decision() -> None:
    """A dogma action that pauses on a choice is an ordinary decision boundary."""

    registry = load_card_registry()
    programs = load_effect_programs()
    state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    board=((Color.PURPLE, (CardId("code-of-laws"),)),),
                    hand=(CardId("city-states"),),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),)),
            ),
        ),
    )
    paused = start_dogma(state, CardId("code-of-laws"), PlayerId.PLAYER_1, programs, registry).state
    assert paused.pending_effects
    decisions = current_decisions(paused, registry, programs)
    assert len(decisions) == 1

    encoded = dumps_state(paused)
    restored = loads_state(encoded, registry)
    assert state_hash(restored) == state_hash(paused)
    assert current_decisions(restored, registry, programs) == decisions


def test_the_effects_fingerprint_is_recorded_and_verified() -> None:
    log = _complete_log(1020)
    assert log.effects_fingerprint == effects_fingerprint()
    payload = json.loads(dumps_game_log(log))
    payload["effects_fingerprint"] = "sha256:" + "0" * 64
    tampered = loads_game_log(json.dumps(payload))
    with pytest.raises(ReplayCompatibilityError, match="effects fingerprint"):
        replay_game_log(tampered)


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
