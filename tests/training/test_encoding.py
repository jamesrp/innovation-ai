from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from innovation_ai.harness.policy import (
    ValuePosition,
    ValuePositionKind,
    build_current_value_position,
    build_value_position,
)
from innovation_ai.innovation.actions import ChooseCardAction, Decision, DecisionKind
from innovation_ai.innovation.observations import InformationPolicy, observe
from innovation_ai.innovation.protocol import current_decision
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GamePhase,
    GameState,
    SupplyState,
    TerminalReason,
    TerminalResult,
    build_explicit_state,
    build_setup_state,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.training.encoding import (
    EncoderCompatibilityError,
    FlatObservationEncoder,
    build_encoder_manifest,
    load_encoder_manifest,
)


def _relative_state(
    *,
    swapped: bool = False,
    information_policy: InformationPolicy = InformationPolicy.PUBLIC_COVERED,
) -> GameState:
    self_id = PlayerId.PLAYER_2 if swapped else PlayerId.PLAYER_1
    opponent_id = PlayerId.PLAYER_1 if swapped else PlayerId.PLAYER_2
    return build_explicit_state(
        positions=(
            (
                self_id,
                ExplicitPlayerPosition(
                    hand=(CardId("writing"),),
                    score_pile=(CardId("agriculture"),),
                    board=((Color.PURPLE, (CardId("code-of-laws"),)),),
                ),
            ),
            (
                opponent_id,
                ExplicitPlayerPosition(
                    hand=(CardId("archery"),),
                    board=((Color.GREEN, (CardId("sailing"),)),),
                ),
            ),
        ),
        active_player=self_id,
        turn_number=8,
        paid_actions_remaining=2,
        information_policy_version=information_policy.value,
    )


def _current_position(state: GameState) -> ValuePosition:
    decision = current_decision(state)
    assert decision is not None
    return build_current_value_position(state, decision)


def test_encoder_manifest_fixture_is_frozen_and_strict() -> None:
    generated = build_encoder_manifest()
    public_fixture = load_encoder_manifest(Path("docs/encoder_v1_public_covered_manifest.json"))
    legacy = build_encoder_manifest(
        information_policy_version=InformationPolicy.RULEBOOK_PRIVATE_COVERED.value
    )
    legacy_fixture = load_encoder_manifest(Path("docs/encoder_v1_manifest.json"))

    assert generated == public_fixture
    assert legacy == legacy_fixture
    assert generated.input_dimension == legacy.input_dimension == 4690
    assert generated.layout_fingerprint != legacy.layout_fingerprint
    assert legacy.layout_fingerprint == (
        "sha256:b472a8911f444bcf7920ff89fab2ff55aa23f054747242b538c32a7851c5b2b5"
    )
    with pytest.raises(ValueError, match="dimension"):
        replace(generated, input_dimension=generated.input_dimension + 1)


def test_encoder_has_fixed_contiguous_float32_shape_for_all_phases_and_decisions() -> None:
    encoder = FlatObservationEncoder()
    setup = build_setup_state(610)
    setup_position = _current_position(setup)
    play = _relative_state()
    play_position = _current_position(play)

    viewer = PlayerId.PLAYER_1
    observation = observe(play, viewer)
    effect_decision = Decision(
        decision_id=play.next_decision_id,
        kind=DecisionKind.EFFECT_CHOICE,
        chooser=viewer,
        executor=viewer,
        observation=observation,
        legal_actions=(ChooseCardAction(play.next_decision_id, CardId("writing")),),
    )
    effect_position = build_value_position(
        play,
        viewer,
        effect_decision,
        position_kind=ValuePositionKind.CURRENT,
    )
    terminal = replace(
        play,
        phase=GamePhase.TERMINAL,
        active_player=None,
        paid_actions_remaining=0,
        terminal_result=TerminalResult(TerminalReason.CARD_EFFECT, (viewer,)),
    )
    terminal_position = build_value_position(
        terminal,
        viewer,
        None,
        position_kind=ValuePositionKind.AFTERSTATE,
    )

    for position in (setup_position, play_position, effect_position, terminal_position):
        vector = encoder.encode(position)
        assert vector.shape == (encoder.manifest.input_dimension,)
        assert vector.dtype == np.float32
        assert vector.flags.c_contiguous
        assert np.isfinite(vector).all()


def test_encoder_is_viewpoint_relative_under_canonical_seat_swap() -> None:
    encoder = FlatObservationEncoder()

    first = encoder.encode(_current_position(_relative_state()))
    swapped = encoder.encode(_current_position(_relative_state(swapped=True)))

    np.testing.assert_array_equal(first, swapped)


def test_hidden_supply_order_does_not_change_position_encoding() -> None:
    state = _relative_state()
    hidden_reordered = replace(
        state,
        supply=SupplyState(tuple(tuple(reversed(pile)) for pile in state.supply.piles)),
    )
    first = _current_position(state)
    second = _current_position(hidden_reordered)
    assert first.observation == second.observation

    encoder = FlatObservationEncoder()
    np.testing.assert_array_equal(encoder.encode(first), encoder.encode(second))


def test_public_monument_progress_and_current_afterstate_marker_encode_differently() -> None:
    state = _relative_state()
    position = _current_position(state)
    progressed = replace(
        position,
        boundary=replace(
            position.boundary,
            turn_progress=replace(position.boundary.turn_progress, self_tucked=6),
        ),
    )
    afterstate = replace(position, position_kind=ValuePositionKind.AFTERSTATE)
    encoder = FlatObservationEncoder()

    current_vector = encoder.encode(position)
    progressed_vector = encoder.encode(progressed)
    afterstate_vector = encoder.encode(afterstate)

    assert not np.array_equal(current_vector, progressed_vector)
    assert not np.array_equal(current_vector, afterstate_vector)
    names = dict(encoder.inspect_nonzero(progressed))
    assert names["boundary.turn_progress.self.tucked"] == pytest.approx(6 / 105)
    assert names["position.kind.current"] == 1.0


def test_unknown_covered_count_differs_from_known_zero() -> None:
    position = _current_position(
        _relative_state(information_policy=InformationPolicy.RULEBOOK_PRIVATE_COVERED)
    )
    observation = position.observation
    opponent = observation.player(PlayerId.PLAYER_2)
    blue = opponent.board[0]
    assert blue.covered_count is None
    known_blue = replace(blue, covered_count=0)
    known_opponent = replace(opponent, board=(known_blue, *opponent.board[1:]))
    known_observation = replace(
        observation,
        players=(observation.player(PlayerId.PLAYER_1), known_opponent),
    )
    known_position = replace(position, observation=known_observation)
    encoder = FlatObservationEncoder(
        manifest=build_encoder_manifest(
            information_policy_version=InformationPolicy.RULEBOOK_PRIVATE_COVERED.value
        )
    )

    unknown = encoder.encode(position)
    known = encoder.encode(known_position)
    offset = encoder.manifest.offsets["player.opponent.stack.blue.covered_count.known"]
    assert unknown[offset] == 0.0
    assert known[offset] == 1.0
    assert not np.array_equal(unknown, known)


def test_value_position_and_encoder_manifest_mismatches_fail_loudly() -> None:
    encoder = FlatObservationEncoder()
    position = _current_position(_relative_state())
    with pytest.raises(ValueError, match="value-position"):
        replace(position, schema_version=99)

    wrong_fingerprint = replace(
        encoder.manifest,
        card_data_fingerprint="sha256:" + "0" * 64,
        layout_fingerprint="",
    )
    with pytest.raises(EncoderCompatibilityError, match="installed"):
        FlatObservationEncoder(manifest=wrong_fingerprint)
