from __future__ import annotations

import json
from dataclasses import replace

import pytest

from innovation_ai.innovation.actions import (
    AchieveAction,
    ChooseBranchAction,
    ChooseCardAction,
    ChooseCardsAction,
    ChooseColorAction,
    ChoosePlayerAction,
    ChooseSplayAction,
    ChooseStartingMeldAction,
    ChooseValueAction,
    DeclineAction,
    DogmaAction,
    DrawAction,
    FinishSelectionAction,
    MeldAction,
    OrderCardsAction,
)
from innovation_ai.innovation.effects import EffectContext, start_effect
from innovation_ai.innovation.effects.synthetic import synthetic_program_registry
from innovation_ai.innovation.protocol import current_decision
from innovation_ai.innovation.serialization import (
    SchemaVersionError,
    SerializationError,
    dumps_action,
    dumps_decision,
    dumps_state,
    dumps_terminal,
    loads_action,
    loads_decision,
    loads_state,
    loads_terminal,
    observation_from_payload,
    observation_payload,
    state_from_payload,
)
from innovation_ai.innovation.state import (
    LEGACY_INFORMATION_POLICY_VERSION,
    GamePhase,
    TerminalReason,
    TerminalResult,
    build_setup_state,
    state_payload,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    NormalAchievementId,
    PlayerId,
    SplayDirection,
)


def test_state_round_trip_preserves_resumable_pending_frame() -> None:
    state = replace(
        build_setup_state(901),
        phase=GamePhase.PLAY,
        active_player=PlayerId.PLAYER_1,
        turn_number=2,
        paid_actions_remaining=1,
        starting_meld_choices=(None, None),
    )
    source = CardId("calendar")
    context = EffectContext(
        actor=PlayerId.PLAYER_1,
        chooser=PlayerId.PLAYER_1,
        executor=PlayerId.PLAYER_1,
        dogma_activator=PlayerId.PLAYER_1,
        source_card_id=source,
        source_effect_id=None,
        turn_id=state.turn_number,
        dogma_action_id=1,
    )
    pending = start_effect(
        state,
        "synthetic-bounded-selection-v1",
        context,
        synthetic_program_registry(),
        pause_before_first_step=True,
    ).state

    encoded = dumps_state(pending)
    assert encoded == dumps_state(pending)
    assert loads_state(encoded) == pending
    assert json.loads(encoded) == state_payload(pending)


def test_all_current_semantic_action_shapes_round_trip() -> None:
    card_ids = (CardId("agriculture"), CardId("writing"))
    actions = (
        ChooseStartingMeldAction(7, card_ids[0]),
        DrawAction(7),
        MeldAction(7, card_ids[0]),
        DogmaAction(7, card_ids[0]),
        AchieveAction(7, NormalAchievementId.AGE_1),
        ChooseCardAction(7, card_ids[0]),
        ChooseCardsAction(7, card_ids),
        ChooseColorAction(7, Color.BLUE),
        ChoosePlayerAction(7, PlayerId.PLAYER_2),
        ChooseValueAction(7, 0),
        ChooseSplayAction(7, SplayDirection.UP),
        ChooseBranchAction(7, "take-left"),
        OrderCardsAction(7, card_ids),
        DeclineAction(7),
        FinishSelectionAction(7),
    )

    for action in actions:
        assert loads_action(dumps_action(action)) == action


def test_decision_observation_and_terminal_round_trip() -> None:
    decision = current_decision(build_setup_state(902))
    assert decision is not None
    terminal = TerminalResult(TerminalReason.CARD_EFFECT, (PlayerId.PLAYER_1,))

    assert loads_decision(dumps_decision(decision)) == decision
    assert (
        observation_from_payload(observation_payload(decision.observation)) == decision.observation
    )
    assert loads_terminal(dumps_terminal(terminal)) == terminal


def test_strict_schemas_reject_versions_unknown_fields_and_bad_json() -> None:
    state = build_setup_state(903)
    payload = state_payload(state)
    payload["schema_version"] = 999
    with pytest.raises(SchemaVersionError, match="state schema version"):
        state_from_payload(payload)

    action = json.loads(dumps_action(DrawAction(3)))
    action["display_name"] = "Draw"
    with pytest.raises(SerializationError, match="unexpected"):
        loads_action(json.dumps(action))
    with pytest.raises(SerializationError, match="invalid JSON"):
        loads_state(dumps_state(state)[:-12])


def test_legacy_information_policy_state_round_trips_without_reinterpretation() -> None:
    legacy = build_setup_state(905, information_policy_version=LEGACY_INFORMATION_POLICY_VERSION)

    restored = loads_state(dumps_state(legacy))
    assert restored == legacy
    assert restored.information_policy_version == LEGACY_INFORMATION_POLICY_VERSION


def test_state_load_rejects_incompatible_catalog_fingerprint() -> None:
    payload = state_payload(build_setup_state(904))
    setup = payload["setup"]
    assert isinstance(setup, dict)
    setup["card_data_fingerprint"] = "sha256:" + "0" * 64

    with pytest.raises(SchemaVersionError, match="fingerprint"):
        state_from_payload(payload)
