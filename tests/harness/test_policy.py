from __future__ import annotations

from dataclasses import replace

import pytest

from innovation_ai.harness.policy import (
    PlayerRelation,
    PublicTurnProgress,
    ValuePositionKind,
    build_afterstate_value_position,
    build_current_value_position,
    sanitize_decision_context,
)
from innovation_ai.innovation.actions import DecisionContext, DrawAction
from innovation_ai.innovation.protocol import apply_action, current_decision
from innovation_ai.innovation.state import (
    GameState,
    PlayerTurnCounters,
    TurnCounters,
    build_setup_state,
)
from innovation_ai.innovation.types import CardId, PlayerId


def _finish_starting_melds(seed: int = 501) -> GameState:
    state = build_setup_state(seed)
    decision = current_decision(state)
    assert decision is not None
    state = apply_action(state, decision.legal_actions[0]).state
    decision = current_decision(state)
    assert decision is not None
    state = apply_action(state, decision.legal_actions[0]).state
    return state


def test_current_position_contains_relative_boundary_and_public_turn_progress() -> None:
    state = _finish_starting_melds()
    viewer = state.active_player
    assert viewer is not None
    opponent = next(player for player in PlayerId if player is not viewer)
    counters = TurnCounters(
        tuple(
            PlayerTurnCounters(player, tucked=5 if player is viewer else 2, scored=1)
            for player in PlayerId
        )  # type: ignore[arg-type]
    )
    state = replace(state, turn_counters=counters)
    decision = current_decision(state)
    assert decision is not None

    position = build_current_value_position(state, decision)

    assert position.viewer is viewer
    assert position.position_kind is ValuePositionKind.CURRENT
    assert position.boundary.chooser_relation is PlayerRelation.SELF
    assert position.boundary.executor_relation is PlayerRelation.SELF
    assert position.boundary.dogma_activator_relation is PlayerRelation.NONE
    assert position.boundary.turn_progress.self_tucked == 5
    assert position.boundary.turn_progress.opponent_tucked == 2
    assert position.observation.player(opponent).player_id is opponent


def test_afterstate_reobserves_for_original_viewer_after_turn_rotates() -> None:
    state = _finish_starting_melds(502)
    original = state.active_player
    assert original is not None
    state = replace(state, paid_actions_remaining=1)
    decision = current_decision(state)
    assert decision is not None
    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))

    transition = apply_action(state, draw)
    assert transition.decision is not None
    assert transition.decision.chooser is not original
    position = build_afterstate_value_position(
        transition.state,
        original,
        transition.decision,
    )

    assert position.viewer is original
    assert position.observation.viewer is original
    assert position.boundary.chooser_relation is PlayerRelation.OPPONENT
    assert position.position_kind is ValuePositionKind.AFTERSTATE
    assert position.observation != transition.decision.observation


def test_context_sanitization_replaces_hidden_selected_identity_with_count() -> None:
    state = _finish_starting_melds(503)
    viewer = state.active_player
    assert viewer is not None
    decision = current_decision(state)
    assert decision is not None
    own_card = decision.observation.player(viewer).hand.known_cards[0]
    opponent = next(player for player in PlayerId if player is not viewer)
    hidden_card = state.player(opponent).hand[0]
    assert hidden_card not in decision.observation.player(opponent).hand.known_cards

    sanitized = sanitize_decision_context(
        DecisionContext(selected_so_far=(own_card, hidden_card)),
        decision.observation,
    )

    assert sanitized is not None
    assert sanitized.visible_selected_cards == (own_card,)
    assert sanitized.unknown_selected_count == 1


def test_current_position_rejects_a_decision_from_another_state() -> None:
    first = _finish_starting_melds(504)
    second = _finish_starting_melds(505)
    decision = current_decision(first)
    assert decision is not None

    with pytest.raises(ValueError, match="observation"):
        build_current_value_position(second, decision)


def test_value_contract_rejects_invalid_progress_and_card_ids_remain_semantic() -> None:
    with pytest.raises(ValueError, match="negative"):
        PublicTurnProgress(-1, 0, 0, 0)
    assert str(CardId.from_name("The Wheel")) == "the-wheel"
