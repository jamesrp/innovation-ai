from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from innovation_ai.innovation.actions import (
    AchieveAction,
    ActionKind,
    ChooseBranchAction,
    ChooseCardsAction,
    ChooseStartingMeldAction,
    DogmaAction,
    DrawAction,
    MeldAction,
    action_payload,
    decision_payload,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import InformationPolicy, observe
from innovation_ai.innovation.protocol import (
    IllegalAction,
    apply_action,
    current_decision,
    current_decisions,
)
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    TerminalReason,
    build_setup_state,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.innovation.zones import (
    CardLocation,
    ZoneKind,
    exchange_cards,
    locate_card,
    meld_card,
    rearrange_stack,
    score_card,
)


def _finish_setup(state: GameState, registry: CardRegistry) -> GameState:
    first = current_decision(state, registry)
    assert first is not None
    first_transition = apply_action(state, first.legal_actions[0], registry)
    second = first_transition.decision
    assert second is not None
    final = apply_action(first_transition.state, second.legal_actions[0], registry)
    assert final.decision is not None
    return final.state


def _cards_available_for_board(
    state: GameState, registry: CardRegistry, color: Color, count: int
) -> tuple[CardId, ...]:
    cards = tuple(
        card.id
        for card in registry.cards
        if card.color is color
        and locate_card(state, card.id).kind is not ZoneKind.NORMAL_ACHIEVEMENT
    )
    assert len(cards) >= count
    return cards[:count]


def test_setup_choices_are_simultaneous_and_choose_first_player_by_title() -> None:
    registry = load_card_registry()
    initial = build_setup_state(310, registry)
    pending = current_decisions(initial, registry)
    assert tuple(decision.chooser for decision in pending) == tuple(PlayerId)
    assert tuple(decision.decision_id for decision in pending) == (1, 2)
    first = pending[0]
    assert first is not None
    assert first.chooser is PlayerId.PLAYER_1
    assert all(isinstance(action, ChooseStartingMeldAction) for action in first.legal_actions)
    assert (
        tuple(
            action.card_id
            for action in first.legal_actions
            if isinstance(action, ChooseStartingMeldAction)
        )
        == initial.players[0].hand
    )

    player_one_card = first.legal_actions[-1]
    assert isinstance(player_one_card, ChooseStartingMeldAction)
    after_one = apply_action(initial, player_one_card, registry)
    assert after_one.state.players[0].board == initial.players[0].board
    assert after_one.decision is not None
    assert after_one.decision.chooser is PlayerId.PLAYER_2
    assert after_one.decision == pending[1]
    assert not hasattr(after_one.decision.observation, "starting_meld_choices")

    player_two_card = after_one.decision.legal_actions[0]
    assert isinstance(player_two_card, ChooseStartingMeldAction)
    complete = apply_action(after_one.state, player_two_card, registry)
    state = complete.state
    assert state.phase is GamePhase.PLAY
    assert state.turn_number == 1
    assert state.paid_actions_remaining == 1
    assert state.starting_meld_choices == (None, None)
    assert isinstance(player_one_card, ChooseStartingMeldAction)
    assert isinstance(player_two_card, ChooseStartingMeldAction)
    expected = (
        PlayerId.PLAYER_1
        if registry.card(player_one_card.card_id).name.casefold()
        < registry.card(player_two_card.card_id).name.casefold()
        else PlayerId.PLAYER_2
    )
    assert state.active_player is expected


def test_either_simultaneous_setup_decision_may_be_submitted_first() -> None:
    registry = load_card_registry()
    initial = build_setup_state(318, registry)
    player_one, player_two = current_decisions(initial, registry)

    after_two = apply_action(initial, player_two.legal_actions[-1], registry)
    assert after_two.decision == player_one
    assert after_two.state.players[1].board == initial.players[1].board
    complete = apply_action(after_two.state, player_one.legal_actions[0], registry)
    assert complete.state.phase is GamePhase.PLAY
    assert complete.decision is not None
    assert complete.decision.decision_id == initial.next_decision_id


def test_first_turn_has_one_action_then_every_turn_has_two() -> None:
    registry = load_card_registry()
    state = _finish_setup(build_setup_state(311, registry), registry)
    first_player = state.active_player
    assert first_player is not None

    first_decision = current_decision(state, registry)
    assert first_decision is not None
    after_first = apply_action(state, first_decision.legal_actions[0], registry)
    assert after_first.decision is not None
    assert after_first.state.active_player is not first_player
    assert after_first.state.turn_number == 2
    assert after_first.state.paid_actions_remaining == 2

    after_one_of_two = apply_action(
        after_first.state, after_first.decision.legal_actions[0], registry
    )
    assert after_one_of_two.decision is not None
    assert after_one_of_two.state.active_player is not first_player
    assert after_one_of_two.state.paid_actions_remaining == 1

    after_two_of_two = apply_action(
        after_one_of_two.state, after_one_of_two.decision.legal_actions[0], registry
    )
    assert after_two_of_two.decision is not None
    assert after_two_of_two.state.active_player is first_player
    assert after_two_of_two.state.turn_number == 3
    assert after_two_of_two.state.paid_actions_remaining == 2


def test_every_enumerated_paid_action_applies_and_input_is_unchanged() -> None:
    registry = load_card_registry()
    state = _finish_setup(build_setup_state(312, registry), registry)
    decision = current_decision(state, registry)
    assert decision is not None
    original = state

    assert isinstance(decision.legal_actions[0], DrawAction)
    kinds = tuple(action.kind for action in decision.legal_actions)
    assert kinds == tuple(sorted(kinds, key=tuple(ActionKind).index))
    assert any(isinstance(action, MeldAction) for action in decision.legal_actions)
    assert any(isinstance(action, DogmaAction) for action in decision.legal_actions)

    for action in decision.legal_actions:
        transition = apply_action(state, action, registry)
        assert transition.state != state
        if isinstance(action, DogmaAction):
            assert transition.effect_resolution_pending
            assert transition.state.pending_effects[0].source_card_id == action.card_id
        else:
            assert transition.decision is not None or transition.terminal is not None
    assert state == original


def test_eligible_normal_achievement_is_enumerated_and_claimed() -> None:
    registry = load_card_registry()
    state = _finish_setup(build_setup_state(319, registry), registry)
    active_player = state.active_player
    assert active_player is not None
    top = set(state.player(active_player).board.stack(color).top for color in Color)
    scoring_cards = tuple(
        card.id
        for card in registry.cards
        if card.id not in top
        and locate_card(state, card.id).kind is not ZoneKind.NORMAL_ACHIEVEMENT
    )[:5]
    for card_id in scoring_cards:
        state, _ = score_card(state, active_player, card_id, registry)

    decision = current_decision(state, registry)
    assert decision is not None
    achieve = next(action for action in decision.legal_actions if isinstance(action, AchieveAction))
    transition = apply_action(state, achieve, registry)
    assert achieve.achievement_id in transition.state.player(active_player).normal_achievements


def test_draw_beyond_age_ten_returns_typed_terminal_draw() -> None:
    registry = load_card_registry()
    state = _finish_setup(build_setup_state(320, registry), registry)
    supplied_cards = tuple(card_id for pile in state.supply.piles for card_id in pile)
    exhausted = replace(
        state,
        supply=replace(state.supply, piles=tuple(() for _ in range(10))),
        removed_cards=(*state.removed_cards, *supplied_cards),
    )
    decision = current_decision(exhausted, registry)
    assert decision is not None
    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
    transition = apply_action(exhausted, draw, registry)
    assert transition.terminal is not None
    assert transition.terminal.reason is TerminalReason.DRAW_BEYOND_AGE_10
    assert transition.terminal.is_draw


def test_stale_or_nonlegal_actions_raise_recoverable_illegal_action() -> None:
    registry = load_card_registry()
    state = _finish_setup(build_setup_state(313, registry), registry)
    decision = current_decision(state, registry)
    assert decision is not None

    with pytest.raises(IllegalAction) as raised:
        apply_action(state, DrawAction(decision.decision_id + 1), registry)
    assert raised.value.action.decision_id == decision.decision_id + 1
    assert raised.value.decision == decision

    absent = next(
        card.id
        for card in registry.cards
        if card.id not in state.player(state.active_player).hand  # type: ignore[arg-type]
    )
    with pytest.raises(IllegalAction):
        apply_action(state, MeldAction(decision.decision_id, absent), registry)


def test_private_hand_identity_swaps_produce_equal_opponent_observations() -> None:
    registry = load_card_registry()
    state = build_setup_state(314, registry)
    hidden = state.player(PlayerId.PLAYER_2).hand[0]
    replacement = state.supply.pile(1)[0]
    swapped, _ = exchange_cards(
        state,
        CardLocation.hand(PlayerId.PLAYER_2),
        (hidden,),
        CardLocation.supply(1),
        (replacement,),
        registry,
    )

    player_one_before = observe(state, PlayerId.PLAYER_1, registry)
    player_one_after = observe(swapped, PlayerId.PLAYER_1, registry)
    assert player_one_before == player_one_after
    assert observe(state, PlayerId.PLAYER_2, registry) != observe(
        swapped, PlayerId.PLAYER_2, registry
    )


def test_rulebook_policy_hides_unsplayed_covered_cards_but_open_policy_reveals_them() -> None:
    registry = load_card_registry()
    state = build_setup_state(315, registry)
    first, second, top = _cards_available_for_board(state, registry, Color.RED, 3)
    for card_id in (first, second, top):
        state, _ = meld_card(state, PlayerId.PLAYER_2, card_id, registry)
    rearranged, _ = rearrange_stack(
        state, PlayerId.PLAYER_2, Color.RED, (second, first, top), registry
    )

    private_before = observe(state, PlayerId.PLAYER_1, registry)
    private_after = observe(rearranged, PlayerId.PLAYER_1, registry)
    red_before = private_before.player(PlayerId.PLAYER_2).board[tuple(Color).index(Color.RED)]
    assert red_before.covered_count is None
    assert red_before.covered_cards == ()
    assert private_before == private_after

    public_before = observe(
        state, PlayerId.PLAYER_1, registry, policy=InformationPolicy.PUBLIC_COVERED
    )
    public_after = observe(
        rearranged, PlayerId.PLAYER_1, registry, policy=InformationPolicy.PUBLIC_COVERED
    )
    assert public_before != public_after


def test_action_and_decision_payloads_are_semantic_and_versioned() -> None:
    state = build_setup_state(316)
    decision = current_decision(state)
    assert decision is not None
    payload = decision_payload(decision)
    action = decision.legal_actions[0]

    assert payload["schema_version"] == 1
    assert payload["decision_id"] == decision.decision_id
    assert payload["kind"] == "starting-meld"
    assert payload["legal_actions"][0] == action_payload(action)  # type: ignore[index]
    assert action_payload(action)["kind"] == "choose-starting-meld"
    assert "card_id" in action_payload(action)

    assert ChooseCardsAction(1, (CardId("writing"), CardId("agriculture"))).card_ids == (
        CardId("agriculture"),
        CardId("writing"),
    )
    with pytest.raises(ValueError, match="branch ID"):
        ChooseBranchAction(1, "Display text")


def test_legal_action_order_is_hash_seed_independent() -> None:
    script = """
import json
from innovation_ai.innovation.protocol import current_decision, apply_action
from innovation_ai.innovation.state import build_setup_state
from innovation_ai.innovation.actions import decision_payload
s = build_setup_state(317)
for _ in range(2):
    d = current_decision(s)
    t = apply_action(s, d.legal_actions[0])
    s = t.state
print(json.dumps(decision_payload(current_decision(s)), sort_keys=True))
"""
    outputs = []
    for seed in ("1", "999"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script], text=True, env=environment
            ).strip()
        )
    assert outputs[0] == outputs[1]
    json.loads(outputs[0])
