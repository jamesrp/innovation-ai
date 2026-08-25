from __future__ import annotations

import json
from dataclasses import replace

import pytest

from innovation_ai.innovation.actions import (
    ChooseCardAction,
    ChooseColorAction,
    Decision,
    DeclineAction,
    FinishSelectionAction,
    OrderCardsAction,
    SemanticAction,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects import (
    EffectContext,
    EffectEventKind,
    EffectStatus,
    current_effect_decision,
    effect_runtime_payload,
    restore_effect_runtime,
    resume_effect,
    start_effect,
    step_effect,
    submit_effect_action,
)
from innovation_ai.innovation.effects.model import frame_value
from innovation_ai.innovation.effects.synthetic import synthetic_program_registry
from innovation_ai.innovation.state import GamePhase, GameState, build_setup_state, state_hash
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection
from innovation_ai.innovation.zones import (
    CardLocation,
    ChangeKind,
    ZoneKind,
    locate_card,
    meld_card,
    move_card,
    return_card,
    score_card,
    set_splay,
)

PROGRAMS = synthetic_program_registry()


def _play_state(registry: CardRegistry) -> GameState:
    return replace(
        build_setup_state(1, registry),
        phase=GamePhase.PLAY,
        active_player=PlayerId.PLAYER_1,
        turn_number=4,
        paid_actions_remaining=1,
    )


def _context(
    state: GameState,
    card: str,
    *,
    executor: PlayerId = PlayerId.PLAYER_1,
    activator: PlayerId = PlayerId.PLAYER_1,
    shared: bool = False,
) -> EffectContext:
    return EffectContext(
        actor=executor,
        chooser=executor,
        executor=executor,
        dogma_activator=activator,
        source_card_id=CardId(card),
        source_effect_id=None,
        turn_id=state.turn_number,
        dogma_action_id=7,
        shared=shared,
    )


def _move_to_hand(
    state: GameState, player_id: PlayerId, card_id: CardId, registry: CardRegistry
) -> GameState:
    updated, _ = move_card(
        state,
        card_id,
        CardLocation.hand(player_id),
        registry,
        kind=ChangeKind.TRANSFER,
    )
    return updated


def _clear_hand(state: GameState, player_id: PlayerId, registry: CardRegistry) -> GameState:
    updated = state
    for card_id in state.player(player_id).hand:
        updated, _ = move_card(
            updated,
            card_id,
            CardLocation.score(player_id),
            registry,
            kind=ChangeKind.TRANSFER,
        )
    return updated


def _round_trip(state: GameState) -> GameState:
    payload = json.loads(json.dumps(effect_runtime_payload(state), sort_keys=True))
    restored = restore_effect_runtime(
        replace(state, pending_effects=(), effect_variables=()), payload, PROGRAMS
    )
    assert restored == state
    assert state_hash(restored) == state_hash(state)
    return restored


def _action(
    decision: Decision | None, action_type: type[object], card_id: CardId | None = None
) -> SemanticAction:
    assert decision is not None
    for action in decision.legal_actions:
        if isinstance(action, action_type) and (
            card_id is None or getattr(action, "card_id", None) == card_id
        ):
            return action
    raise AssertionError(f"no {action_type.__name__} action found")


def test_pottery_bounded_selection_order_round_trips_and_decline() -> None:
    registry = load_card_registry()
    state = _play_state(registry)
    state, _ = meld_card(state, PlayerId.PLAYER_1, CardId("pottery"), registry)
    state = _move_to_hand(state, PlayerId.PLAYER_1, CardId("sailing"), registry)
    state = _move_to_hand(state, PlayerId.PLAYER_1, CardId("the-wheel"), registry)

    started = start_effect(
        state,
        "synthetic-pottery-v1",
        _context(state, "pottery"),
        PROGRAMS,
        registry,
    )
    assert started.status is EffectStatus.AWAIT_DECISION
    paused = _round_trip(started.state)
    decision = current_effect_decision(paused, PROGRAMS, registry)
    assert decision == started.decision
    assert decision is not None
    assert isinstance(decision.legal_actions[-1], FinishSelectionAction)

    first = _action(decision, ChooseCardAction, CardId("sailing"))
    after_first = submit_effect_action(paused, first, PROGRAMS, registry)
    assert after_first.status is EffectStatus.AWAIT_DECISION
    paused = _round_trip(after_first.state)
    second = _action(after_first.decision, ChooseCardAction, CardId("the-wheel"))
    after_second = submit_effect_action(paused, second, PROGRAMS, registry)
    assert after_second.status is EffectStatus.AWAIT_DECISION
    finish = _action(after_second.decision, FinishSelectionAction)
    ordering = submit_effect_action(
        _round_trip(after_second.state),
        finish,
        PROGRAMS,
        registry,
    )
    assert ordering.status is EffectStatus.AWAIT_DECISION
    assert ordering.decision is not None
    assert all(isinstance(action, OrderCardsAction) for action in ordering.decision.legal_actions)
    chosen_order = next(
        action
        for action in ordering.decision.legal_actions
        if isinstance(action, OrderCardsAction)
        and action.card_ids == (CardId("the-wheel"), CardId("sailing"))
    )

    restored = _round_trip(ordering.state)
    completed = submit_effect_action(restored, chosen_order, PROGRAMS, registry)
    direct = submit_effect_action(ordering.state, chosen_order, PROGRAMS, registry)
    assert completed.status is EffectStatus.COMPLETE
    assert state_hash(completed.state) == state_hash(direct.state)
    assert completed.state.supply.pile(1)[-2:] == (
        CardId("the-wheel"),
        CardId("sailing"),
    )
    assert sum(registry.card(card).age for card in completed.state.players[0].score_pile) >= 2
    assert any(
        event.change and event.change.kind is ChangeKind.RETURN for event in completed.events
    )
    assert any(event.change and event.change.kind is ChangeKind.SCORE for event in completed.events)

    declined = start_effect(
        state,
        "synthetic-pottery-v1",
        _context(state, "pottery"),
        PROGRAMS,
        registry,
    )
    decline = _action(declined.decision, FinishSelectionAction)
    declined = submit_effect_action(declined.state, decline, PROGRAMS, registry)
    assert declined.status is EffectStatus.COMPLETE
    assert not declined.events


def test_metalworking_repeat_reveal_branch_is_checkpoint_resumable() -> None:
    registry = load_card_registry()
    state = _play_state(registry)
    state, _ = meld_card(state, PlayerId.PLAYER_1, CardId("metalworking"), registry)
    state, _ = return_card(state, CardId("agriculture"), registry)
    pile = tuple(
        card
        for card in state.supply.pile(1)
        if card not in {CardId("archery"), CardId("agriculture")}
    )
    state = replace(
        state,
        supply=state.supply.replace_pile(1, (CardId("archery"), CardId("agriculture"), *pile)),
    )
    started = start_effect(
        state,
        "synthetic-metalworking-v1",
        _context(state, "metalworking"),
        PROGRAMS,
        registry,
        pause_before_first_step=True,
    )

    checkpoint = started.state
    for _ in range(8):
        checkpoint = _round_trip(checkpoint)
        result = step_effect(checkpoint, PROGRAMS, registry)
        checkpoint = result.state
        if result.status is not EffectStatus.CONTINUE:
            break
    restored = _round_trip(checkpoint)
    completed = resume_effect(restored, PROGRAMS, registry)
    assert completed.status is EffectStatus.COMPLETE
    assert CardId("archery") in completed.state.player(PlayerId.PLAYER_1).score_pile
    assert CardId("agriculture") in completed.state.player(PlayerId.PLAYER_1).hand
    assert sum(event.kind is EffectEventKind.REVEAL for event in completed.events) >= 1
    assert any(event.kind is EffectEventKind.KEEP for event in completed.events)

    direct = resume_effect(started.state, PROGRAMS, registry)
    assert state_hash(completed.state) == state_hash(direct.state)


def test_machinery_demand_exchange_is_atomic_and_has_full_provenance() -> None:
    registry = load_card_registry()
    state = _play_state(registry)
    state, _ = meld_card(state, PlayerId.PLAYER_1, CardId("machinery"), registry)
    state = _clear_hand(state, PlayerId.PLAYER_1, registry)
    state = _clear_hand(state, PlayerId.PLAYER_2, registry)
    for card in (CardId("archery"), CardId("sailing")):
        state = _move_to_hand(state, PlayerId.PLAYER_1, card, registry)
    for card in (CardId("agriculture"), CardId("the-wheel")):
        state = _move_to_hand(state, PlayerId.PLAYER_2, card, registry)
    before_one = state.player(PlayerId.PLAYER_1).hand
    before_two = state.player(PlayerId.PLAYER_2).hand

    result = start_effect(
        state,
        "synthetic-machinery-v1",
        _context(
            state,
            "machinery",
            executor=PlayerId.PLAYER_2,
            activator=PlayerId.PLAYER_1,
        ),
        PROGRAMS,
        registry,
    )
    assert result.status is EffectStatus.AWAIT_DECISION
    exchange = next(event for event in result.events if event.change is not None)
    assert exchange.change is not None and exchange.change.kind is ChangeKind.EXCHANGE
    assert exchange.demand
    assert exchange.executor is PlayerId.PLAYER_2
    assert exchange.dogma_activator is PlayerId.PLAYER_1
    assert result.state.player(PlayerId.PLAYER_1).hand == before_two
    assert result.state.player(PlayerId.PLAYER_2).hand == before_one
    assert len({event.atomic_group_id for event in result.events}) == 1

    while result.status is EffectStatus.AWAIT_DECISION:
        decision = result.decision
        assert decision is not None
        action = next(
            (action for action in decision.legal_actions if isinstance(action, ChooseCardAction)),
            decision.legal_actions[-1],
        )
        result = submit_effect_action(result.state, action, PROGRAMS, registry)
    assert result.status is EffectStatus.COMPLETE


def test_publications_arbitrary_order_preserves_splay_and_round_trips() -> None:
    registry = load_card_registry()
    state = _play_state(registry)
    for card in (CardId("calendar"), CardId("tools"), CardId("publications")):
        if locate_card(state, card).kind is ZoneKind.NORMAL_ACHIEVEMENT:
            pytest.skip("seed unexpectedly set aside a required fixture card")
        state, _ = meld_card(state, PlayerId.PLAYER_1, card, registry)
    state, _ = set_splay(state, PlayerId.PLAYER_1, Color.BLUE, SplayDirection.RIGHT, registry)
    result = start_effect(
        state,
        "synthetic-publications-v1",
        _context(state, "publications"),
        PROGRAMS,
        registry,
    )
    assert result.status is EffectStatus.AWAIT_DECISION
    color = _action(result.decision, ChooseColorAction)
    result = submit_effect_action(_round_trip(result.state), color, PROGRAMS, registry)
    assert result.status is EffectStatus.AWAIT_DECISION
    assert result.decision is not None
    original = state.player(PlayerId.PLAYER_1).board.stack(Color.BLUE).cards
    reverse = next(
        action
        for action in result.decision.legal_actions
        if isinstance(action, OrderCardsAction) and action.card_ids == tuple(reversed(original))
    )
    result = submit_effect_action(_round_trip(result.state), reverse, PROGRAMS, registry)
    assert result.status is EffectStatus.COMPLETE
    stack = result.state.player(PlayerId.PLAYER_1).board.stack(Color.BLUE)
    assert stack.cards == tuple(reversed(original))
    assert stack.splay is SplayDirection.RIGHT
    assert result.events[0].change is not None
    assert result.events[0].change.kind is ChangeKind.REARRANGE

    decline = start_effect(
        state,
        "synthetic-publications-v1",
        _context(state, "publications"),
        PROGRAMS,
        registry,
    )
    action = _action(decline.decision, DeclineAction)
    decline = submit_effect_action(decline.state, action, PROGRAMS, registry)
    assert decline.status is EffectStatus.COMPLETE


def test_fission_mass_removal_aborts_dogma_but_preserves_turn_and_achievements() -> None:
    registry = load_card_registry()
    state = _play_state(registry)
    state, _ = meld_card(state, PlayerId.PLAYER_1, CardId("fission"), registry)
    state, _ = score_card(state, PlayerId.PLAYER_2, CardId("agriculture"), registry)
    red_ten = CardId("robotics")
    pile = tuple(card for card in state.supply.pile(10) if card != red_ten)
    state = replace(state, supply=state.supply.replace_pile(10, (red_ten, *pile)))
    paid_actions = state.paid_actions_remaining
    achievements = tuple(
        (player.normal_achievements, player.special_achievements) for player in state.players
    )

    result = start_effect(
        state,
        "synthetic-fission-v1",
        _context(
            state,
            "fission",
            executor=PlayerId.PLAYER_2,
            activator=PlayerId.PLAYER_1,
            shared=True,
        ),
        PROGRAMS,
        registry,
        pause_before_first_step=True,
    )
    checkpoint = result.state
    while True:
        top = checkpoint.pending_effects[-1]
        if frame_value(top, "node_id") == "mass-removal":
            break
        checkpoint = step_effect(checkpoint, PROGRAMS, registry).state
    restored = _round_trip(checkpoint)
    result = resume_effect(restored, PROGRAMS, registry)
    assert result.status is EffectStatus.ABORT_DOGMA
    assert result.state.paid_actions_remaining == paid_actions
    assert all(not player.hand and not player.score_pile for player in result.state.players)
    assert all(not stack.cards for player in result.state.players for stack in player.board.stacks)
    assert (
        tuple(
            (player.normal_achievements, player.special_achievements)
            for player in result.state.players
        )
        == achievements
    )
    removal = next(event for event in result.events if event.kind is EffectEventKind.CHANGE)
    assert removal.change is not None and removal.change.kind is ChangeKind.REMOVE
    assert removal.atomic_group_id is not None
    assert result.events[-1].kind is EffectEventKind.ABORT_DOGMA


def test_fission_non_red_draw_does_not_abort_or_remove() -> None:
    registry = load_card_registry()
    state = _play_state(registry)
    state, _ = meld_card(state, PlayerId.PLAYER_1, CardId("fission"), registry)
    non_red = CardId("databases")
    pile = tuple(card for card in state.supply.pile(10) if card != non_red)
    state = replace(state, supply=state.supply.replace_pile(10, (non_red, *pile)))
    before_board_count = sum(
        len(stack.cards) for player in state.players for stack in player.board.stacks
    )
    result = start_effect(
        state,
        "synthetic-fission-v1",
        _context(state, "fission", executor=PlayerId.PLAYER_2),
        PROGRAMS,
        registry,
    )
    assert result.status is EffectStatus.COMPLETE
    assert (
        sum(len(stack.cards) for player in result.state.players for stack in player.board.stacks)
        == before_board_count
    )
    assert non_red in result.state.player(PlayerId.PLAYER_2).hand


def test_self_service_nested_execution_skips_demand_and_preserves_outer_cause() -> None:
    registry = load_card_registry()
    state = _play_state(registry)
    for card in (CardId("self-service"), CardId("machinery")):
        state, _ = meld_card(state, PlayerId.PLAYER_1, card, registry)
    state = _move_to_hand(state, PlayerId.PLAYER_1, CardId("archery"), registry)
    opponent_hand = state.player(PlayerId.PLAYER_2).hand

    result = start_effect(
        state,
        "synthetic-self-service-v1",
        _context(state, "self-service", shared=True),
        PROGRAMS,
        registry,
    )
    choose_machinery = _action(result.decision, ChooseCardAction, CardId("machinery"))
    result = submit_effect_action(
        _round_trip(result.state),
        choose_machinery,
        PROGRAMS,
        registry,
    )
    assert result.status is EffectStatus.AWAIT_DECISION
    assert result.decision is not None
    assert result.decision.source is not None
    assert result.decision.source.effect_id is not None
    assert result.decision.source.effect_id.ordinal == 2
    assert result.state.player(PlayerId.PLAYER_2).hand == opponent_hand
    assert not result.events

    choose_castle = _action(result.decision, ChooseCardAction, CardId("archery"))
    result = submit_effect_action(
        _round_trip(result.state),
        choose_castle,
        PROGRAMS,
        registry,
    )
    assert result.status is EffectStatus.AWAIT_DECISION
    assert result.qualifying_changes == 1
    score_event = next(event for event in result.events if event.changed)
    assert score_event.nested
    assert score_event.shared
    assert not score_event.demand
    assert score_event.source_card_id == CardId("machinery")
    assert score_event.source_effect_id is not None
    assert score_event.source_effect_id.ordinal == 2

    decline = _action(result.decision, DeclineAction)
    result = submit_effect_action(_round_trip(result.state), decline, PROGRAMS, registry)
    assert result.status is EffectStatus.COMPLETE
    assert result.qualifying_changes == 1
    assert CardId("archery") in result.state.player(PlayerId.PLAYER_1).score_pile
