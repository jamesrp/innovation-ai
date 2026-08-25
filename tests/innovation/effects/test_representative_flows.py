from __future__ import annotations

import json
from dataclasses import replace

import pytest

from innovation_ai.innovation.actions import (
    ChooseCardAction,
    ChooseColorAction,
    Decision,
    DeclineAction,
    SemanticAction,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects import (
    EffectContext,
    EffectStatus,
    effect_runtime_payload,
    restore_effect_runtime,
    start_effect,
    submit_effect_action,
)
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

    # Arbitrary rearrangement is resolved incrementally: the chooser names the next card each
    # time, so ordering k cards costs at most k decisions of at most k actions rather than k!
    # actions. The final card is forced, so it needs no decision at all.
    original = state.player(PlayerId.PLAYER_1).board.stack(Color.BLUE).cards
    for expected in tuple(reversed(original))[:-1]:
        assert result.status is EffectStatus.AWAIT_DECISION
        assert result.decision is not None
        assert all(isinstance(action, ChooseCardAction) for action in result.decision.legal_actions)
        assert len(result.decision.legal_actions) <= len(original)
        action = _action(result.decision, ChooseCardAction, expected)
        result = submit_effect_action(_round_trip(result.state), action, PROGRAMS, registry)

    assert result.status is EffectStatus.COMPLETE
    stack = result.state.player(PlayerId.PLAYER_1).board.stack(Color.BLUE)
    assert stack.cards == tuple(reversed(original))
    assert stack.splay is SplayDirection.RIGHT
    rearrange = next(event for event in result.events if event.change is not None)
    assert rearrange.change is not None
    assert rearrange.change.kind is ChangeKind.REARRANGE

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
