from __future__ import annotations

from dataclasses import replace

import pytest

from innovation_ai.innovation.actions import DogmaAction, DrawAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.invariants import (
    InvariantViolation,
    assert_card_conservation,
    assert_icon_geometry,
    assert_legal_action_completeness,
    assert_observation_leak_resistance,
    assert_score_consistency,
    assert_state_properties,
    assert_terminal_immutability,
    assert_transition_purity,
    assert_turn_consistency,
    assert_unique_card_locations,
    checked_apply_action,
)
from innovation_ai.innovation.observations import InformationPolicy
from innovation_ai.innovation.protocol import (
    Transition,
    apply_action,
    current_decision,
    current_decisions,
)
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    NormalAchievementState,
    TerminalReason,
    TerminalState,
    build_setup_state,
    clone_state,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection
from innovation_ai.innovation.zones import (
    CardLocation,
    ZoneKind,
    exchange_cards,
    locate_card,
    meld_card,
    rearrange_stack,
    score_card,
    set_splay,
)


def _finish_setup(state: GameState, registry: CardRegistry) -> GameState:
    for _ in range(2):
        decision = current_decision(state, registry)
        assert decision is not None
        state = checked_apply_action(state, decision.legal_actions[0], registry).state
    return state


def _available_color_cards(
    state: GameState, registry: CardRegistry, color: Color, count: int
) -> tuple[CardId, ...]:
    cards = tuple(
        card.id
        for card in registry.cards
        if card.color is color
        and locate_card(state, card.id).kind not in {ZoneKind.NORMAL_ACHIEVEMENT, ZoneKind.BOARD}
    )
    assert len(cards) >= count
    return cards[:count]


def _terminal_by_exhaustion(state: GameState, registry: CardRegistry) -> GameState:
    state = _finish_setup(state, registry)
    supply_cards = tuple(card_id for pile in state.supply.piles for card_id in pile)
    state = replace(
        state,
        supply=replace(state.supply, piles=tuple(() for _ in range(10))),
        removed_cards=(*state.removed_cards, *supply_cards),
    )
    decision = current_decision(state, registry)
    assert decision is not None
    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
    transition = checked_apply_action(state, draw, registry)
    assert transition.terminal is not None
    return transition.state


def test_reusable_state_properties_cover_setup_play_geometry_and_scores() -> None:
    registry = load_card_registry()
    state = build_setup_state(1001, registry)
    assert_state_properties(state, registry)

    state = _finish_setup(state, registry)
    first, second, third = _available_color_cards(state, registry, Color.RED, 3)
    for card_id in (first, second, third):
        state, _ = meld_card(state, PlayerId.PLAYER_1, card_id, registry)
    state, _ = set_splay(state, PlayerId.PLAYER_1, Color.RED, SplayDirection.UP, registry)
    score_candidate = next(
        card.id
        for card in registry.cards
        if locate_card(state, card.id).kind is not ZoneKind.NORMAL_ACHIEVEMENT
        and card.id not in (first, second, third)
    )
    state, _ = score_card(state, PlayerId.PLAYER_2, score_candidate, registry)

    assert_card_conservation(state, registry)
    assert_unique_card_locations(state)
    assert_score_consistency(state, registry)
    assert_icon_geometry(state, registry)
    assert_turn_consistency(state)
    assert_legal_action_completeness(state, registry)


def test_conservation_and_unique_location_fail_independently_and_loudly() -> None:
    registry = load_card_registry()
    state = build_setup_state(1002, registry)
    duplicated = state.supply.pile(1)[0]
    player = state.player(PlayerId.PLAYER_1)
    duplicate_state = state.replace_player(replace(player, hand=(*player.hand, duplicated)))

    with pytest.raises(InvariantViolation, match="unique locations"):
        assert_unique_card_locations(duplicate_state)
    with pytest.raises(InvariantViolation, match="conservation"):
        assert_card_conservation(duplicate_state, registry)

    piles = list(state.supply.piles)
    piles[0] = piles[0][1:]
    missing_state = replace(state, supply=replace(state.supply, piles=tuple(piles)))
    assert_unique_card_locations(missing_state)
    with pytest.raises(InvariantViolation, match="missing"):
        assert_card_conservation(missing_state, registry)


def test_legal_action_completeness_detects_an_omitted_action() -> None:
    registry = load_card_registry()
    state = build_setup_state(1003, registry)
    decisions = current_decisions(state, registry)
    incomplete = replace(decisions[0], legal_actions=decisions[0].legal_actions[:-1])

    with pytest.raises(InvariantViolation, match="starting actions"):
        assert_legal_action_completeness(
            state,
            registry,
            decisions=(incomplete, decisions[1]),
        )


def test_checked_transitions_cover_setup_turn_switch_and_a_real_dogma_action() -> None:
    registry = load_card_registry()
    state = build_setup_state(1004, registry)
    snapshot = clone_state(state)
    state = _finish_setup(state, registry)
    assert_transition_purity(snapshot, build_setup_state(1004, registry))

    decision = current_decision(state, registry)
    assert decision is not None
    first = checked_apply_action(state, decision.legal_actions[0], registry)
    assert first.state.turn_number == 2
    assert first.state.paid_actions_remaining == 2

    # Meld a known implemented card so a Dogma action is available regardless of the seed.
    active = first.state.active_player
    assert active is not None
    melded, _ = meld_card(first.state, active, CardId("the-wheel"), registry)
    decision = current_decision(melded, registry)
    assert decision is not None
    dogma = next(
        action
        for action in decision.legal_actions
        if isinstance(action, DogmaAction) and action.card_id == CardId("the-wheel")
    )
    resolved = checked_apply_action(melded, dogma, registry)
    # A dogma action now always resolves to a decision or a terminal result.
    assert resolved.decision is not None or resolved.terminal is not None
    assert not resolved.state.pending_effects


def test_progression_validator_rejects_a_transition_that_skips_turns() -> None:
    registry = load_card_registry()
    state = _finish_setup(build_setup_state(1005, registry), registry)
    decision = current_decision(state, registry)
    assert decision is not None
    action = decision.legal_actions[0]
    valid = apply_action(state, action, registry)
    corrupted_state = replace(valid.state, turn_number=state.turn_number + 2)
    corrupted = Transition(corrupted_state, decision=current_decision(corrupted_state, registry))

    from innovation_ai.innovation.invariants import assert_turn_progression

    with pytest.raises(InvariantViolation, match="skipped a turn"):
        assert_turn_progression(state, action, corrupted)


def test_hidden_supply_order_and_normal_achievement_identity_do_not_leak() -> None:
    registry = load_card_registry()
    state = build_setup_state(1006, registry)
    piles = list(state.supply.piles)
    piles[0] = tuple(reversed(piles[0]))
    reordered = replace(state, supply=replace(state.supply, piles=tuple(piles)))
    for viewer in PlayerId:
        assert_observation_leak_resistance(state, reordered, viewer, registry)

    achievement = state.normal_achievements.cards[0]
    replacement_card = state.supply.pile(1)[0]
    achievement_cards = list(state.normal_achievements.cards)
    achievement_cards[0] = replacement_card
    swapped_pile = (achievement, *state.supply.pile(1)[1:])
    piles = list(state.supply.piles)
    piles[0] = swapped_pile
    swapped = replace(
        state,
        supply=replace(state.supply, piles=tuple(piles)),
        normal_achievements=NormalAchievementState(tuple(achievement_cards)),
    )
    assert_state_properties(swapped, registry)
    for viewer in PlayerId:
        assert_observation_leak_resistance(state, swapped, viewer, registry)


def test_hidden_hand_score_and_unsplayed_covered_identities_do_not_leak() -> None:
    registry = load_card_registry()
    state = build_setup_state(1007, registry)
    hidden = state.player(PlayerId.PLAYER_2).hand[0]
    replacement_card = state.supply.pile(registry.card(hidden).age)[0]
    swapped_hand, _ = exchange_cards(
        state,
        CardLocation.hand(PlayerId.PLAYER_2),
        (hidden,),
        CardLocation.supply(registry.card(hidden).age),
        (replacement_card,),
        registry,
    )
    assert_observation_leak_resistance(state, swapped_hand, PlayerId.PLAYER_1, registry)

    scored, _ = score_card(state, PlayerId.PLAYER_2, hidden, registry)
    replacement_card = scored.supply.pile(registry.card(hidden).age)[0]
    swapped_score, _ = exchange_cards(
        scored,
        CardLocation.score(PlayerId.PLAYER_2),
        (hidden,),
        CardLocation.supply(registry.card(hidden).age),
        (replacement_card,),
        registry,
    )
    assert_observation_leak_resistance(scored, swapped_score, PlayerId.PLAYER_1, registry)

    covered = state
    first, second, top = _available_color_cards(covered, registry, Color.BLUE, 3)
    for card_id in (first, second, top):
        covered, _ = meld_card(covered, PlayerId.PLAYER_2, card_id, registry)
    rearranged, _ = rearrange_stack(
        covered, PlayerId.PLAYER_2, Color.BLUE, (second, first, top), registry
    )
    assert_observation_leak_resistance(
        covered,
        rearranged,
        PlayerId.PLAYER_1,
        registry,
        policy=InformationPolicy.RULEBOOK_PRIVATE_COVERED,
    )
    with pytest.raises(InvariantViolation, match="leaked"):
        assert_observation_leak_resistance(
            covered,
            rearranged,
            PlayerId.PLAYER_1,
            registry,
            policy=InformationPolicy.PUBLIC_COVERED,
        )


def test_terminal_state_rejects_every_current_public_mutation_entry_point() -> None:
    registry = load_card_registry()
    terminal = _terminal_by_exhaustion(build_setup_state(1008, registry), registry)
    assert terminal.phase is GamePhase.TERMINAL
    assert terminal.terminal_result == TerminalState(TerminalReason.DRAW_BEYOND_AGE_10, ())
    assert_state_properties(terminal, registry)
    assert_terminal_immutability(terminal, registry)


def test_terminal_exposes_no_action_and_a_paused_effect_exposes_exactly_one_decision() -> None:
    registry = load_card_registry()
    terminal = _terminal_by_exhaustion(build_setup_state(1009, registry), registry)
    assert_legal_action_completeness(terminal, registry)
    assert current_decisions(terminal, registry) == ()

    # A mid-dogma state must be decidable: a running engine that asks nothing is undecidable
    # for a runner, a fuzzer, and a replay alike.
    from innovation_ai.innovation.effects import load_effect_programs, start_dogma
    from innovation_ai.innovation.state import ExplicitPlayerPosition, build_explicit_state

    programs = load_effect_programs()
    paused = start_dogma(
        build_explicit_state(
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
        ),
        CardId("code-of-laws"),
        PlayerId.PLAYER_1,
        programs,
        registry,
    ).state
    assert paused.pending_effects
    decisions = current_decisions(paused, registry)
    assert len(decisions) == 1
    assert decisions[0].context is not None
    assert_legal_action_completeness(paused, registry)


def test_leak_validator_rejects_vacuous_equal_states() -> None:
    state = build_setup_state(1011)
    with pytest.raises(InvariantViolation, match="distinct authoritative"):
        assert_observation_leak_resistance(state, state, PlayerId.PLAYER_1)
