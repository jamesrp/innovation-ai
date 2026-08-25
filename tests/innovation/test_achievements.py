from __future__ import annotations

from dataclasses import replace

import pytest
from achievement_fixtures import (
    ACTIVE,
    FIVE_NORMAL_ACHIEVEMENTS,
    FOUR_NORMAL_ACHIEVEMENTS,
    OPPONENT,
    UNIVERSE_TOPS,
    WONDER_STACKS,
    card_registry,
    empire_board,
    monument_counters,
    place,
    playable_state,
    universe_board,
    with_achievements,
    with_score,
    wonder_board,
    world_board,
)

from innovation_ai.innovation.achievements import (
    MASONRY_MELD_COUNT,
    MONUMENT_MOVEMENT_COUNT,
    SPECIAL_CHECK_ORDER,
    AchievementCheckResult,
    AchievementClaim,
    AchievementClaimError,
    ClaimRoute,
    LinkedRouteContext,
    MonumentCountKind,
    QualifyingMovement,
    astronomy_universe_route,
    automatic_predicate_satisfied,
    automatically_eligible_special_achievements,
    available_normal_achievements,
    available_special_achievements,
    check_after_change,
    check_atomic_boundary,
    check_order,
    claim_linked_route,
    claim_normal_achievement,
    claimed_normal_achievements,
    claimed_special_achievements,
    construction_empire_route,
    eligible_normal_achievements,
    empire_predicate,
    invention_wonder_route,
    linked_route,
    linked_route_satisfied,
    masonry_monument_route,
    monument_predicate,
    monument_progress,
    normal_achievement_age,
    normal_achievement_is_eligible,
    qualifying_monument_movements,
    record_qualifying_movements,
    translation_world_route,
    universe_predicate,
    wonder_predicate,
    world_predicate,
)
from innovation_ai.innovation.state import GamePhase, TerminalReason, TerminalResult
from innovation_ai.innovation.terminal import apply_terminal
from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)
from innovation_ai.innovation.zones import (
    CardLocation,
    CardMove,
    ChangeKind,
    ChangeRecord,
    exchange_cards,
    move_card,
    score_card,
    set_splay,
    tuck_card,
)

# ---------------------------------------------------------------------------------------------
# Normal achievements
# ---------------------------------------------------------------------------------------------


def test_normal_achievement_ages_and_availability_are_canonical() -> None:
    registry = card_registry()
    state = playable_state(registry)

    assert [normal_achievement_age(item) for item in NormalAchievementId] == list(range(1, 10))
    assert available_normal_achievements(state) == tuple(NormalAchievementId)
    assert available_special_achievements(state) == tuple(SpecialAchievementId)
    assert claimed_normal_achievements(state) == frozenset()
    assert claimed_special_achievements(state) == frozenset()

    owned = with_achievements(
        state,
        OPPONENT,
        normal=(NormalAchievementId.AGE_2,),
        special=(SpecialAchievementId.WORLD,),
    )
    assert NormalAchievementId.AGE_2 not in available_normal_achievements(owned)
    assert SpecialAchievementId.WORLD not in available_special_achievements(owned)
    assert claimed_normal_achievements(owned) == frozenset({NormalAchievementId.AGE_2})
    assert claimed_special_achievements(owned) == frozenset({SpecialAchievementId.WORLD})


@pytest.mark.parametrize(
    ("score_ages", "board", "achievement", "eligible"),
    [
        ((), (), NormalAchievementId.AGE_1, False),
        ((5,), ("tools",), NormalAchievementId.AGE_1, True),
        ((4,), ("tools",), NormalAchievementId.AGE_1, False),
        ((5,), (), NormalAchievementId.AGE_1, False),
        ((5, 5), ("mathematics",), NormalAchievementId.AGE_2, True),
        ((5, 4), ("tools",), NormalAchievementId.AGE_2, False),
        ((5, 5), ("pottery",), NormalAchievementId.AGE_2, False),
        ((10, 10, 10), ("quantum-theory",), NormalAchievementId.AGE_6, True),
        ((10, 10, 9), ("quantum-theory",), NormalAchievementId.AGE_6, False),
        ((10, 10, 10, 10, 5), ("quantum-theory",), NormalAchievementId.AGE_9, False),
        ((10, 10, 10, 10, 5), ("databases",), NormalAchievementId.AGE_9, True),
    ],
)
def test_normal_achievement_requires_score_and_top_card_value(
    score_ages: tuple[int, ...],
    board: tuple[str, ...],
    achievement: NormalAchievementId,
    eligible: bool,
) -> None:
    registry = card_registry()
    state = with_score(playable_state(registry), ACTIVE, score_ages, registry)
    if board:
        state = place(state, ACTIVE, board, registry)

    assert normal_achievement_is_eligible(state, ACTIVE, achievement, registry) is eligible
    assert (achievement in eligible_normal_achievements(state, ACTIVE, registry)) is eligible


def test_normal_achievement_is_unavailable_once_either_player_owns_it() -> None:
    registry = card_registry()
    state = with_score(playable_state(registry), ACTIVE, (5,), registry)
    state = place(state, ACTIVE, ("tools",), registry)
    assert normal_achievement_is_eligible(state, ACTIVE, NormalAchievementId.AGE_1, registry)

    owned = with_achievements(state, OPPONENT, normal=(NormalAchievementId.AGE_1,))
    assert not normal_achievement_is_eligible(owned, ACTIVE, NormalAchievementId.AGE_1, registry)
    with pytest.raises(AchievementClaimError, match="cannot claim"):
        claim_normal_achievement(owned, ACTIVE, NormalAchievementId.AGE_1, registry)


def test_claiming_a_normal_achievement_keeps_score_and_records_the_route() -> None:
    registry = card_registry()
    state = with_score(playable_state(registry), ACTIVE, (5,), registry)
    state = place(state, ACTIVE, ("tools",), registry)

    result = claim_normal_achievement(state, ACTIVE, NormalAchievementId.AGE_1, registry)
    assert result.terminal is None
    assert not result.game_over
    assert result.changed
    assert result.claims == (
        AchievementClaim(ACTIVE, NormalAchievementId.AGE_1, ClaimRoute.ACHIEVE_ACTION),
    )
    assert result.state.player(ACTIVE).normal_achievements == (NormalAchievementId.AGE_1,)
    assert result.state.player(ACTIVE).score_pile == state.player(ACTIVE).score_pile
    assert state.player(ACTIVE).normal_achievements == ()


def test_sixth_achievement_from_a_normal_claim_ends_the_game_immediately() -> None:
    registry = card_registry()
    state = with_score(playable_state(registry), ACTIVE, (5,), registry)
    state = place(state, ACTIVE, ("tools",), registry)
    state = with_achievements(
        state,
        ACTIVE,
        normal=(*FIVE_NORMAL_ACHIEVEMENTS[1:], NormalAchievementId.AGE_6),
    )

    result = claim_normal_achievement(state, ACTIVE, NormalAchievementId.AGE_1, registry)
    assert result.game_over
    assert result.terminal is not None
    assert result.terminal.reason is TerminalReason.ACHIEVEMENT_VICTORY
    assert result.terminal.winners == (ACTIVE,)
    assert result.state.phase is GamePhase.TERMINAL


def test_achievement_check_result_rejects_inconsistent_terminal_phase() -> None:
    registry = card_registry()
    state = playable_state(registry)
    with pytest.raises(ValueError, match="terminal result and state phase"):
        AchievementCheckResult(state, (), TerminalResult(TerminalReason.CARD_EFFECT, ()))


def test_achievement_claim_record_validates_linked_route_provenance() -> None:
    with pytest.raises(ValueError, match="must record its source effect"):
        AchievementClaim(ACTIVE, SpecialAchievementId.WORLD, ClaimRoute.LINKED_CARD)
    with pytest.raises(ValueError, match="only a linked-card claim"):
        AchievementClaim(
            ACTIVE,
            SpecialAchievementId.WORLD,
            ClaimRoute.AUTOMATIC,
            DogmaEffectId(CardId("translation"), 2),
        )


# ---------------------------------------------------------------------------------------------
# Automatic special-achievement predicates
# ---------------------------------------------------------------------------------------------


def test_empire_needs_three_visible_icons_of_every_type() -> None:
    registry = card_registry()
    satisfied = empire_board(playable_state(registry), ACTIVE, registry)
    assert empire_predicate(satisfied, ACTIVE, registry)
    assert not empire_predicate(satisfied, OPPONENT, registry)

    unsplayed, _ = set_splay(satisfied, ACTIVE, Color.PURPLE, SplayDirection.NONE, registry)
    assert not empire_predicate(unsplayed, ACTIVE, registry)


def test_world_needs_twelve_visible_clocks() -> None:
    registry = card_registry()
    satisfied = world_board(playable_state(registry), ACTIVE, registry)
    assert world_predicate(satisfied, ACTIVE, registry)

    collapsed, _ = set_splay(satisfied, ACTIVE, Color.GREEN, SplayDirection.NONE, registry)
    assert not world_predicate(collapsed, ACTIVE, registry)


def test_wonder_needs_five_colors_splayed_right_or_up() -> None:
    registry = card_registry()
    satisfied = wonder_board(playable_state(registry), ACTIVE, registry)
    assert wonder_predicate(satisfied, ACTIVE, registry)

    left, _ = set_splay(satisfied, ACTIVE, Color.BLUE, SplayDirection.LEFT, registry)
    assert not wonder_predicate(left, ACTIVE, registry)
    flat, _ = set_splay(satisfied, ACTIVE, Color.RED, SplayDirection.NONE, registry)
    assert not wonder_predicate(flat, ACTIVE, registry)

    missing_color = playable_state(registry)
    for names, splay in WONDER_STACKS[0][:4]:
        missing_color = place(missing_color, ACTIVE, names, registry, splay=splay)
    assert not wonder_predicate(missing_color, ACTIVE, registry)


def test_universe_needs_five_top_cards_of_value_eight_or_higher() -> None:
    registry = card_registry()
    satisfied = universe_board(playable_state(registry), ACTIVE, registry)
    assert universe_predicate(satisfied, ACTIVE, registry)

    covered = place(satisfied, ACTIVE, ("sanitation",), registry)
    assert not universe_predicate(covered, ACTIVE, registry)

    four_colors = playable_state(registry)
    for name in UNIVERSE_TOPS[0][:4]:
        four_colors = place(four_colors, ACTIVE, (name,), registry)
    assert not universe_predicate(four_colors, ACTIVE, registry)


def test_monument_counts_tucks_and_scores_separately() -> None:
    registry = card_registry()
    state = playable_state(registry)
    assert monument_progress(state, ACTIVE) == (0, 0)
    assert not monument_predicate(state, ACTIVE, registry)

    tucked = monument_counters(state, ACTIVE, MonumentCountKind.TUCK, MONUMENT_MOVEMENT_COUNT)
    assert monument_progress(tucked, ACTIVE) == (MONUMENT_MOVEMENT_COUNT, 0)
    assert monument_predicate(tucked, ACTIVE, registry)
    assert not monument_predicate(tucked, OPPONENT, registry)

    scored = monument_counters(state, ACTIVE, MonumentCountKind.SCORE, MONUMENT_MOVEMENT_COUNT)
    assert monument_predicate(scored, ACTIVE, registry)

    mixed = replace(state, turn_counters=state.turn_counters.increment(ACTIVE, tucked=3, scored=3))
    assert not monument_predicate(mixed, ACTIVE, registry)
    assert monument_progress(mixed, ACTIVE) == (3, 3)


@pytest.mark.parametrize("achievement", list(SPECIAL_CHECK_ORDER))
def test_every_special_achievement_has_a_predicate_and_a_linked_route(
    achievement: SpecialAchievementId,
) -> None:
    registry = card_registry()
    state = playable_state(registry)

    assert automatic_predicate_satisfied(state, ACTIVE, achievement, registry) in {True, False}
    assert linked_route_satisfied(state, ACTIVE, achievement, registry) in {True, False}
    assert linked_route(achievement, registry).achievement_id is achievement


def test_special_check_order_covers_all_five_achievements_monument_first() -> None:
    assert SPECIAL_CHECK_ORDER[0] is SpecialAchievementId.MONUMENT
    assert set(SPECIAL_CHECK_ORDER) == set(SpecialAchievementId)
    assert len(SPECIAL_CHECK_ORDER) == len(SpecialAchievementId)


def test_automatically_eligible_special_achievements_uses_check_order() -> None:
    registry = card_registry()
    state = universe_board(playable_state(registry), ACTIVE, registry)
    state = monument_counters(state, ACTIVE, MonumentCountKind.SCORE, MONUMENT_MOVEMENT_COUNT)

    assert automatically_eligible_special_achievements(state, ACTIVE, registry) == (
        SpecialAchievementId.MONUMENT,
        SpecialAchievementId.UNIVERSE,
    )
    owned = with_achievements(state, OPPONENT, special=(SpecialAchievementId.MONUMENT,))
    assert automatically_eligible_special_achievements(owned, ACTIVE, registry) == (
        SpecialAchievementId.UNIVERSE,
    )


# ---------------------------------------------------------------------------------------------
# Linked-card alternate routes
# ---------------------------------------------------------------------------------------------


def test_linked_routes_point_at_their_printed_card_effects() -> None:
    registry = card_registry()
    assert linked_route(SpecialAchievementId.MONUMENT, registry).source_effect_id == DogmaEffectId(
        CardId("masonry"), 1
    )
    assert linked_route(SpecialAchievementId.EMPIRE, registry).source_card_id == CardId(
        "construction"
    )
    assert linked_route(SpecialAchievementId.WORLD, registry).source_card_id == CardId(
        "translation"
    )
    assert linked_route(SpecialAchievementId.WONDER, registry).source_card_id == CardId("invention")
    assert linked_route(SpecialAchievementId.UNIVERSE, registry).source_card_id == CardId(
        "astronomy"
    )


@pytest.mark.parametrize(
    ("melded", "satisfied"),
    [(0, False), (3, False), (MASONRY_MELD_COUNT, True), (7, True)],
)
def test_masonry_route_counts_cards_melded_by_that_effect(melded: int, satisfied: bool) -> None:
    registry = card_registry()
    state = playable_state(registry)
    context = LinkedRouteContext(melded_card_count=melded)
    assert masonry_monument_route(state, ACTIVE, registry, context=context) is satisfied
    assert (
        linked_route_satisfied(
            state, ACTIVE, SpecialAchievementId.MONUMENT, registry, context=context
        )
        is satisfied
    )


def test_linked_route_context_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        LinkedRouteContext(melded_card_count=-1)


def test_construction_route_needs_the_only_five_top_card_board() -> None:
    registry = card_registry()
    state = universe_board(playable_state(registry), ACTIVE, registry)
    assert construction_empire_route(state, ACTIVE, registry)
    assert not construction_empire_route(state, OPPONENT, registry)

    both = universe_board(state, OPPONENT, registry, variant=1)
    assert not construction_empire_route(both, ACTIVE, registry)
    assert not construction_empire_route(both, OPPONENT, registry)

    # The linked route differs from the automatic predicate: five low top cards qualify here.
    low = playable_state(registry)
    for name in ("pottery", "the-wheel", "city-states", "archery", "masonry"):
        low = place(low, ACTIVE, (name,), registry)
    assert construction_empire_route(low, ACTIVE, registry)
    assert not empire_predicate(low, ACTIVE, registry)


def test_translation_route_needs_every_top_card_to_have_a_crown() -> None:
    registry = card_registry()
    state = playable_state(registry)
    crowned = state
    for name in ("writing", "sailing", "city-states", "oars", "domestication"):
        crowned = place(crowned, ACTIVE, (name,), registry)
    assert translation_world_route(crowned, ACTIVE, registry)
    assert not world_predicate(crowned, ACTIVE, registry)

    spoiled = place(crowned, ACTIVE, ("pottery",), registry)
    assert not translation_world_route(spoiled, ACTIVE, registry)

    # Decision 10: a universal predicate holds vacuously for an empty top-card set.
    assert translation_world_route(state, ACTIVE, registry)


def test_invention_route_accepts_any_splay_direction() -> None:
    registry = card_registry()
    state = playable_state(registry)
    splayed = state
    for names, _ in WONDER_STACKS[0]:
        splayed = place(splayed, ACTIVE, names, registry, splay=SplayDirection.LEFT)
    assert invention_wonder_route(splayed, ACTIVE, registry)
    assert not wonder_predicate(splayed, ACTIVE, registry)

    partial, _ = set_splay(splayed, ACTIVE, Color.YELLOW, SplayDirection.NONE, registry)
    assert not invention_wonder_route(partial, ACTIVE, registry)
    assert not invention_wonder_route(state, ACTIVE, registry)


def test_astronomy_route_ignores_purple_and_accepts_value_six() -> None:
    registry = card_registry()
    state = playable_state(registry)
    board = state
    for name in ("encyclopedia", "classification", "canning", "machine-tools", "mysticism"):
        board = place(board, ACTIVE, (name,), registry)
    assert astronomy_universe_route(board, ACTIVE, registry)
    assert not universe_predicate(board, ACTIVE, registry)

    spoiled = place(board, ACTIVE, ("pottery",), registry)
    assert not astronomy_universe_route(spoiled, ACTIVE, registry)

    # Decision 10: an empty non-purple top-card set satisfies the condition.
    purple_only = place(state, ACTIVE, ("mysticism",), registry)
    assert astronomy_universe_route(purple_only, ACTIVE, registry)
    assert astronomy_universe_route(state, ACTIVE, registry)


def test_claim_linked_route_awards_records_provenance_and_is_idempotent() -> None:
    registry = card_registry()
    state = universe_board(playable_state(registry), ACTIVE, registry)

    result = claim_linked_route(state, ACTIVE, SpecialAchievementId.EMPIRE, registry)
    assert SpecialAchievementId.EMPIRE in result.state.player(ACTIVE).special_achievements
    linked = next(claim for claim in result.claims if claim.route is ClaimRoute.LINKED_CARD)
    assert linked.achievement_id is SpecialAchievementId.EMPIRE
    assert linked.source_effect_id == DogmaEffectId(CardId("construction"), 2)
    # The following boundary check also claimed the automatically eligible Universe achievement.
    assert SpecialAchievementId.UNIVERSE in result.state.player(ACTIVE).special_achievements

    again = claim_linked_route(result.state, ACTIVE, SpecialAchievementId.EMPIRE, registry)
    assert again.state == result.state
    assert again.claims == ()


def test_claim_linked_route_does_nothing_when_the_route_condition_fails() -> None:
    registry = card_registry()
    state = playable_state(registry)
    result = claim_linked_route(state, ACTIVE, SpecialAchievementId.MONUMENT, registry)
    assert result.state is state
    assert result.claims == ()
    assert not result.changed


def test_claim_linked_route_can_skip_the_following_boundary_check() -> None:
    registry = card_registry()
    state = universe_board(playable_state(registry), ACTIVE, registry)
    result = claim_linked_route(
        state, ACTIVE, SpecialAchievementId.EMPIRE, registry, check_boundary=False
    )
    assert result.state.player(ACTIVE).special_achievements == (SpecialAchievementId.EMPIRE,)
    assert len(result.claims) == 1


def test_linked_route_can_produce_the_sixth_achievement_victory() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("writing",), registry)
    state = with_achievements(state, ACTIVE, normal=FIVE_NORMAL_ACHIEVEMENTS)

    result = claim_linked_route(state, ACTIVE, SpecialAchievementId.WORLD, registry)
    assert result.game_over
    assert result.terminal is not None
    assert result.terminal.reason is TerminalReason.ACHIEVEMENT_VICTORY
    assert result.terminal.winners == (ACTIVE,)


# ---------------------------------------------------------------------------------------------
# Monument counters and provenance exclusions
# ---------------------------------------------------------------------------------------------


def test_tuck_and_score_primitives_count_toward_monument() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("pottery",), registry)
    hand_card = state.supply.pile(1)[0]
    state, _ = move_card(state, hand_card, CardLocation.hand(ACTIVE), registry)

    tucked, change = tuck_card(state, ACTIVE, hand_card, registry)
    assert monument_progress(tucked, ACTIVE) == (1, 0)
    assert qualifying_monument_movements(change) == (
        QualifyingMovement(ACTIVE, MonumentCountKind.TUCK, hand_card),
    )

    scored_card = tucked.supply.pile(2)[0]
    scored, score_change = score_card(tucked, ACTIVE, scored_card, registry)
    assert monument_progress(scored, ACTIVE) == (1, 1)
    assert qualifying_monument_movements(score_change) == (
        QualifyingMovement(ACTIVE, MonumentCountKind.SCORE, scored_card),
    )


def test_transfers_and_exchanges_into_a_score_pile_never_count_for_monument() -> None:
    registry = card_registry()
    state = playable_state(registry)
    donor_card = state.supply.pile(3)[0]
    state, _ = move_card(state, donor_card, CardLocation.score(OPPONENT), registry)

    transferred, change = move_card(
        state, donor_card, CardLocation.score(ACTIVE), registry, kind=ChangeKind.TRANSFER
    )
    assert change.kind is ChangeKind.TRANSFER
    assert qualifying_monument_movements(change) == ()
    assert monument_progress(transferred, ACTIVE) == (0, 0)
    counted = record_qualifying_movements(transferred, qualifying_monument_movements(change))
    assert monument_progress(counted, ACTIVE) == (0, 0)

    hand_card = transferred.supply.pile(4)[0]
    transferred, _ = move_card(transferred, hand_card, CardLocation.hand(ACTIVE), registry)
    exchanged, exchange_change = exchange_cards(
        transferred,
        CardLocation.hand(ACTIVE),
        (hand_card,),
        CardLocation.score(ACTIVE),
        (donor_card,),
        registry,
    )
    assert exchange_change.kind is ChangeKind.EXCHANGE
    assert hand_card in exchanged.player(ACTIVE).score_pile
    assert qualifying_monument_movements(exchange_change) == ()
    assert monument_progress(exchanged, ACTIVE) == (0, 0)


def test_record_qualifying_movements_folds_bulk_atoms_and_ignores_setup() -> None:
    registry = card_registry()
    state = playable_state(registry)
    movements = (
        QualifyingMovement(ACTIVE, MonumentCountKind.TUCK),
        QualifyingMovement(ACTIVE, MonumentCountKind.TUCK),
        QualifyingMovement(OPPONENT, MonumentCountKind.SCORE),
    )
    counted = record_qualifying_movements(state, movements)
    assert monument_progress(counted, ACTIVE) == (2, 0)
    assert monument_progress(counted, OPPONENT) == (0, 1)
    assert record_qualifying_movements(state, ()) is state

    setup = replace(state, phase=GamePhase.STARTING_MELDS, active_player=None, turn_number=0)
    assert record_qualifying_movements(setup, movements) is setup


def test_qualifying_monument_movements_filters_by_change_kind_and_destination() -> None:
    registry = card_registry()
    state = playable_state(registry)
    card_id = state.supply.pile(1)[0]
    mixed = ChangeRecord(
        ChangeKind.TUCK,
        (
            CardMove(card_id, CardLocation.hand(ACTIVE), CardLocation.supply(1)),
            CardMove(card_id, CardLocation.hand(ACTIVE), CardLocation.board(ACTIVE, Color.BLUE)),
        ),
    )
    assert qualifying_monument_movements(mixed) == (
        QualifyingMovement(ACTIVE, MonumentCountKind.TUCK, card_id),
    )
    assert qualifying_monument_movements(ChangeRecord(ChangeKind.SPLAY)) == ()
    assert qualifying_monument_movements(ChangeRecord(ChangeKind.DRAW)) == ()


def test_check_after_change_counts_bulk_movements_only_when_requested() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("pottery",), registry)
    hand_card = state.supply.pile(1)[0]
    state, _ = move_card(state, hand_card, CardLocation.hand(ACTIVE), registry)
    tucked, change = tuck_card(state, ACTIVE, hand_card, registry)

    default = check_after_change(tucked, change, registry)
    assert monument_progress(default.state, ACTIVE) == (1, 0)

    doubled = check_after_change(tucked, change, registry, count_monument_movements=True)
    assert monument_progress(doubled.state, ACTIVE) == (2, 0)
    assert check_after_change(tucked, None, registry).state == tucked


def test_check_after_change_claims_monument_for_a_bulk_score_atom() -> None:
    registry = card_registry()
    state = playable_state(registry)
    bulk = ChangeRecord(
        ChangeKind.SCORE,
        tuple(
            CardMove(
                state.supply.pile(1)[index],
                CardLocation.hand(ACTIVE),
                CardLocation.score(ACTIVE),
            )
            for index in range(MONUMENT_MOVEMENT_COUNT)
        ),
    )
    assert len(qualifying_monument_movements(bulk)) == MONUMENT_MOVEMENT_COUNT

    result = check_after_change(state, bulk, registry, count_monument_movements=True)
    assert SpecialAchievementId.MONUMENT in result.state.player(ACTIVE).special_achievements
    assert result.claims[0].route is ClaimRoute.AUTOMATIC


# ---------------------------------------------------------------------------------------------
# Atomic boundary ordering
# ---------------------------------------------------------------------------------------------


def test_check_order_puts_the_active_player_first_then_the_documented_sequence() -> None:
    registry = card_registry()
    state = playable_state(registry)

    order = check_order(state)
    assert order[: len(SPECIAL_CHECK_ORDER)] == tuple(
        (ACTIVE, achievement) for achievement in SPECIAL_CHECK_ORDER
    )
    assert order[len(SPECIAL_CHECK_ORDER) :] == tuple(
        (OPPONENT, achievement) for achievement in SPECIAL_CHECK_ORDER
    )
    assert check_order(state, OPPONENT)[0] == (OPPONENT, SpecialAchievementId.MONUMENT)

    setup = replace(state, phase=GamePhase.STARTING_MELDS, active_player=None, turn_number=0)
    assert check_order(setup)[0] == (ACTIVE, SpecialAchievementId.MONUMENT)


def test_boundary_check_claims_every_eligible_special_achievement_in_order() -> None:
    registry = card_registry()
    state = universe_board(playable_state(registry), ACTIVE, registry)
    state = monument_counters(state, ACTIVE, MonumentCountKind.TUCK, MONUMENT_MOVEMENT_COUNT)
    state = world_board(state, OPPONENT, registry)

    result = check_atomic_boundary(state, registry)
    assert not result.game_over
    assert tuple((claim.player_id, claim.achievement_id) for claim in result.claims) == (
        (ACTIVE, SpecialAchievementId.MONUMENT),
        (ACTIVE, SpecialAchievementId.UNIVERSE),
        (OPPONENT, SpecialAchievementId.WORLD),
    )
    assert all(claim.route is ClaimRoute.AUTOMATIC for claim in result.claims)
    assert result.state.player(ACTIVE).special_achievements == (
        SpecialAchievementId.MONUMENT,
        SpecialAchievementId.UNIVERSE,
    )
    assert result.state.player(OPPONENT).special_achievements == (SpecialAchievementId.WORLD,)


@pytest.mark.parametrize("active", list(PlayerId))
def test_same_special_simultaneous_eligibility_goes_to_the_active_player(
    active: PlayerId,
) -> None:
    registry = card_registry()
    state = playable_state(registry, active=active)
    state = wonder_board(state, ACTIVE, registry)
    state = wonder_board(state, OPPONENT, registry, variant=1)
    assert wonder_predicate(state, ACTIVE, registry)
    assert wonder_predicate(state, OPPONENT, registry)

    result = check_atomic_boundary(state, registry)
    assert len(result.claims) == 1
    assert result.claims[0].player_id is active
    assert result.claims[0].achievement_id is SpecialAchievementId.WONDER
    other = OPPONENT if active is ACTIVE else ACTIVE
    assert result.state.player(other).special_achievements == ()


@pytest.mark.parametrize(
    ("active", "expected"),
    [
        (ACTIVE, SpecialAchievementId.UNIVERSE),
        (OPPONENT, SpecialAchievementId.EMPIRE),
    ],
)
def test_different_sixth_special_achievements_use_active_player_priority(
    active: PlayerId, expected: SpecialAchievementId
) -> None:
    registry = card_registry()
    state = playable_state(registry, active=active)
    state = universe_board(state, ACTIVE, registry)
    state = empire_board(state, OPPONENT, registry)
    state = with_achievements(state, ACTIVE, normal=FIVE_NORMAL_ACHIEVEMENTS)
    state = with_achievements(
        state,
        OPPONENT,
        normal=FOUR_NORMAL_ACHIEVEMENTS,
        special=(SpecialAchievementId.MONUMENT,),
    )

    result = check_atomic_boundary(state, registry)
    assert result.game_over
    assert result.terminal is not None
    assert result.terminal.reason is TerminalReason.ACHIEVEMENT_VICTORY
    assert result.terminal.winners == (active,)
    assert len(result.claims) == 1
    assert result.claims[0].achievement_id is expected

    # The immediate win stops every remaining check, so the loser claims nothing.
    loser = OPPONENT if active is ACTIVE else ACTIVE
    unclaimed = SpecialAchievementId.EMPIRE if active is ACTIVE else SpecialAchievementId.UNIVERSE
    assert unclaimed not in result.state.player(loser).special_achievements


def test_boundary_check_is_idempotent_and_passes_through_terminal_states() -> None:
    registry = card_registry()
    state = universe_board(playable_state(registry), ACTIVE, registry)

    first = check_atomic_boundary(state, registry)
    second = check_atomic_boundary(first.state, registry)
    assert second.state == first.state
    assert second.claims == ()

    ended = apply_terminal(first.state, TerminalResult(TerminalReason.CARD_EFFECT, (ACTIVE,)))
    passthrough = check_atomic_boundary(ended, registry, previous_claims=first.claims)
    assert passthrough.game_over
    assert passthrough.claims == first.claims
    assert passthrough.state is ended


def test_boundary_check_uses_live_state_not_frozen_dogma_counts() -> None:
    registry = card_registry()
    state = playable_state(registry)
    before = check_atomic_boundary(state, registry)
    assert before.claims == ()

    grown = world_board(before.state, ACTIVE, registry)
    after = check_atomic_boundary(grown, registry)
    assert after.claims[0].achievement_id is SpecialAchievementId.WORLD
