"""THE WHEEL: trivial draw, sharing bonus, and a legal no-op dogma."""

from __future__ import annotations

from support import assert_conserved, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_ordinary_execution_draws_exactly_two_age_one_cards() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(1, ("agriculture", "clothing"))
        .build()
    )
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).hand == (CardId("agriculture"), CardId("clothing"))
    assert result.decisions == (), "the-wheel raises no choice"
    assert_conserved(result.state, REGISTRY)


def test_no_choice_is_ever_raised_so_the_card_is_fully_deterministic() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.RED, ("metalworking",))
        .build()
    )
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()


def test_upward_fallback_applies_when_the_age_one_supply_is_empty() -> None:
    """Rule 5: never fall back to a lower age; search upward only."""

    age_one = tuple(
        sorted(
            (
                card.id
                for card in REGISTRY.cards
                if card.age == 1 and card.id not in {CardId("the-wheel"), CardId("pottery")}
            ),
            key=str,
        )
    )
    # One age 1 card is still needed for that age's hidden normal achievement.
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.BLUE, ("pottery",))
        .score(P2, age_one[1:])
        .supply(2, ("canal-building", "construction"))
        .build()
    )
    assert not state.supply.pile(1)
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("canal-building"), CardId("construction"))


def test_draw_beyond_age_ten_ends_the_game_inside_the_dogma_action() -> None:
    """Rule 12 and decision 1: an impossible draw ends the game immediately."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.BLUE, ("pottery",))
        .exhaust_supply(into=P2)
        .build()
    )
    assert all(not state.supply.pile(age) for age in range(1, 11))
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
    assert not result.state.pending_effects
    assert not result.state.revealed


def test_sharing_grants_one_free_draw_and_only_one() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.YELLOW, ("masonry",))
        .build()
    )
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    # Two own draws plus exactly one bonus draw; the opponent gets their own two.
    assert len(result.state.player(P1).hand) == 3
    assert len(result.state.player(P2).hand) == 2


def test_a_stronger_opponent_shares_because_at_least_as_many_shares() -> None:
    # A right-splayed red stack exposes extra castles, so the opponent strictly exceeds three.
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.RED, ("metalworking", "oars"), splay=SplayDirection.RIGHT)
        .build()
    )
    from innovation_ai.innovation.board import visible_icons
    from innovation_ai.innovation.types import Icon

    assert visible_icons(state.player(P2).board, REGISTRY)[Icon.CASTLE] > 3
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    assert len(result.state.player(P2).hand) == 2
    assert len(result.state.player(P1).hand) == 3
