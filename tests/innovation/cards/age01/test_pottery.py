"""POTTERY: canonical bounded multi-select, return-order decisions, and two effects."""

from __future__ import annotations

from support import ScenarioBuilder, assert_conserved, choose_card, finish, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    """Return a position where the opponent has no leaves and therefore does not share."""

    return (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("pottery",))
        .board(P2, Color.RED, ("metalworking",))
    )


def test_an_empty_hand_raises_no_decision_and_effect_two_still_draws() -> None:
    state = _solo().supply(1, ("agriculture",)).build()
    result = resolve_dogma(state, "pottery", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert result.state.player(P1).hand == (CardId("agriculture"),)
    assert not result.state.player(P1).score_pile


def test_declining_the_return_still_runs_effect_two() -> None:
    state = _solo().hand(P1, ("tools",)).supply(1, ("agriculture",)).build()
    result = resolve_dogma(state, "pottery", finish(), registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert not result.state.player(P1).score_pile
    assert set(result.state.player(P1).hand) == {CardId("tools"), CardId("agriculture")}


def test_returning_two_cards_draws_and_scores_a_value_two_card() -> None:
    state = (
        _solo()
        .hand(P1, ("tools", "writing", "sailing"))
        .supply(2, ("canal-building",))
        .supply(1, ("agriculture",))
        .build()
    )
    result = resolve_dogma(
        state,
        "pottery",
        choose_card("sailing"),
        choose_card("tools"),
        finish(),
        # Both returned cards are age 1, so their order inside that pile is a real decision.
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).score_pile == (CardId("canal-building"),)
    assert_conserved(result.state, REGISTRY)


def test_incremental_subset_selection_is_canonical_by_card_id() -> None:
    """Decision 16: each successive pick must strictly exceed the previous card ID."""

    state = _solo().hand(P1, ("sailing", "tools", "writing")).build()
    result = resolve_dogma(
        state,
        "pottery",
        choose_card("sailing"),
        choose_card("tools"),
        finish(),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    first, second = result.decisions[0], result.decisions[1]
    assert {action.card_id for action in first.legal_actions if hasattr(action, "card_id")} == {
        CardId("sailing"),
        CardId("tools"),
        CardId("writing"),
    }
    # After picking "sailing", only strictly greater IDs remain, so {sailing, tools} has one path.
    assert {action.card_id for action in second.legal_actions if hasattr(action, "card_id")} == {
        CardId("tools"),
        CardId("writing"),
    }
    assert second.context is not None
    assert second.context.selected_so_far == (CardId("sailing"),)
    assert second.context.maximum_count == 3


def test_the_upper_bound_of_three_cards_is_enforced() -> None:
    state = (
        _solo()
        .hand(P1, ("sailing", "tools", "writing", "clothing"))
        .supply(3, ("alchemy",))
        .build()
    )
    result = resolve_dogma(
        state,
        "pottery",
        choose_card("clothing"),
        choose_card("sailing"),
        choose_card("tools"),
        # Three same-age returns need two order picks; the last card is forced.
        choose_card("tools"),
        choose_card("sailing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    # A fourth selection is never offered, so the selection closes automatically at three.
    # The hand keeps the unselected card plus effect 2's own age 1 draw.
    assert CardId("writing") in result.state.player(P1).hand
    assert len(result.state.player(P1).hand) == 2
    assert result.state.player(P1).score_pile == (CardId("alchemy"),)


def test_same_age_returns_raise_an_order_decision_the_owner_makes() -> None:
    """Decisions 5 and 16: order matters only inside one age pile, and the owner chooses."""

    state = _solo().hand(P1, ("sailing", "tools")).supply(2, ("canal-building",)).build()
    result = resolve_dogma(
        state,
        "pottery",
        choose_card("sailing"),
        choose_card("tools"),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    order_decision = result.decisions[-1]
    assert order_decision.chooser is P1
    # Incremental ordering: one decision per remaining card, never a permutation enumeration.
    assert len(order_decision.legal_actions) == 2
    # Tools was placed first, so it sits deeper in the age 1 pile than sailing.
    pile = result.state.supply.pile(1)
    assert pile.index(CardId("tools")) < pile.index(CardId("sailing"))


def test_mixed_age_returns_need_no_order_decision() -> None:
    """Cards going to different age piles cannot be distinguished by order."""

    state = _solo().hand(P1, ("tools", "canal-building")).supply(2, ("construction",)).build()
    result = resolve_dogma(
        state,
        "pottery",
        choose_card("canal-building"),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert len(result.decisions) == 2, "selection decisions only, no ordering decision"
    assert result.state.player(P1).score_pile == (CardId("construction"),)


def test_the_reward_quantity_is_snapshotted_before_the_returns_happen() -> None:
    """Decision 17: the instruction's quantity does not change while it executes."""

    state = _solo().hand(P1, ("sailing", "tools", "writing")).supply(3, ("alchemy",)).build()
    result = resolve_dogma(
        state,
        "pottery",
        choose_card("sailing"),
        choose_card("tools"),
        choose_card("writing"),
        choose_card("sailing"),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    # Three cards returned means exactly a value 3 reward, even though the hand is now empty.
    assert result.state.player(P1).score_pile == (CardId("alchemy"),)


def test_the_opponent_shares_and_earns_the_activator_one_free_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("pottery",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .hand(P2, ("tools",))
        .supply(1, ("clothing", "domestication", "masonry"))
        .build()
    )
    result = resolve_dogma(
        state,
        "pottery",
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert tuple(decision.chooser for decision in result.decisions) == (P2,)
    assert len(result.state.player(P2).score_pile) == 1
    # Effect 2's own draw plus one sharing-bonus draw.
    assert len(result.state.player(P1).hand) == 2
