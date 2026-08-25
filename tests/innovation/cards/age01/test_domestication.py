"""DOMESTICATION: empty-hand partial execution, lowest ties, sharing, and terminal draw."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("domestication",))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_an_empty_hand_skips_the_meld_but_still_draws() -> None:
    state = _solo().supply(1, ("agriculture",)).build()
    result = resolve_dogma(state, "domestication", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).hand == (CardId("agriculture"),)


def test_a_unique_lowest_card_is_melded_without_a_choice() -> None:
    state = _solo().hand(P1, ("canal-building", "alchemy")).supply(1, ("agriculture",)).build()
    result = resolve_dogma(state, "domestication", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).board.stack(Color.YELLOW).top == CardId("canal-building")
    assert set(result.state.player(P1).hand) == {CardId("alchemy"), CardId("agriculture")}


def test_the_owner_chooses_among_tied_lowest_cards() -> None:
    state = (
        _solo()
        .hand(P1, ("agriculture", "writing", "canal-building"))
        .supply(1, ("clothing",))
        .build()
    )
    result = resolve_dogma(
        state,
        "domestication",
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    }
    assert offered == {CardId("agriculture"), CardId("writing")}
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("writing")


def test_a_shared_execution_draws_for_both_then_awards_one_bonus() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("domestication",))
        .board(P2, Color.YELLOW, ("masonry",))
        .supply(1, ("agriculture", "clothing", "writing"))
        .build()
    )
    result = resolve_dogma(state, "domestication", registry=REGISTRY, programs=PROGRAMS)
    assert len(result.state.player(P2).hand) == 1
    assert len(result.state.player(P1).hand) == 2


def test_an_impossible_draw_ends_the_dogma_after_the_empty_hand_meld() -> None:
    state = _solo().exhaust_supply(into=P2).build()
    result = resolve_dogma(state, "domestication", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.TERMINAL
    assert result.terminal is not None
