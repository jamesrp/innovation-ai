"""CANAL BUILDING: optional atomic all-highest exchange, empty sides, and sharing order."""

from __future__ import annotations

from support import ScenarioBuilder, choose_branch, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.innovation.zones import ChangeKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("canal-building",))
        .board(P2, Color.RED, ("archery",))
    )


def test_empty_hand_and_score_pile_need_no_decision() -> None:
    result = resolve_dogma(_solo().build(), "canal-building", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.qualifying_changes == 0


def test_the_exchange_can_be_declined() -> None:
    state = _solo().hand(P1, ("tools",)).score(P1, ("alchemy",)).build()
    result = resolve_dogma(state, "canal-building", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert result.state.player(P1).score_pile == (CardId("alchemy"),)


def test_all_cards_tied_for_each_highest_value_exchange_atomically() -> None:
    state = (
        _solo()
        .hand(P1, ("tools", "construction", "currency"))
        .score(P1, ("agriculture", "alchemy", "compass"))
        .build()
    )
    result = resolve_dogma(
        state,
        "canal-building",
        choose_branch("exchange-highest"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).hand) == {
        CardId("tools"),
        CardId("alchemy"),
        CardId("compass"),
    }
    assert set(result.state.player(P1).score_pile) == {
        CardId("agriculture"),
        CardId("construction"),
        CardId("currency"),
    }
    exchanges = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind is ChangeKind.EXCHANGE
    )
    assert len(exchanges) == 1
    assert len(exchanges[0].change.card_moves) == 4  # type: ignore[union-attr]


def test_an_empty_side_still_exchanges_the_other_sides_highest_cards() -> None:
    state = _solo().score(P1, ("tools", "writing")).build()
    result = resolve_dogma(
        state,
        "canal-building",
        choose_branch("exchange-highest"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert set(result.state.player(P1).hand) == {CardId("tools"), CardId("writing")}
    assert not result.state.player(P1).score_pile


def test_shared_optional_choices_are_opponent_first() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("canal-building",))
        .board(P2, Color.GREEN, ("mapmaking",))
        .hand(P1, ("tools",))
        .hand(P2, ("writing",))
        .build()
    )
    result = resolve_dogma(
        state,
        "canal-building",
        decline(),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    assert result.qualifying_changes == 0
