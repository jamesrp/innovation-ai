"""COMBUSTION: crown-scaled hidden score transfers and bottom-red return."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_every_four_activator_crowns_demands_one_victim_owned_score_choice() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("combustion",))
        .board(P1, Color.PURPLE, ("enterprise",))
        .board(P1, Color.BLUE, ("translation",))
        .board(P2, Color.RED, ("metalworking",))
        .score(P2, ("canal-building", "construction", "currency"))
        .build()
    )
    result = resolve_dogma(
        state,
        "combustion",
        choose_card("construction"),
        choose_card("currency"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2)
    assert all(
        decision.observation.player(P2).score_pile.known_cards
        == (CardId("canal-building"), CardId("construction"), CardId("currency"))
        for decision in result.decisions
    )
    transfers = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind.value == "transfer"
    )
    assert len(transfers) == 1
    assert set(transfers[0].card_ids) == {CardId("construction"), CardId("currency")}
    assert set(result.state.player(P1).score_pile) == {
        CardId("construction"),
        CardId("currency"),
    }
    assert result.state.player(P2).score_pile == (CardId("canal-building"),)
    # The second printed effect returns the activator's deterministic bottom red card.
    assert not result.state.player(P1).board.stack(Color.RED).cards


def test_mandatory_partial_execution_stops_when_the_score_pile_is_empty() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("combustion",))
        .board(P1, Color.PURPLE, ("enterprise",))
        .board(P1, Color.BLUE, ("translation",))
        .board(P2, Color.RED, ("metalworking",))
        .score(P2, ("canal-building",))
        .build()
    )
    result = resolve_dogma(state, "combustion", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).score_pile == (CardId("canal-building"),)
    assert not result.state.player(P2).score_pile
