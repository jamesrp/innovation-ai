"""MEDICINE: each hidden tied score choice belongs to that score pile's owner."""

from __future__ import annotations

from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.innovation.zones import ChangeKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_each_zone_owner_disambiguates_their_own_hidden_tie() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("medicine",))
        .board(P2, Color.PURPLE, ("education",))
        .score(P1, ("tools", "writing", "compass"))
        .score(P2, ("machinery", "optics", "agriculture"))
        .build()
    )
    result = resolve_dogma(
        state,
        "medicine",
        choose_card("optics"),
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P1)
    first_offered = {
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    }
    second_offered = {
        action.card_id for action in result.decisions[1].legal_actions if hasattr(action, "card_id")
    }
    assert first_offered == {CardId("machinery"), CardId("optics")}
    assert second_offered == {CardId("tools"), CardId("writing")}
    assert set(result.state.player(P1).score_pile) == {
        CardId("compass"),
        CardId("optics"),
        CardId("tools"),
    }
    assert set(result.state.player(P2).score_pile) == {
        CardId("agriculture"),
        CardId("machinery"),
        CardId("writing"),
    }


def test_the_victims_observation_does_not_reveal_the_activators_tied_identities() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("medicine",))
        .board(P2, Color.PURPLE, ("education",))
        .score(P1, ("tools", "writing"))
        .score(P2, ("machinery", "optics"))
        .build()
    )
    result = resolve_dogma(
        state,
        "medicine",
        choose_card("machinery"),
        choose_card("tools"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    victim_view = result.decisions[0].observation.player(P1).score_pile
    assert victim_view.known_cards == ()
    exchange = next(
        event
        for event in result.events
        if event.change is not None and event.change.kind is ChangeKind.EXCHANGE
    )
    assert set(exchange.card_ids) == {CardId("machinery"), CardId("tools")}


def test_an_empty_side_still_exchanges_the_single_available_extreme() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("medicine",))
        .board(P2, Color.PURPLE, ("education",))
        .score(P1, ("tools",))
        .build()
    )
    result = resolve_dogma(state, "medicine", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).score_pile == ()
    assert result.state.player(P2).score_pile == (CardId("tools"),)
