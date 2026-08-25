"""OPTICS: atomic draw-and-X branches and the poorer-opponent score transfer."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.innovation.zones import ChangeKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return scenario(REGISTRY).board(P1, Color.RED, ("optics",)).board(P2, Color.BLUE, ("pottery",))


def test_a_crown_three_is_melded_then_a_four_is_drawn_and_scored() -> None:
    state = _solo().supply(3, ("translation",)).supply(4, ("enterprise",)).build()
    result = resolve_dogma(state, "optics", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("translation")
    assert result.state.player(P1).score_pile == (CardId("enterprise"),)
    changes = tuple(event for event in result.events if event.change is not None)
    # Each Draw-and-X pair is one batch and therefore shares an atomic group ID.
    assert changes[0].change is not None and changes[0].change.kind is ChangeKind.DRAW
    assert changes[1].change is not None and changes[1].change.kind is ChangeKind.MELD
    assert changes[0].atomic_group_id == changes[1].atomic_group_id
    assert changes[2].change is not None and changes[2].change.kind is ChangeKind.DRAW
    assert changes[3].change is not None and changes[3].change.kind is ChangeKind.SCORE
    assert changes[2].atomic_group_id == changes[3].atomic_group_id


def test_a_non_crown_three_transfers_a_chosen_score_card_to_the_poorer_opponent() -> None:
    state = (
        _solo().score(P1, ("machinery",)).score(P2, ("tools",)).supply(3, ("education",)).build()
    )
    result = resolve_dogma(
        state,
        "optics",
        choose_card("machinery"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.PURPLE).top == CardId("education")
    assert result.state.player(P1).score_pile == ()
    assert set(result.state.player(P2).score_pile) == {CardId("tools"), CardId("machinery")}


def test_the_transfer_branch_does_nothing_when_the_opponent_is_not_poorer() -> None:
    state = (
        _solo().score(P1, ("tools",)).score(P2, ("machinery",)).supply(3, ("education",)).build()
    )
    result = resolve_dogma(state, "optics", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.state.player(P1).score_pile == (CardId("tools"),)
