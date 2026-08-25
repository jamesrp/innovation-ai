"""COMPOSITES private-zone choices and partial execution."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _vulnerable() -> ScenarioBuilder:
    return scenario(REGISTRY).board(P1, Color.RED, ("composites",))


def test_victim_keeps_one_hand_card_and_breaks_a_highest_score_tie() -> None:
    state = (
        _vulnerable()
        .hand(P2, ("tools", "writing", "sailing"))
        .score(P2, ("calendar", "canal-building", "mathematics"))
        .build()
    )
    result = resolve_dogma(
        state,
        "composites",
        choose_card("tools"),
        choose_card("mathematics"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert [decision.chooser for decision in result.decisions] == [P2, P2]
    assert result.state.player(P2).hand == (CardId("tools"),)
    assert set(result.state.player(P1).hand) == {CardId("sailing"), CardId("writing")}
    assert result.state.player(P1).score_pile == (CardId("mathematics"),)
    assert set(result.state.player(P2).score_pile) == {
        CardId("calendar"),
        CardId("canal-building"),
    }


def test_empty_private_zones_partially_execute_without_a_decision() -> None:
    result = resolve_dogma(
        _vulnerable().build(),
        "composites",
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert not result.decisions
    assert not result.state.player(P1).hand
    assert not result.state.player(P1).score_pile
