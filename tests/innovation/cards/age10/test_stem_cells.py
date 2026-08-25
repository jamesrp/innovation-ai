"""STEM CELLS optional all-or-none scoring of the complete hand."""

from __future__ import annotations

from support import choose_branch, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import AllOrNoneNode, EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_accepting_scores_every_hand_card_without_a_subset_choice() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("stem-cells",))
        .hand(P1, ("tools", "writing"))
        .build()
    )
    result = resolve_dogma(
        state,
        "stem-cells",
        choose_branch("score-all"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert not result.state.player(P1).hand
    assert set(result.state.player(P1).score_pile) == {CardId("tools"), CardId("writing")}
    assert len(result.decisions) == 1
    assert any(
        isinstance(node, AllOrNoneNode)
        for node in PROGRAMS.program_for_card(CardId("stem-cells")).nodes
    )


def test_declining_scores_none_of_the_hand() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.YELLOW, ("stem-cells",))
        .hand(P1, ("tools", "writing"))
        .build()
    )
    result = resolve_dogma(
        state,
        "stem-cells",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).hand) == {CardId("tools"), CardId("writing")}
    assert not result.state.player(P1).score_pile
