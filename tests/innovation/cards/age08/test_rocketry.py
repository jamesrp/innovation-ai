"""ROCKETRY clock quantity and two-stage hidden score-pile choices."""

from __future__ import annotations

from support import choose_card, choose_value, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction, ChooseValueAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_every_two_clocks_returns_one_score_with_hidden_ties_owned_by_the_opponent() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("rocketry",))
        .board(P1, Color.RED, ("flight",))
        .score(P2, ("canal-building", "construction", "tools"))
        .build()
    )
    result = resolve_dogma(
        state,
        "rocketry",
        choose_value(2),
        choose_card("construction"),
        choose_value(2),
        choose_card("construction"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert tuple(decision.chooser for decision in result.decisions) == (P1, P2, P1, P2)
    assert all(
        isinstance(action, ChooseValueAction) for action in result.decisions[0].legal_actions
    )
    assert {
        action.card_id
        for action in result.decisions[1].legal_actions
        if isinstance(action, ChooseCardAction)
    } == {CardId("canal-building"), CardId("construction")}
    assert result.decisions[2].observation.player(P2).score_pile.values == (1, 2, 2)
    assert result.state.player(P2).score_pile == (CardId("tools"),)
