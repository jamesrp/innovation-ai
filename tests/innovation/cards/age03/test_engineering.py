"""ENGINEERING: all castle tops transfer atomically and red splaying remains optional."""

from __future__ import annotations

from support import choose_branch, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection
from innovation_ai.innovation.zones import ChangeKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_all_qualifying_top_cards_transfer_to_the_activators_score_in_one_atom() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("engineering",))
        .board(P1, Color.BLUE, ("alchemy",))
        .board(P1, Color.PURPLE, ("feudalism",))
        .board(P2, Color.YELLOW, ("machinery",))
        .board(P2, Color.RED, ("archery",))
        .build()
    )
    result = resolve_dogma(
        state,
        "engineering",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).score_pile) == {
        CardId("archery"),
        CardId("machinery"),
    }
    transfers = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind is ChangeKind.TRANSFER
    )
    assert len(transfers) == 1
    assert set(transfers[0].card_ids) == {CardId("archery"), CardId("machinery")}


def test_the_red_splay_can_be_accepted_after_a_no_op_demand() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("archery", "engineering"))
        .board(P1, Color.BLUE, ("alchemy",))
        .board(P2, Color.BLUE, ("pottery",))
        .build()
    )
    result = resolve_dogma(
        state,
        "engineering",
        choose_branch("splay-left"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).board.stack(Color.RED).splay is SplayDirection.LEFT


def test_an_equal_or_stronger_opponent_is_immune_to_the_demand() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("engineering",))
        .board(P2, Color.RED, ("archery",))
        .build()
    )
    # Equal castle counts skip the demand; both players then execute the optional splay effect.
    result = resolve_dogma(
        state,
        "engineering",
        decline(),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P2).board.stack(Color.RED).top == CardId("archery")
