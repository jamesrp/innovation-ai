"""BICYCLE: optional all-or-none hand/score exchange."""

from __future__ import annotations

from support import choose_branch, decline, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.state import GameState
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _state() -> GameState:
    return (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("bicycle",))
        .board(P2, Color.RED, ("metalworking",))
        .hand(P1, ("tools", "writing"))
        .score(P1, ("canal-building", "construction"))
        .build()
    )


def test_exchange_moves_both_complete_zones_in_one_atom() -> None:
    result = resolve_dogma(
        _state(),
        "bicycle",
        choose_branch("exchange"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).hand) == {
        CardId("canal-building"),
        CardId("construction"),
    }
    assert set(result.state.player(P1).score_pile) == {CardId("tools"), CardId("writing")}
    changes = tuple(event.change for event in result.events if event.change is not None)
    assert len(changes) == 1
    assert changes[0].kind.value == "exchange"
    assert len(changes[0].card_moves) == 4


def test_declining_exchanges_nothing() -> None:
    before = _state()
    result = resolve_dogma(
        before,
        "bicycle",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == before.player(P1).hand
    assert result.state.player(P1).score_pile == before.player(P1).score_pile
    assert result.qualifying_changes == 0


def test_the_complete_exchange_can_move_only_one_nonempty_side() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("bicycle",))
        .board(P2, Color.RED, ("metalworking",))
        .hand(P1, ("tools", "writing"))
        .build()
    )
    result = resolve_dogma(
        state,
        "bicycle",
        choose_branch("exchange"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert not result.state.player(P1).hand
    assert set(result.state.player(P1).score_pile) == {CardId("tools"), CardId("writing")}


def test_two_empty_zones_raise_no_meaningless_exchange_decision() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("bicycle",))
        .board(P2, Color.RED, ("metalworking",))
        .build()
    )
    result = resolve_dogma(state, "bicycle", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert result.qualifying_changes == 0
