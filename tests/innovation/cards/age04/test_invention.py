"""INVENTION: left-stack targeting, right splay reward, Wonder route, and replay."""

from __future__ import annotations

from support import choose_card, decline, resolve_dogma, scenario

from innovation_ai.innovation.actions import ChooseCardAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import load_effect_programs
from innovation_ai.innovation.state import state_hash
from innovation_ai.innovation.types import (
    CardId,
    Color,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_only_currently_left_splayed_stack_tops_are_offered() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("invention",))
        .board(P1, Color.BLUE, ("tools", "writing"), splay=SplayDirection.LEFT)
        .board(P1, Color.RED, ("archery", "metalworking"), splay=SplayDirection.RIGHT)
        .board(P2, Color.PURPLE, ("city-states",))
        .supply(4, ("anatomy",))
        .build()
    )
    result = resolve_dogma(
        state,
        "invention",
        choose_card("writing"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = {
        action.card_id
        for action in result.decisions[0].legal_actions
        if isinstance(action, ChooseCardAction)
    }
    assert offered == {CardId("writing")}
    assert result.state.player(P1).board.stack(Color.BLUE).splay is SplayDirection.RIGHT
    assert result.state.player(P1).score_pile == (CardId("anatomy"),)


def test_declining_the_left_splay_produces_no_reward() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("invention",))
        .board(P1, Color.BLUE, ("tools", "writing"), splay=SplayDirection.LEFT)
        .board(P2, Color.RED, ("metalworking",))
        .supply(4, ("anatomy",))
        .build()
    )
    result = resolve_dogma(
        state,
        "invention",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert not result.state.player(P1).score_pile
    assert CardId("anatomy") in result.state.supply.pile(4)


def test_five_colors_splayed_in_any_directions_claim_wonder_through_invention() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("sailing", "invention"), splay=SplayDirection.UP)
        .board(P1, Color.BLUE, ("tools", "writing"), splay=SplayDirection.UP)
        .board(P1, Color.RED, ("archery", "metalworking"), splay=SplayDirection.RIGHT)
        .board(P1, Color.YELLOW, ("agriculture", "masonry"), splay=SplayDirection.LEFT)
        .board(P1, Color.PURPLE, ("city-states", "code-of-laws"), splay=SplayDirection.RIGHT)
        .build()
    )
    result = resolve_dogma(
        state,
        "invention",
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert SpecialAchievementId.WONDER in result.state.player(P1).special_achievements


def test_choice_heavy_splay_resolution_replays_identically() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("invention",))
        .board(P1, Color.BLUE, ("tools", "writing"), splay=SplayDirection.LEFT)
        .board(P2, Color.RED, ("metalworking",))
        .supply(4, ("anatomy",))
        .build()
    )
    choices = (choose_card("writing"),)
    first = resolve_dogma(
        state, "invention", *choices, registry=REGISTRY, programs=PROGRAMS, verify_resume=True
    )
    second = resolve_dogma(
        state, "invention", *choices, registry=REGISTRY, programs=PROGRAMS, verify_resume=True
    )
    assert state_hash(first.state) == state_hash(second.state)
    assert first.decisions == second.decisions
