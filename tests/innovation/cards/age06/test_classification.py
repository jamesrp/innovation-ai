"""CLASSIFICATION: colour-only disclosure and mandatory matching transfer/meld."""

from __future__ import annotations

from support import choose_card, choose_color, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectEventKind, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_only_the_chosen_colour_is_disclosed_and_every_matching_card_moves() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("classification",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .hand(P1, ("sailing", "archery"))
        .hand(P2, ("clothing", "currency", "oars"))
        .build()
    )
    result = resolve_dogma(
        state,
        "classification",
        choose_color(Color.GREEN),
        choose_card("clothing"),
        choose_card("currency"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.state.player(P1).hand == (CardId("archery"),)
    assert result.state.player(P2).hand == (CardId("oars"),)
    assert result.state.player(P1).board.stack(Color.GREEN).cards == (
        CardId("classification"),
        CardId("clothing"),
        CardId("currency"),
        CardId("sailing"),
    )
    # "Reveal the color" is public while the instruction resolves, but no selected private card
    # identity is ever exposed.
    color_reveals = tuple(
        event
        for event in result.events
        if event.kind is EffectEventKind.REVEAL and event.revealed_colors
    )
    assert len(color_reveals) == 1
    assert color_reveals[0].revealed_colors == (Color.GREEN,)
    assert color_reveals[0].card_ids == ()
    assert result.state.revealed == ()
    assert result.decisions[0].observation.revealed_cards == ()
    assert all(
        decision.observation.revealed_colors == (Color.GREEN,)
        and decision.observation.revealed_cards == ()
        for decision in result.decisions[1:]
    )
    assert tuple(decision.chooser for decision in result.decisions) == (P1, P1, P1)
    first_actions = result.decisions[0].legal_actions
    assert {getattr(action, "color", None) for action in first_actions} == {
        Color.GREEN,
        Color.RED,
    }
    assert all(not hasattr(action, "card_id") for action in first_actions)


def test_matching_transfer_and_single_meld_are_automatic_after_the_colour_choice() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("classification",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .hand(P1, ("writing",))
        .hand(P2, ("oars",))
        .build()
    )
    result = resolve_dogma(
        state,
        "classification",
        choose_color(Color.BLUE),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert len(result.decisions) == 1
    assert result.state.player(P1).board.stack(Color.BLUE).top == CardId("writing")
    assert result.state.player(P2).hand == (CardId("oars"),)
