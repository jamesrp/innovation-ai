"""METALWORKING: repeat, physical reveal state, and the castle branch."""

from __future__ import annotations

import pytest
from support import ScenarioBuilder, assert_conserved, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    EffectEventKind,
    EffectStatus,
    load_effect_programs,
    start_dogma,
    step_effect,
)
from innovation_ai.innovation.effects.model import EffectInvariantError
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.serialization import dumps_state, loads_state
from innovation_ai.innovation.state import state_hash
from innovation_ai.innovation.types import CardId, Color, Icon, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    """Return a position where the opponent has fewer castles and does not share."""

    return (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("metalworking",))
        .board(P2, Color.BLUE, ("pottery",))
    )


def test_a_non_castle_draw_is_kept_and_the_repeat_stops() -> None:
    state = _solo().supply(1, ("agriculture",)).build()
    assert Icon.CASTLE not in REGISTRY.card(CardId("agriculture")).functional_icons
    result = resolve_dogma(state, "metalworking", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P1).hand == (CardId("agriculture"),)
    assert not result.state.player(P1).score_pile
    assert any(event.kind is EffectEventKind.KEEP for event in result.events)


def test_a_castle_draw_is_scored_and_the_effect_repeats() -> None:
    state = _solo().supply(1, ("archery", "masonry", "agriculture")).build()
    result = resolve_dogma(state, "metalworking", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).score_pile) == {CardId("archery"), CardId("masonry")}
    assert result.state.player(P1).hand == (CardId("agriculture"),)
    assert_conserved(result.state, REGISTRY)


def test_every_drawn_card_is_revealed_exactly_once() -> None:
    state = _solo().supply(1, ("archery", "agriculture")).build()
    result = resolve_dogma(state, "metalworking", registry=REGISTRY, programs=PROGRAMS)
    revealed = tuple(
        card_id
        for event in result.events
        if event.kind is EffectEventKind.REVEAL
        for card_id in event.card_ids
    )
    assert revealed == (CardId("archery"), CardId("agriculture"))


def test_a_reveal_marker_is_public_while_the_card_is_face_up() -> None:
    """Decision 18: a revealed card is deliberately visible to both players right now."""

    state = _solo().supply(1, ("archery", "agriculture")).build()
    checkpoint = start_dogma(
        state, CardId("metalworking"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    ).state
    saw_public_reveal = False
    for _ in range(60):
        if checkpoint.revealed:
            marker = checkpoint.revealed[0]
            opponent_view = observe(checkpoint, P2, REGISTRY)
            # The revealing player's hand is otherwise private, so this is the reveal itself.
            assert marker.card_id in opponent_view.revealed_cards
            assert marker.card_id in opponent_view.player(P1).hand.known_cards
            saw_public_reveal = True
        result = step_effect(checkpoint, PROGRAMS, REGISTRY)
        checkpoint = result.state
        if result.status is not EffectStatus.CONTINUE:
            break
    assert saw_public_reveal


def test_the_reveal_marker_clears_the_moment_the_card_is_scored_or_kept() -> None:
    """Decision 18: the marker is physical, so it does not survive the movement or the keep."""

    state = _solo().supply(1, ("archery", "agriculture")).build()
    result = resolve_dogma(state, "metalworking", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.revealed == ()
    final_view = observe(result.state, P2, REGISTRY)
    assert final_view.revealed_cards == ()
    # The kept card is back in a private hand and is no longer identifiable to the opponent.
    assert final_view.player(P1).hand.known_cards == ()


def test_a_mid_reveal_checkpoint_round_trips_to_an_identical_hash() -> None:
    state = _solo().supply(1, ("archery", "agriculture")).build()
    checkpoint = start_dogma(
        state, CardId("metalworking"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    ).state
    checked_a_reveal = False
    for _ in range(60):
        restored = loads_state(dumps_state(checkpoint), REGISTRY)
        assert state_hash(restored) == state_hash(checkpoint)
        assert restored.revealed == checkpoint.revealed
        checked_a_reveal = checked_a_reveal or bool(checkpoint.revealed)
        result = step_effect(restored, PROGRAMS, REGISTRY)
        checkpoint = result.state
        if result.status is not EffectStatus.CONTINUE:
            break
    assert checked_a_reveal


def test_the_repeat_ceiling_fails_loudly_rather_than_truncating() -> None:
    """Decision 6's principle: a deterministic ceiling must raise, never silently stop."""

    from innovation_ai.innovation.effects.program import (
        EXECUTOR,
        CardSelector,
        ConditionNode,
        DrawNode,
        EffectProgram,
        EffectProgramRegistry,
        MovementKind,
        MoveNode,
        Predicate,
        ProgramEffect,
        RepeatNode,
        SequenceNode,
        ValueRef,
    )
    from innovation_ai.innovation.types import DogmaEffectId

    card = CardId("metalworking")
    tight = EffectProgramRegistry(
        (
            EffectProgram(
                "metalworking-tight-ceiling",
                card,
                (ProgramEffect(DogmaEffectId(card, 1), False, "repeat"),),
                (
                    RepeatNode(
                        "repeat",
                        "body",
                        Predicate.card_has_icon("drawn", Icon.CASTLE),
                        maximum_iterations=1,
                    ),
                    SequenceNode("body", ("draw", "branch")),
                    DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
                    ConditionNode(
                        "branch",
                        Predicate.card_has_icon("drawn", Icon.CASTLE),
                        "score",
                    ),
                    MoveNode(
                        "score",
                        MovementKind.SCORE,
                        CardSelector.from_variable("drawn"),
                        destination_player=EXECUTOR,
                    ),
                ),
            ),
        )
    )
    state = _solo().supply(1, ("archery", "masonry", "oars")).build()
    with pytest.raises(EffectInvariantError, match="exceeded"):
        start_dogma(state, card, P1, tight, REGISTRY)


def test_a_shared_execution_gives_the_activator_one_free_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("metalworking",))
        .board(P2, Color.YELLOW, ("masonry",))
        .supply(1, ("agriculture", "clothing", "domestication"))
        .build()
    )
    result = resolve_dogma(state, "metalworking", registry=REGISTRY, programs=PROGRAMS)
    # The opponent keeps one non-castle card; the activator keeps one and gains the bonus Draw.
    assert len(result.state.player(P2).hand) == 1
    assert len(result.state.player(P1).hand) == 2
