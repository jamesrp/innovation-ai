"""Synthetic WP4 programs that specify shared VM primitives without a real card.

These fixtures exist only for primitives that no implemented card exercises yet: demand plus
mandatory exchange (Machinery), arbitrary stack rearrangement (Publications), and nested
non-demand execution (Self Service). Pottery, Metalworking, and Fission are now real cards in
``innovation.cards`` and their specification tests run against those production programs.

The synthetic registry is deliberately separate from
:func:`innovation_ai.innovation.effects.registry.load_effect_programs` and is never used by the
paid-turn protocol.
"""

from __future__ import annotations

from functools import lru_cache

from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

from .program import (
    ACTIVATOR,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    EffectProgramRegistry,
    ExchangeNode,
    Extreme,
    MovementKind,
    MoveNode,
    NestedNode,
    Predicate,
    ProgramEffect,
    RearrangeNode,
    SequenceNode,
    SplayNode,
)


def _effect(card: str, ordinal: int, root: str, *, demand: bool = False) -> ProgramEffect:
    return ProgramEffect(DogmaEffectId(CardId(card), ordinal), demand, root)


def _machinery() -> EffectProgram:
    card = CardId("machinery")
    return EffectProgram(
        "synthetic-machinery-v1",
        card,
        (
            _effect("machinery", 1, "machinery-demand", demand=True),
            _effect("machinery", 2, "machinery-shared"),
        ),
        (
            ExchangeNode(
                "machinery-demand",
                CardSelector.hand(EXECUTOR),
                CardSelector.hand(ACTIVATOR, extreme=Extreme.HIGHEST),
            ),
            SequenceNode(
                "machinery-shared",
                ("choose-castle", "score-castle", "optional-red-splay", "if-red-splay"),
            ),
            ChoiceNode(
                "choose-castle",
                ChoiceKind.CARD,
                "castle-card",
                cards=CardSelector.hand(icon=Icon.CASTLE),
            ),
            MoveNode(
                "score-castle",
                MovementKind.SCORE,
                CardSelector.from_variable("castle-card"),
                destination_player=EXECUTOR,
            ),
            ChoiceNode(
                "optional-red-splay",
                ChoiceKind.BRANCH,
                "red-splay",
                branches=("splay-left",),
                optional=True,
            ),
            ConditionNode(
                "if-red-splay",
                Predicate.truthy("red-splay"),
                "splay-red-left",
            ),
            SplayNode(
                "splay-red-left",
                EXECUTOR,
                Color.RED,
                SplayDirection.LEFT,
            ),
        ),
    )


def _publications() -> EffectProgram:
    card = CardId("publications")
    return EffectProgram(
        "synthetic-publications-v1",
        card,
        (_effect("publications", 1, "publications-effect"),),
        (
            SequenceNode(
                "publications-effect",
                ("choose-color", "if-color"),
            ),
            ChoiceNode(
                "choose-color",
                ChoiceKind.COLOR,
                "color",
                colors=tuple(Color),
                optional=True,
                minimum_stack_size=2,
            ),
            ConditionNode(
                "if-color",
                Predicate.truthy("color"),
                "reorder-sequence",
            ),
            SequenceNode(
                "reorder-sequence",
                ("choose-order", "apply-order"),
            ),
            ChoiceNode(
                "choose-order",
                ChoiceKind.ORDER_CARDS,
                "order",
                cards=CardSelector.stack(color_variable="color"),
            ),
            RearrangeNode("apply-order", EXECUTOR, "color", "order"),
        ),
    )


def _self_service() -> EffectProgram:
    card = CardId("self-service")
    return EffectProgram(
        "synthetic-self-service-v1",
        card,
        (_effect("self-service", 1, "self-service-effect"),),
        (
            SequenceNode(
                "self-service-effect",
                ("choose-top", "execute-selected"),
            ),
            ChoiceNode(
                "choose-top",
                ChoiceKind.CARD,
                "selected-card",
                cards=CardSelector.top_cards(exclude_source_card=True),
            ),
            NestedNode("execute-selected", "selected-card"),
        ),
    )


def _bounded_selection() -> EffectProgram:
    """A minimal optional bounded multi-select over a hand.

    Used by runtime/VM tests that need a program which pauses on its very first step without
    depending on a production card's full behaviour.
    """

    card = CardId("calendar")
    return EffectProgram(
        "synthetic-bounded-selection-v1",
        card,
        (_effect("calendar", 1, "choose-cards"),),
        (
            ChoiceNode(
                "choose-cards",
                ChoiceKind.BOUNDED_CARDS,
                "selected",
                cards=CardSelector.hand(EXECUTOR),
                minimum=0,
                maximum=3,
            ),
        ),
    )


@lru_cache(maxsize=1)
def synthetic_program_registry() -> EffectProgramRegistry:
    """Return the immutable representative WP4 program registry."""

    return EffectProgramRegistry(
        (
            _bounded_selection(),
            _machinery(),
            _publications(),
            _self_service(),
        )
    )
