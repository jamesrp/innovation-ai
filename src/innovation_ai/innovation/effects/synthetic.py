"""Representative WP4 programs used to specify primitives before the card catalog exists.

These fixtures intentionally model only the printed effect portions needed to freeze the shared
VM contract. They are not the Milestone 1 broad card implementations owned by WP7.
"""

from __future__ import annotations

from functools import lru_cache

from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

from .program import (
    ACTIVATOR,
    EXECUTOR,
    AbortDogmaNode,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    EffectProgramRegistry,
    ExchangeNode,
    KeepNode,
    MovementKind,
    MoveNode,
    NestedNode,
    Predicate,
    ProgramEffect,
    RearrangeNode,
    RemoveAllPlayCardsNode,
    RepeatNode,
    RevealNode,
    SequenceNode,
    SplayNode,
    ValueRef,
)


def _effect(card: str, ordinal: int, root: str, *, demand: bool = False) -> ProgramEffect:
    return ProgramEffect(DogmaEffectId(CardId(card), ordinal), demand, root)


def _pottery() -> EffectProgram:
    card = CardId("pottery")
    return EffectProgram(
        "synthetic-pottery-v1",
        card,
        (_effect("pottery", 1, "pottery-effect"),),
        (
            SequenceNode(
                "pottery-effect",
                ("choose-returns", "if-returned"),
            ),
            ChoiceNode(
                "choose-returns",
                ChoiceKind.BOUNDED_CARDS,
                "returned",
                cards=CardSelector.hand(),
                minimum=0,
                maximum=3,
            ),
            ConditionNode(
                "if-returned",
                Predicate.truthy("returned"),
                "return-score-sequence",
            ),
            SequenceNode(
                "return-score-sequence",
                ("order-returns", "return-cards", "draw-reward", "score-reward"),
            ),
            ChoiceNode(
                "order-returns",
                ChoiceKind.ORDER_CARDS,
                "return-order",
                cards=CardSelector.from_variable("returned"),
                only_effective_return_order=True,
            ),
            MoveNode(
                "return-cards",
                MovementKind.RETURN,
                CardSelector.from_variable("return-order"),
            ),
            DrawNode("draw-reward", ValueRef.count("returned"), "reward"),
            MoveNode(
                "score-reward",
                MovementKind.SCORE,
                CardSelector.from_variable("reward"),
                destination_player=EXECUTOR,
            ),
        ),
    )


def _metalworking() -> EffectProgram:
    card = CardId("metalworking")
    return EffectProgram(
        "synthetic-metalworking-v1",
        card,
        (_effect("metalworking", 1, "metalworking-repeat"),),
        (
            RepeatNode(
                "metalworking-repeat",
                "metalworking-body",
                Predicate.card_has_icon("drawn", Icon.CASTLE),
            ),
            SequenceNode(
                "metalworking-body",
                ("draw-one", "reveal-one", "castle-branch"),
            ),
            DrawNode("draw-one", ValueRef.literal(1), "drawn"),
            RevealNode("reveal-one", CardSelector.from_variable("drawn")),
            ConditionNode(
                "castle-branch",
                Predicate.card_has_icon("drawn", Icon.CASTLE),
                "score-drawn",
                "keep-drawn",
            ),
            MoveNode(
                "score-drawn",
                MovementKind.SCORE,
                CardSelector.from_variable("drawn"),
                destination_player=EXECUTOR,
            ),
            KeepNode("keep-drawn", CardSelector.from_variable("drawn")),
        ),
    )


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
                CardSelector.hand(ACTIVATOR, highest_only=True),
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


def _fission() -> EffectProgram:
    card = CardId("fission")
    return EffectProgram(
        "synthetic-fission-v1",
        card,
        (_effect("fission", 1, "fission-demand", demand=True),),
        (
            SequenceNode(
                "fission-demand",
                ("draw-ten", "red-branch"),
            ),
            DrawNode("draw-ten", ValueRef.literal(10), "drawn"),
            ConditionNode(
                "red-branch",
                Predicate.card_color_is("drawn", Color.RED),
                "fission-red",
            ),
            SequenceNode(
                "fission-red",
                ("mass-removal", "abort-dogma"),
            ),
            BatchNode("mass-removal", ("remove-all",)),
            RemoveAllPlayCardsNode("remove-all"),
            AbortDogmaNode("abort-dogma"),
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


@lru_cache(maxsize=1)
def synthetic_program_registry() -> EffectProgramRegistry:
    """Return the immutable representative WP4 program registry."""

    return EffectProgramRegistry(
        (
            _pottery(),
            _metalworking(),
            _machinery(),
            _publications(),
            _fission(),
            _self_service(),
        )
    )
