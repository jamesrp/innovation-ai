"""CODE OF LAWS - "You may tuck a card from your hand of the same color as any card on your
board. If you do, you may splay that color of your cards left."

This card is in the slice because it exercises three things at once: an optional choice whose
candidate set is *relational* (same colour as any card already on the board), an "if you do"
guard that must read the tuck's own result rather than a global flag, and a splay of "that
color" - a colour derived from the tucked card, not a literal.

Rules decision 15 applies to the splay: the choice is offered whenever the tuck happened, and a
splay that changes nothing is still legal but earns no sharing credit, because the shared zone
primitive emits no change event for a no-op re-splay.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    CardSelectorKind,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SelectorRelation,
    SelectorRelationKind,
    SequenceNode,
    SplayNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, SplayDirection

CARD_ID: Final[CardId] = CardId("code-of-laws")

_TUCKABLE: Final = CardSelector(
    CardSelectorKind.HAND,
    EXECUTOR,
    relation=SelectorRelation(
        SelectorRelationKind.SAME_COLOR_AS_ANY,
        CardSelector.board(EXECUTOR),
    ),
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "code-of-laws-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "code-of-laws-effect"),),
    (
        SequenceNode("code-of-laws-effect", ("choose-tuck", "if-tucked")),
        ChoiceNode(
            "choose-tuck",
            ChoiceKind.CARD,
            "tuck-card",
            cards=_TUCKABLE,
            optional=True,
        ),
        ConditionNode("if-tucked", Predicate.truthy("tuck-card"), "tuck-then-splay"),
        SequenceNode("tuck-then-splay", ("tuck-chosen", "if-tuck-happened")),
        MoveNode(
            "tuck-chosen",
            MovementKind.TUCK,
            CardSelector.from_variable("tuck-card"),
            destination_player=EXECUTOR,
            result_variable="tucked",
        ),
        # "If you do" reads the tuck's own recorded result, not a global change flag.
        ConditionNode("if-tuck-happened", Predicate.truthy("tucked"), "splay-sequence"),
        SequenceNode("splay-sequence", ("bind-color", "offer-splay", "if-splay")),
        # "that color" is the tucked card's colour, bound before the splay choice so the splay
        # node never needs to re-derive it.
        LetNode("bind-color", "tuck-color", color_of="tuck-card"),
        ChoiceNode(
            "offer-splay",
            ChoiceKind.BRANCH,
            "splay-left",
            branches=("splay",),
            optional=True,
        ),
        ConditionNode("if-splay", Predicate.truthy("splay-left"), "splay-that-color"),
        SplayNode(
            "splay-that-color",
            EXECUTOR,
            color_variable="tuck-color",
            direction=SplayDirection.LEFT,
            result_variable="splayed",
        ),
    ),
)
