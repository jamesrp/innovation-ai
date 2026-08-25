"""ROCKETRY - for every two visible clocks, return one card from the opponent's
score pile.

Each iteration uses the standard two-stage hidden-zone choice: the executor chooses a public
value, then the score-pile owner disambiguates tied identities.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    OPPONENT,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    ProgramEffect,
    SequenceNode,
    TimesNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("rocketry")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "rocketry-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "rocketry-effect"),),
    (
        TimesNode(
            "rocketry-effect",
            ValueRef.icon_count(Icon.CLOCK, EXECUTOR, per=2),
            "return-one-score",
        ),
        SequenceNode("return-one-score", ("choose-opponent-score", "return-opponent-score")),
        ChoiceNode(
            "choose-opponent-score",
            ChoiceKind.HIDDEN_CARD,
            "opponent-score",
            chooser=EXECUTOR,
            cards=CardSelector.score(OPPONENT),
            owner=OPPONENT,
        ),
        MoveNode(
            "return-opponent-score",
            MovementKind.RETURN,
            CardSelector.from_variable("opponent-score"),
        ),
    ),
)
