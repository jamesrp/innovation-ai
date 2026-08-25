"""FISSION - effect 1 (demand): "I demand you draw a 10! If it is red, remove all hands, boards,
and score piles from the game! If this occurs, the dogma action is complete."
effect 2: "Return a top card other than Fission from any player's board. Draw a 10."

Fission is the slice's abort card and its hardest resume case. It pins:

* mass removal as **one** atomic operation, so no intermediate state is ever observable
  (decision 4) and no achievement check runs mid-removal;
* the abort itself: all remaining dogma work and the sharing bonus are skipped (decision 7),
  while the paid action stays spent and any second paid action stays available;
* self-removal - Fission removes itself from the board, so nothing during the unwind may assume
  the source card is still in play;
* a cross-player selector, since effect 2 targets a top card on *any* player's board.

The demand's own draw is not affected by the removal because it happens first, and the removal is
followed immediately by the abort, so the drawn card is removed along with everything else.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    ALL_PLAYERS,
    EXECUTOR,
    AbortDogmaNode,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    RemoveAllPlayCardsNode,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId

CARD_ID: Final[CardId] = CardId("fission")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "fission-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "fission-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "fission-return"),
    ),
    (
        SequenceNode("fission-demand", ("draw-ten", "red-branch")),
        DrawNode("draw-ten", ValueRef.literal(10), "drawn", player=EXECUTOR),
        ConditionNode(
            "red-branch",
            Predicate.card_color_is("drawn", Color.RED),
            "fission-red",
        ),
        SequenceNode("fission-red", ("mass-removal", "abort-dogma")),
        BatchNode("mass-removal", ("remove-all",)),
        RemoveAllPlayCardsNode("remove-all", result_variable="removed"),
        AbortDogmaNode("abort-dogma"),
        SequenceNode("fission-return", ("choose-top", "return-top", "draw-replacement")),
        # "any player's board" is a public zone, so this is an ordinary card choice; excluding
        # Fission itself is a printed restriction, not a general rule.
        ChoiceNode(
            "choose-top",
            ChoiceKind.CARD,
            "returned-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(ALL_PLAYERS, exclude_source_card=True),
        ),
        MoveNode(
            "return-top",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-top"),
        ),
        DrawNode("draw-replacement", ValueRef.literal(10), "replacement", player=EXECUTOR),
    ),
)
