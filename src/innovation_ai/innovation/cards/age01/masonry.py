"""MASONRY - "You may meld any number of cards from your hand, each with a castle. If you
melded four or more cards in this way, claim the Monument achievement."

The chosen subset is canonical, while meld order is a separate decision only within colors where
it can affect the resulting top card.  The linked Monument route receives the exact cards that
actually moved, not merely the selected count.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ClaimAchievementNode,
    EffectProgram,
    MovementKind,
    MoveNode,
    OrderGroup,
    ProgramEffect,
    SequenceNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon, SpecialAchievementId

CARD_ID: Final[CardId] = CardId("masonry")

EFFECTS: Final[EffectProgram] = EffectProgram(
    "masonry-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "masonry-effect"),),
    (
        SequenceNode(
            "masonry-effect",
            ("choose-castles", "order-melds", "meld-castles", "claim-monument"),
        ),
        ChoiceNode(
            "choose-castles",
            ChoiceKind.BOUNDED_CARDS,
            "selected-castles",
            cards=CardSelector.hand(EXECUTOR, icon=Icon.CASTLE),
            minimum=0,
            maximum=105,
        ),
        ChoiceNode(
            "order-melds",
            ChoiceKind.ORDER_CARDS,
            "meld-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("selected-castles"),
            order_group=OrderGroup.COLOR,
        ),
        MoveNode(
            "meld-castles",
            MovementKind.MELD,
            CardSelector.from_variable("selected-castles"),
            destination_player=EXECUTOR,
            moved_variable="melded-castles",
            order_variable="meld-order",
        ),
        ClaimAchievementNode(
            "claim-monument",
            SpecialAchievementId.MONUMENT,
            player=EXECUTOR,
            melded_count_variable="melded-castles",
        ),
    ),
)
