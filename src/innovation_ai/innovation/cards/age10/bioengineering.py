"""BIOENGINEERING - transfer an opponent's leafy top card to your score pile;
if anyone has fewer than three visible leaves, the unique leaf leader wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from innovation_ai.innovation.board import visible_icons
from innovation_ai.innovation.catalog import CardRegistry
from innovation_ai.innovation.effects import EffectContext, NamedPredicate
from innovation_ai.innovation.effects.program import (
    ALL_OTHER_PLAYERS,
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    EffectProgram,
    Extreme,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    WinMetric,
    WinMode,
    WinNode,
    ZoneKind,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("bioengineering")


def _any_player_has_fewer_than_three_leaves(
    state: Any, context: EffectContext, registry: CardRegistry
) -> bool:
    """Whether at least one player's live board shows fewer than three leaves."""

    del context
    return any(visible_icons(player.board, registry)[Icon.LEAF] < 3 for player in state.players)


EFFECTS: Final[EffectProgram] = EffectProgram(
    "bioengineering-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "bioengineering-transfer"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "bioengineering-win"),
    ),
    (
        SequenceNode("bioengineering-transfer", ("choose-leafy-top", "transfer-leafy-top")),
        ChoiceNode(
            "choose-leafy-top",
            ChoiceKind.CARD,
            "leafy-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(ALL_OTHER_PLAYERS, icon=Icon.LEAF),
        ),
        MoveNode(
            "transfer-leafy-top",
            MovementKind.TRANSFER,
            CardSelector.from_variable("leafy-top"),
            destination_player=EXECUTOR,
            destination_zone=ZoneKind.SCORE,
        ),
        ConditionNode(
            "bioengineering-win",
            Predicate.named("any-player-has-fewer-than-three-leaves"),
            "unique-leaf-leader-wins",
        ),
        WinNode(
            "unique-leaf-leader-wins",
            mode=WinMode.UNIQUE_EXTREME,
            metric=WinMetric.VISIBLE_ICON,
            icon=Icon.LEAF,
            extreme=Extreme.HIGHEST,
        ),
    ),
)

PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "any-player-has-fewer-than-three-leaves": _any_player_has_fewer_than_three_leaves,
}
