"""GLOBALIZATION - demand a leafy top-card return, then draw and score a 6;
if nobody shows more leaves than factories, the unique points leader wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from innovation_ai.innovation.board import visible_icons
from innovation_ai.innovation.catalog import CardRegistry
from innovation_ai.innovation.effects import EffectContext, NamedPredicate
from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    Extreme,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
    WinMetric,
    WinMode,
    WinNode,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon

CARD_ID: Final[CardId] = CardId("globalization")


def _no_player_has_more_leaves_than_factories(
    state: Any, context: EffectContext, registry: CardRegistry
) -> bool:
    """Whether every player's live leaf count is at most their factory count."""

    del context
    return all(
        visible_icons(player.board, registry)[Icon.LEAF]
        <= visible_icons(player.board, registry)[Icon.FACTORY]
        for player in state.players
    )


EFFECTS: Final[EffectProgram] = EffectProgram(
    "globalization-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), True, "globalization-demand"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "globalization-score-and-win"),
    ),
    (
        SequenceNode("globalization-demand", ("choose-leafy-top", "return-leafy-top")),
        ChoiceNode(
            "choose-leafy-top",
            ChoiceKind.CARD,
            "returned-leafy-top",
            chooser=EXECUTOR,
            cards=CardSelector.top_cards(EXECUTOR, icon=Icon.LEAF),
        ),
        MoveNode(
            "return-leafy-top",
            MovementKind.RETURN,
            CardSelector.from_variable("returned-leafy-top"),
        ),
        SequenceNode(
            "globalization-score-and-win",
            ("draw-and-score-six", "check-globalization-win"),
        ),
        BatchNode("draw-and-score-six", ("draw-six", "score-six")),
        DrawNode("draw-six", ValueRef.literal(6), "drawn-six", player=EXECUTOR),
        MoveNode(
            "score-six",
            MovementKind.SCORE,
            CardSelector.from_variable("drawn-six"),
            destination_player=EXECUTOR,
        ),
        ConditionNode(
            "check-globalization-win",
            Predicate.named("no-player-has-more-leaves-than-factories"),
            "unique-points-leader-wins",
        ),
        WinNode(
            "unique-points-leader-wins",
            mode=WinMode.UNIQUE_EXTREME,
            metric=WinMetric.SCORE,
            extreme=Extreme.HIGHEST,
        ),
    ),
)

PREDICATES: Final[Mapping[str, NamedPredicate]] = {
    "no-player-has-more-leaves-than-factories": _no_player_has_more_leaves_than_factories,
}
