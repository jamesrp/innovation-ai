"""DEMOCRACY - optionally return any number of hand cards; reward a strict new
per-dogma return-count record with a drawn-and-scored 8.

Each selected return is a separate instruction, so the root dogma qualifying-change counter is a
serializable tally of returned cards.  Previous reward draws/scores contribute a known two events;
the named predicate subtracts those before comparing the current scoped count.  Dogma execution
is opponent-first, so entry 2 observes entry 1's completed state while a new dogma action starts
with a fresh root counter (rules decision 11).
"""

from __future__ import annotations

from typing import Any, Final

from innovation_ai.innovation.effects.model import get_effect_variable
from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    BatchNode,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    ConditionNode,
    DrawNode,
    EffectProgram,
    ForEachCardNode,
    LetNode,
    MovementKind,
    MoveNode,
    NamedPredicate,
    OrderGroup,
    Predicate,
    ProgramEffect,
    SequenceNode,
    ValueRef,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, SpecialAchievementId

CARD_ID: Final[CardId] = CardId("democracy")


def _returned_more_than_every_prior_executor(state: Any, context: Any, registry: Any) -> bool:
    """Compare this entry's scoped count with the greatest prior Democracy count.

    Two-player dogma has at most one prior executor.  The prior reward contributes one draw and
    one score event.  If that score also claimed Monument, its achievement event is discounted;
    an immediate sixth-achievement win has already terminated before another entry can execute.
    """

    del registry
    current = get_effect_variable(state, context, "return-count", 0)
    if not isinstance(current, int) or isinstance(current, bool):
        return False
    if context.nested or "/entry-2/" not in f"/{context.scope}/":
        return current > 0
    aggregate = next(
        (
            variable.value
            for variable in state.effect_variables
            if variable.name == "dogma:qualifying-change-count"
        ),
        0,
    )
    if not isinstance(aggregate, int) or isinstance(aggregate, bool):
        return False
    prior_contribution = max(0, aggregate - current)
    if prior_contribution == 0:
        prior = 0
    else:
        previous = next(
            player for player in state.players if player.player_id is not context.executor
        )
        monument_event = int(
            SpecialAchievementId.MONUMENT in previous.special_achievements
            and state.turn_counters.for_player(previous.player_id).scored >= 6
        )
        prior = max(0, prior_contribution - 2 - monument_event)
    return current > prior


PREDICATES: Final[dict[str, NamedPredicate]] = {
    "returned-more-than-every-prior-executor": _returned_more_than_every_prior_executor,
}

EFFECTS: Final[EffectProgram] = EffectProgram(
    "democracy-v1",
    CARD_ID,
    (ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "democracy-effect"),),
    (
        SequenceNode(
            "democracy-effect",
            (
                "choose-returns",
                "snapshot-return-count",
                "order-returns",
                "return-each-card",
                "if-new-record",
            ),
        ),
        ChoiceNode(
            "choose-returns",
            ChoiceKind.BOUNDED_CARDS,
            "selected-returns",
            chooser=EXECUTOR,
            cards=CardSelector.hand(EXECUTOR),
            minimum=0,
            maximum=105,
        ),
        LetNode(
            "snapshot-return-count",
            "return-count",
            value=ValueRef.count("selected-returns"),
        ),
        ChoiceNode(
            "order-returns",
            ChoiceKind.ORDER_CARDS,
            "return-order",
            chooser=EXECUTOR,
            cards=CardSelector.from_variable("selected-returns"),
            order_group=OrderGroup.AGE,
        ),
        ForEachCardNode("return-each-card", "return-order", "returning-card", "return-card"),
        MoveNode(
            "return-card",
            MovementKind.RETURN,
            CardSelector.from_variable("returning-card"),
        ),
        ConditionNode(
            "if-new-record",
            Predicate.named("returned-more-than-every-prior-executor"),
            "draw-and-score-eight",
        ),
        BatchNode("draw-and-score-eight", ("draw-eight", "score-eight")),
        DrawNode("draw-eight", ValueRef.literal(8), "reward-eight", player=EXECUTOR),
        MoveNode(
            "score-eight",
            MovementKind.SCORE,
            CardSelector.from_variable("reward-eight"),
            destination_player=EXECUTOR,
        ),
    ),
)
