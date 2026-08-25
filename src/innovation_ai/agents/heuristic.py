"""Small deterministic, non-ML Innovation policy."""

from __future__ import annotations

from collections import Counter

from innovation_ai.innovation.actions import (
    AchieveAction,
    ChooseStartingMeldAction,
    Decision,
    DecisionKind,
    DeclineAction,
    DogmaAction,
    DrawAction,
    FinishSelectionAction,
    MeldAction,
    SemanticAction,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import PlayerObservation
from innovation_ai.innovation.types import Icon


def _visible_icon_counts(player: PlayerObservation, registry: CardRegistry) -> Counter[Icon]:
    counts: Counter[Icon] = Counter()
    for stack in player.board:
        if stack.top_card_id is not None:
            counts.update(registry.card(stack.top_card_id).functional_icons)
        for covered in stack.covered_cards:
            counts.update(covered.visible_icons)
    return counts


class SimpleHeuristicAgent:
    """Prefer achievements, favorable Dogma actions, high melds, then draws.

    The policy uses only the supplied observation and immutable printed card data. It does not
    inspect authoritative state or parse dogma prose. Effect choices use the first legal semantic
    option, preferring an affirmative option over decline/finish when one exists.
    """

    def __init__(self, registry: CardRegistry | None = None) -> None:
        self._registry = registry or load_card_registry()

    def choose_action(self, decision: Decision, /) -> SemanticAction:
        """Choose deterministically from the current legal actions."""

        if decision.kind is DecisionKind.STARTING_MELD:
            choices = tuple(
                action
                for action in decision.legal_actions
                if isinstance(action, ChooseStartingMeldAction)
            )
            return min(
                choices,
                key=lambda action: (-self._registry.card(action.card_id).age, str(action.card_id)),
            )
        if decision.kind is DecisionKind.EFFECT_CHOICE:
            return next(
                (
                    action
                    for action in decision.legal_actions
                    if not isinstance(action, (DeclineAction, FinishSelectionAction))
                ),
                decision.legal_actions[0],
            )

        achievements = tuple(
            action for action in decision.legal_actions if isinstance(action, AchieveAction)
        )
        if achievements:
            return min(achievements, key=lambda action: action.achievement_id.value)

        observation = decision.observation
        own_icons = _visible_icon_counts(observation.player(decision.chooser), self._registry)
        opponent = next(
            player for player in observation.players if player.player_id is not decision.chooser
        )
        opposing_icons = _visible_icon_counts(opponent, self._registry)
        dogmas = tuple(
            action for action in decision.legal_actions if isinstance(action, DogmaAction)
        )
        if dogmas:
            ranked_dogmas = sorted(
                dogmas,
                key=lambda action: (
                    -(
                        own_icons[self._registry.card(action.card_id).featured_icon]
                        - opposing_icons[self._registry.card(action.card_id).featured_icon]
                    ),
                    -self._registry.card(action.card_id).age,
                    str(action.card_id),
                ),
            )
            best_dogma = ranked_dogmas[0]
            icon = self._registry.card(best_dogma.card_id).featured_icon
            if own_icons[icon] >= opposing_icons[icon]:
                return best_dogma

        melds = tuple(action for action in decision.legal_actions if isinstance(action, MeldAction))
        if melds:
            return min(
                melds,
                key=lambda action: (-self._registry.card(action.card_id).age, str(action.card_id)),
            )
        draw = next(
            (action for action in decision.legal_actions if isinstance(action, DrawAction)), None
        )
        if draw is not None:
            return draw
        if dogmas:
            return ranked_dogmas[0]
        return decision.legal_actions[0]


HeuristicAgent = SimpleHeuristicAgent
