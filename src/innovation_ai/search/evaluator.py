"""Version-1 hand-engineered leaf evaluator for sampled Innovation search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from innovation_ai.innovation.board import score_value, visible_icons
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import GameState, TerminalResult
from innovation_ai.innovation.types import Color, Icon, PlayerId

EVALUATOR_VERSION = "hand-engineered-leaf-v1"

# Nonterminal values are in [-1, 1].  Finite sentinels make terminal ordering explicit while
# remaining safe in canonical JSON telemetry and ordinary alpha/beta comparisons.
TERMINAL_WIN_SENTINEL = 2.0
TERMINAL_DRAW_SENTINEL = 0.0
TERMINAL_LOSS_SENTINEL = -2.0
POSITIVE_TERMINAL_SENTINEL = TERMINAL_WIN_SENTINEL
NEGATIVE_TERMINAL_SENTINEL = TERMINAL_LOSS_SENTINEL


class TerminalOrdering(IntEnum):
    """Strict ordering category for a terminal result from the root player's viewpoint."""

    LOSS = -1
    DRAW = 0
    WIN = 1


@dataclass(frozen=True, slots=True)
class LeafEvaluation:
    """Exact additive components of a nonterminal v1 evaluation."""

    achievement: float
    score: float
    icons: float
    board: float
    hand: float

    @property
    def value(self) -> float:
        """Return the component sum in the evaluator's declared order."""

        return self.achievement + self.score + self.icons + self.board + self.hand

    def payload(self) -> dict[str, float]:
        """Return finite JSON-compatible component telemetry."""

        return {
            "achievement": self.achievement,
            "score": self.score,
            "icons": self.icons,
            "board": self.board,
            "hand": self.hand,
            "value": self.value,
        }


def _other_player(player_id: PlayerId) -> PlayerId:
    return PlayerId.PLAYER_2 if player_id is PlayerId.PLAYER_1 else PlayerId.PLAYER_1


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return min(maximum, max(minimum, value))


def terminal_ordering(result: TerminalResult, root_player: PlayerId) -> TerminalOrdering:
    """Classify a terminal result before any nonterminal component is inspected.

    A sole root winner is a win, a sole opponent winner is a loss, and terminal results with both
    or neither player winning are neutral as required by the Milestone 4 contract.
    """

    if result.winners == (root_player,):
        return TerminalOrdering.WIN
    if result.winners == (_other_player(root_player),):
        return TerminalOrdering.LOSS
    return TerminalOrdering.DRAW


def terminal_value(result: TerminalResult, root_player: PlayerId) -> float:
    """Return the finite sentinel associated with :func:`terminal_ordering`."""

    ordering = terminal_ordering(result, root_player)
    if ordering is TerminalOrdering.WIN:
        return TERMINAL_WIN_SENTINEL
    if ordering is TerminalOrdering.LOSS:
        return TERMINAL_LOSS_SENTINEL
    return TERMINAL_DRAW_SENTINEL


def evaluate_nonterminal_components(
    state: GameState,
    root_player: PlayerId,
    registry: CardRegistry | None = None,
) -> LeafEvaluation:
    """Evaluate the exact frozen v1 formula and return its five additive components."""

    if state.terminal_result is not None:
        raise ValueError("nonterminal evaluator cannot score a terminal state")
    registry = registry or load_card_registry()
    opponent = _other_player(root_player)
    root = state.player(root_player)
    other = state.player(opponent)

    achievement = 0.10 * _clamp(root.achievement_count - other.achievement_count, -5, 5)
    score = 0.01 * _clamp(
        score_value(root, registry) - score_value(other, registry),
        -15,
        15,
    )

    root_icons = visible_icons(root.board, registry)
    opponent_icons = visible_icons(other.board, registry)
    icon_mean = sum(
        _clamp(root_icons[icon] - opponent_icons[icon], -3, 3) / 3 for icon in Icon
    ) / len(Icon)
    icons = 0.15 * icon_mean

    board_mean = sum(
        _clamp(
            len(root.board.stack(color).cards) - len(other.board.stack(color).cards),
            -2,
            2,
        )
        / 2
        for color in Color
    ) / len(Color)
    board = 0.10 * board_mean

    hand = 0.02 * _clamp(len(root.hand) - len(other.hand), -5, 5)
    return LeafEvaluation(achievement, score, icons, board, hand)


def evaluate_nonterminal(
    state: GameState,
    root_player: PlayerId,
    registry: CardRegistry | None = None,
) -> float:
    """Return the exact frozen v1 value for a nonterminal authoritative state."""

    return evaluate_nonterminal_components(state, root_player, registry).value


def evaluate_state(
    state: GameState,
    root_player: PlayerId,
    registry: CardRegistry | None = None,
) -> float:
    """Evaluate a state with terminal ordering checked before the leaf formula."""

    if state.terminal_result is not None:
        return terminal_value(state.terminal_result, root_player)
    return evaluate_nonterminal(state, root_player, registry)


@dataclass(frozen=True, slots=True)
class HandEngineeredEvaluator:
    """Small callable wrapper for dependency injection into later search code."""

    registry: CardRegistry | None = None
    version: str = EVALUATOR_VERSION

    def __post_init__(self) -> None:
        if self.version != EVALUATOR_VERSION:
            raise ValueError(f"unsupported hand-engineered evaluator version {self.version!r}")

    def evaluate(self, state: GameState, root_player: PlayerId) -> float:
        """Evaluate ``state`` from the fixed root player's perspective."""

        return evaluate_state(state, root_player, self.registry)

    def __call__(self, state: GameState, root_player: PlayerId) -> float:
        return self.evaluate(state, root_player)


# A concise functional alias is useful in tree code without obscuring the public contract name.
evaluate_leaf = evaluate_state
