"""Terminal-result construction for achievement, deck-exhaustion, and card-effect wins.

This module owns every way an Innovation game can end and the tie rules that decide whether a
win effect fires at all. It is deliberately free of effect-frame types so that the concurrent
effect work package can call it without a shared dependency:

* :func:`achievement_victory_result` implements the sixth-achievement rule.
* :func:`draw_beyond_age_ten_result` implements score, then achievement count, then draw.
* :func:`direct_card_effect_win` implements ``you win`` card text.
* :func:`unique_most_result` and :func:`unique_lowest_result` implement the rule that a
  ``single player with the most/lowest X`` effect is ignored entirely when players tie.

:func:`apply_terminal` is the only supported way to move authoritative state into its terminal
phase; callers must abandon all remaining effect work as soon as it returns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from innovation_ai.innovation.board import score_value, top_cards, visible_icons
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    TerminalReason,
    TerminalResult,
)
from innovation_ai.innovation.types import CardId, Color, Icon, PlayerId

ACHIEVEMENT_VICTORY_COUNT = 6


class TerminalError(RuntimeError):
    """A terminal result was requested for a state that cannot produce it."""


def _canonical_winners(*players: PlayerId) -> tuple[PlayerId, ...]:
    unique = set(players)
    return tuple(player_id for player_id in PlayerId if player_id in unique)


def apply_terminal(state: GameState, result: TerminalResult) -> GameState:
    """Return ``state`` moved into its terminal phase with ``result`` recorded.

    Callers must stop every remaining dogma effect, sharing bonus, and paid action immediately;
    the returned state rejects further zone mutation.
    """

    if state.phase is GamePhase.TERMINAL:
        raise TerminalError("a terminal game state cannot be finalized twice")
    return replace(state, phase=GamePhase.TERMINAL, terminal_result=result)


def achievement_victory_result(state: GameState, player_id: PlayerId) -> TerminalResult | None:
    """Return the sixth-achievement victory for ``player_id``, or ``None``."""

    if state.player(player_id).achievement_count >= ACHIEVEMENT_VICTORY_COUNT:
        return TerminalResult(TerminalReason.ACHIEVEMENT_VICTORY, (player_id,))
    return None


def score_counts(state: GameState, registry: CardRegistry | None = None) -> dict[PlayerId, int]:
    """Return each player's score-pile point total in canonical player order."""

    registry = registry or load_card_registry()
    return {player_id: score_value(state.player(player_id), registry) for player_id in PlayerId}


def achievement_counts(state: GameState) -> dict[PlayerId, int]:
    """Return each player's total claimed achievement count."""

    return {player_id: state.player(player_id).achievement_count for player_id in PlayerId}


def visible_icon_counts(
    state: GameState, icon: Icon, registry: CardRegistry | None = None
) -> dict[PlayerId, int]:
    """Return each player's currently visible count of one icon."""

    registry = registry or load_card_registry()
    return {
        player_id: visible_icons(state.player(player_id).board, registry)[icon]
        for player_id in PlayerId
    }


def board_color_counts(
    state: GameState, color: Color, registry: CardRegistry | None = None
) -> dict[PlayerId, int]:
    """Return how many cards of one color each player has on their board."""

    return {
        player_id: len(state.player(player_id).board.stack(color).cards) for player_id in PlayerId
    }


def card_is_top_anywhere(state: GameState, card_id: CardId) -> bool:
    """Whether ``card_id`` is currently a top card on either player's board."""

    return any(card_id in top_cards(player.board) for player in state.players)


def draw_beyond_age_ten_result(
    state: GameState, registry: CardRegistry | None = None
) -> TerminalResult:
    """Resolve deck exhaustion using score, then achievement count, then a draw."""

    registry = registry or load_card_registry()
    scores = score_counts(state, registry)
    best_score = max(scores.values())
    candidates = tuple(player_id for player_id in PlayerId if scores[player_id] == best_score)
    if len(candidates) == 1:
        return TerminalResult(TerminalReason.DRAW_BEYOND_AGE_10, candidates)
    counts = achievement_counts(state)
    best_count = max(counts[player_id] for player_id in candidates)
    winners = tuple(player_id for player_id in candidates if counts[player_id] == best_count)
    return TerminalResult(TerminalReason.DRAW_BEYOND_AGE_10, winners if len(winners) == 1 else ())


def direct_card_effect_win(player_id: PlayerId) -> TerminalResult:
    """Return the immediate card-effect victory for an unconditional ``you win`` effect."""

    return TerminalResult(TerminalReason.CARD_EFFECT, (player_id,))


def card_effect_draw() -> TerminalResult:
    """Return a card-effect game end that awards victory to nobody."""

    return TerminalResult(TerminalReason.CARD_EFFECT, ())


def _unique_extreme_winner(counts: Mapping[PlayerId, int], *, highest: bool) -> PlayerId | None:
    if not counts:
        raise TerminalError("a card-effect comparison requires at least one player count")
    extreme = max(counts.values()) if highest else min(counts.values())
    winners = _canonical_winners(
        *(player_id for player_id, count in counts.items() if count == extreme)
    )
    return winners[0] if len(winners) == 1 else None


def unique_most_winner(counts: Mapping[PlayerId, int]) -> PlayerId | None:
    """Return the single player with the most of a quantity, or ``None`` on a tie."""

    return _unique_extreme_winner(counts, highest=True)


def unique_lowest_winner(counts: Mapping[PlayerId, int]) -> PlayerId | None:
    """Return the single player with the lowest of a quantity, or ``None`` on a tie."""

    return _unique_extreme_winner(counts, highest=False)


def unique_most_result(counts: Mapping[PlayerId, int]) -> TerminalResult | None:
    """Return the card-effect win for the single player with the most, else ``None``.

    ``None`` means the entire win effect is ignored and play continues.
    """

    winner = unique_most_winner(counts)
    return None if winner is None else direct_card_effect_win(winner)


def unique_lowest_result(counts: Mapping[PlayerId, int]) -> TerminalResult | None:
    """Return the card-effect win for the single player with the lowest, else ``None``."""

    winner = unique_lowest_winner(counts)
    return None if winner is None else direct_card_effect_win(winner)


def has_strict_maximum(counts: Mapping[PlayerId, int], player_id: PlayerId) -> bool:
    """Whether ``player_id`` strictly exceeds every other supplied player count."""

    if player_id not in counts:
        raise TerminalError(f"comparison does not include {player_id}")
    return all(
        counts[player_id] > count for other, count in counts.items() if other is not player_id
    )


def unique_most_points_result(
    state: GameState, registry: CardRegistry | None = None
) -> TerminalResult | None:
    """Card-effect win for the single player with the most points, else ``None``."""

    return unique_most_result(score_counts(state, registry))


def unique_lowest_score_result(
    state: GameState, registry: CardRegistry | None = None
) -> TerminalResult | None:
    """Card-effect win for the single player with the lowest score, else ``None``."""

    return unique_lowest_result(score_counts(state, registry))


def unique_most_visible_icon_result(
    state: GameState, icon: Icon, registry: CardRegistry | None = None
) -> TerminalResult | None:
    """Card-effect win for the single player with the most visible ``icon``, else ``None``."""

    return unique_most_result(visible_icon_counts(state, icon, registry))
