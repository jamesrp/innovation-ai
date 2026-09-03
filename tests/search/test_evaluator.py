from __future__ import annotations

from dataclasses import replace

import pytest

from innovation_ai.innovation import (
    CardId,
    Color,
    ExplicitPlayerPosition,
    GamePhase,
    PlayerId,
    SpecialAchievementId,
    TerminalReason,
    TerminalResult,
    apply_terminal,
    build_explicit_state,
)
from innovation_ai.search.evaluator import (
    TERMINAL_DRAW_SENTINEL,
    TERMINAL_LOSS_SENTINEL,
    TERMINAL_WIN_SENTINEL,
    TerminalOrdering,
    evaluate_nonterminal_components,
    evaluate_state,
    terminal_ordering,
)

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2


def test_exact_leaf_formula_uses_visible_icons_stack_sizes_and_clamps() -> None:
    state = build_explicit_state(
        positions=(
            (
                P1,
                ExplicitPlayerPosition(
                    hand=(
                        CardId("agriculture"),
                        CardId("archery"),
                        CardId("clothing"),
                        CardId("code-of-laws"),
                        CardId("domestication"),
                        CardId("masonry"),
                    ),
                    score_pile=(CardId("construction"), CardId("engineering")),
                    board=(
                        (
                            Color.BLUE,
                            (CardId("writing"), CardId("mathematics"), CardId("calendar")),
                        ),
                        (Color.RED, (CardId("metalworking"), CardId("gunpowder"))),
                    ),
                    special_achievements=(
                        SpecialAchievementId.EMPIRE,
                        SpecialAchievementId.MONUMENT,
                    ),
                ),
            ),
            (
                P2,
                ExplicitPlayerPosition(
                    hand=(CardId("oars"),),
                    score_pile=(CardId("tools"),),
                    board=((Color.GREEN, (CardId("sailing"),)),),
                ),
            ),
        )
    )

    parts = evaluate_nonterminal_components(state, P1)

    assert parts.achievement == pytest.approx(0.2)
    assert parts.score == pytest.approx(0.04)  # ages 2+3 versus age 1
    # Formula expectations below are independently recomputed from the public geometry helpers;
    # component-level checks catch accidental use of printed covered icons or stack colors.
    assert parts.icons == pytest.approx(0.025)
    assert parts.board == pytest.approx(0.03)
    assert parts.hand == pytest.approx(0.10)  # difference five after clamp
    assert evaluate_state(state, P1) == pytest.approx(0.395)
    assert evaluate_state(state, P2) == pytest.approx(-0.395)


def test_terminal_ordering_precedes_leaf_formula_and_uses_finite_sentinels() -> None:
    base = build_explicit_state()
    p1_win = apply_terminal(base, TerminalResult(TerminalReason.CARD_EFFECT, (P1,)))
    p2_win = apply_terminal(base, TerminalResult(TerminalReason.CARD_EFFECT, (P2,)))
    draw = apply_terminal(base, TerminalResult(TerminalReason.CARD_EFFECT))
    both = replace(
        p1_win,
        terminal_result=TerminalResult(TerminalReason.CARD_EFFECT, (P1, P2)),
        phase=GamePhase.TERMINAL,
    )

    assert terminal_ordering(p1_win.terminal_result, P1) is TerminalOrdering.WIN  # type: ignore[arg-type]
    assert terminal_ordering(p2_win.terminal_result, P1) is TerminalOrdering.LOSS  # type: ignore[arg-type]
    assert terminal_ordering(draw.terminal_result, P1) is TerminalOrdering.DRAW  # type: ignore[arg-type]
    assert evaluate_state(p1_win, P1) == TERMINAL_WIN_SENTINEL
    assert evaluate_state(p2_win, P1) == TERMINAL_LOSS_SENTINEL
    assert evaluate_state(draw, P1) == TERMINAL_DRAW_SENTINEL
    assert evaluate_state(both, P1) == TERMINAL_DRAW_SENTINEL
    assert TERMINAL_LOSS_SENTINEL < -1.0 < 1.0 < TERMINAL_WIN_SENTINEL


def test_nonterminal_formula_rejects_terminal_state() -> None:
    state = apply_terminal(
        build_explicit_state(), TerminalResult(TerminalReason.CARD_EFFECT, (P1,))
    )
    with pytest.raises(ValueError, match="terminal"):
        evaluate_nonterminal_components(state, P1)
