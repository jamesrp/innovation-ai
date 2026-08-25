from __future__ import annotations

from dataclasses import replace

import pytest
from achievement_fixtures import (
    ACTIVE,
    OPPONENT,
    card_registry,
    place,
    playable_state,
    with_achievements,
    with_score,
)

from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    TerminalReason,
    TerminalResult,
)
from innovation_ai.innovation.terminal import (
    ACHIEVEMENT_VICTORY_COUNT,
    TerminalError,
    achievement_counts,
    achievement_victory_result,
    apply_terminal,
    board_color_counts,
    card_effect_draw,
    card_is_top_anywhere,
    direct_card_effect_win,
    draw_beyond_age_ten_result,
    has_strict_maximum,
    score_counts,
    unique_lowest_result,
    unique_lowest_score_result,
    unique_lowest_winner,
    unique_most_points_result,
    unique_most_result,
    unique_most_visible_icon_result,
    unique_most_winner,
    visible_icon_counts,
)
from innovation_ai.innovation.types import CardId, Color, Icon, NormalAchievementId, PlayerId
from innovation_ai.innovation.zones import ZoneOperationError, meld_card, score_card


def _empty_supplies(state: GameState) -> GameState:
    """Remove every supply card from the game so any draw exceeds age ten."""

    supplied = tuple(card_id for pile in state.supply.piles for card_id in pile)
    return replace(
        state,
        supply=replace(state.supply, piles=tuple(() for _ in range(10))),
        removed_cards=(*state.removed_cards, *supplied),
    )


# ---------------------------------------------------------------------------------------------
# Applying terminal results
# ---------------------------------------------------------------------------------------------


def test_apply_terminal_freezes_the_state_and_rejects_further_mutation() -> None:
    registry = card_registry()
    state = playable_state(registry)
    result = TerminalResult(TerminalReason.CARD_EFFECT, (ACTIVE,))

    ended = apply_terminal(state, result)
    assert ended.phase is GamePhase.TERMINAL
    assert ended.terminal_result == result
    assert state.phase is GamePhase.PLAY

    with pytest.raises(ZoneOperationError, match="terminal game state cannot be mutated"):
        score_card(ended, ACTIVE, ended.supply.pile(1)[0], registry)
    with pytest.raises(TerminalError, match="cannot be finalized twice"):
        apply_terminal(ended, result)


def test_terminal_results_canonicalize_winners_and_report_draws() -> None:
    assert TerminalResult(TerminalReason.CARD_EFFECT, ()).is_draw
    assert not direct_card_effect_win(ACTIVE).is_draw
    assert card_effect_draw() == TerminalResult(TerminalReason.CARD_EFFECT, ())
    assert direct_card_effect_win(OPPONENT).winners == (OPPONENT,)
    with pytest.raises(ValueError, match="canonical player order"):
        TerminalResult(TerminalReason.CARD_EFFECT, (OPPONENT, ACTIVE))
    with pytest.raises(ValueError, match="duplicates"):
        TerminalResult(TerminalReason.CARD_EFFECT, (ACTIVE, ACTIVE))


# ---------------------------------------------------------------------------------------------
# Sixth achievement
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("owned", list(range(0, ACHIEVEMENT_VICTORY_COUNT + 1)))
def test_achievement_victory_needs_exactly_six_achievements(owned: int) -> None:
    registry = card_registry()
    state = playable_state(registry)
    normal = tuple(NormalAchievementId)[:owned]
    state = with_achievements(state, ACTIVE, normal=normal)

    result = achievement_victory_result(state, ACTIVE)
    if owned >= ACHIEVEMENT_VICTORY_COUNT:
        assert result == TerminalResult(TerminalReason.ACHIEVEMENT_VICTORY, (ACTIVE,))
    else:
        assert result is None
    assert achievement_victory_result(state, OPPONENT) is None


def test_achievement_counts_include_normal_and_special_achievements() -> None:
    from innovation_ai.innovation.types import SpecialAchievementId

    registry = card_registry()
    state = playable_state(registry)
    state = with_achievements(
        state,
        ACTIVE,
        normal=(NormalAchievementId.AGE_1, NormalAchievementId.AGE_2),
        special=(SpecialAchievementId.WORLD,),
    )
    assert achievement_counts(state) == {ACTIVE: 3, OPPONENT: 0}


# ---------------------------------------------------------------------------------------------
# Draw above age ten
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("active_scores", "opponent_scores", "active_extra", "opponent_extra", "winners"),
    [
        ((5,), (), (), (), (PlayerId.PLAYER_1,)),
        ((), (5,), (), (), (PlayerId.PLAYER_2,)),
        ((5,), (5,), (NormalAchievementId.AGE_1,), (), (PlayerId.PLAYER_1,)),
        ((5,), (5,), (), (NormalAchievementId.AGE_1,), (PlayerId.PLAYER_2,)),
        ((5,), (5,), (), (), ()),
        ((), (), (), (), ()),
        ((5,), (5,), (NormalAchievementId.AGE_1,), (NormalAchievementId.AGE_2,), ()),
        ((5, 4), (5,), (), (NormalAchievementId.AGE_1,), (PlayerId.PLAYER_1,)),
    ],
)
def test_draw_beyond_age_ten_uses_score_then_achievements_then_a_draw(
    active_scores: tuple[int, ...],
    opponent_scores: tuple[int, ...],
    active_extra: tuple[NormalAchievementId, ...],
    opponent_extra: tuple[NormalAchievementId, ...],
    winners: tuple[PlayerId, ...],
) -> None:
    registry = card_registry()
    state = playable_state(registry)
    state = with_score(state, ACTIVE, active_scores, registry)
    state = with_score(state, OPPONENT, opponent_scores, registry)
    state = with_achievements(state, ACTIVE, normal=active_extra)
    state = with_achievements(state, OPPONENT, normal=opponent_extra)

    result = draw_beyond_age_ten_result(state, registry)
    assert result.reason is TerminalReason.DRAW_BEYOND_AGE_10
    assert result.winners == winners
    assert result.is_draw is (winners == ())


def test_draw_beyond_age_ten_reads_live_scores_and_supports_an_exhausted_supply() -> None:
    registry = card_registry()
    state = playable_state(registry)
    state = with_score(state, OPPONENT, (10, 10), registry)
    exhausted = _empty_supplies(state)

    assert score_counts(exhausted, registry) == {ACTIVE: 0, OPPONENT: 20}
    result = draw_beyond_age_ten_result(exhausted, registry)
    assert result.winners == (OPPONENT,)


# ---------------------------------------------------------------------------------------------
# Direct card-effect winners
# ---------------------------------------------------------------------------------------------


def test_direct_card_effect_win_is_immediate_and_typed() -> None:
    result = direct_card_effect_win(ACTIVE)
    assert result.reason is TerminalReason.CARD_EFFECT
    assert result.winners == (ACTIVE,)


@pytest.mark.parametrize(
    ("counts", "most", "lowest"),
    [
        ({PlayerId.PLAYER_1: 3, PlayerId.PLAYER_2: 1}, PlayerId.PLAYER_1, PlayerId.PLAYER_2),
        ({PlayerId.PLAYER_1: 1, PlayerId.PLAYER_2: 3}, PlayerId.PLAYER_2, PlayerId.PLAYER_1),
        ({PlayerId.PLAYER_1: 2, PlayerId.PLAYER_2: 2}, None, None),
        ({PlayerId.PLAYER_1: 0, PlayerId.PLAYER_2: 0}, None, None),
        ({PlayerId.PLAYER_2: 4}, PlayerId.PLAYER_2, PlayerId.PLAYER_2),
    ],
)
def test_unique_most_and_lowest_ignore_the_whole_effect_on_a_tie(
    counts: dict[PlayerId, int], most: PlayerId | None, lowest: PlayerId | None
) -> None:
    assert unique_most_winner(counts) is most
    assert unique_lowest_winner(counts) is lowest
    assert unique_most_result(counts) == (None if most is None else direct_card_effect_win(most))
    assert unique_lowest_result(counts) == (
        None if lowest is None else direct_card_effect_win(lowest)
    )


def test_unique_comparisons_reject_an_empty_count_mapping() -> None:
    with pytest.raises(TerminalError, match="at least one player count"):
        unique_most_winner({})
    with pytest.raises(TerminalError, match="at least one player count"):
        unique_lowest_winner({})


def test_has_strict_maximum_requires_beating_every_other_player() -> None:
    assert has_strict_maximum({ACTIVE: 3, OPPONENT: 2}, ACTIVE)
    assert not has_strict_maximum({ACTIVE: 2, OPPONENT: 2}, ACTIVE)
    assert not has_strict_maximum({ACTIVE: 1, OPPONENT: 2}, ACTIVE)
    assert has_strict_maximum({ACTIVE: 0}, ACTIVE)
    with pytest.raises(TerminalError, match="does not include"):
        has_strict_maximum({OPPONENT: 1}, ACTIVE)


def test_unique_most_points_matches_globalization_style_win_text() -> None:
    registry = card_registry()
    state = playable_state(registry)
    tied = with_score(with_score(state, ACTIVE, (5,), registry), OPPONENT, (5,), registry)
    assert unique_most_points_result(tied, registry) is None

    ahead = with_score(tied, ACTIVE, (1,), registry)
    assert unique_most_points_result(ahead, registry) == direct_card_effect_win(ACTIVE)


def test_unique_lowest_score_matches_ai_style_win_text() -> None:
    registry = card_registry()
    state = playable_state(registry)
    tied = with_score(with_score(state, ACTIVE, (5,), registry), OPPONENT, (5,), registry)
    assert unique_lowest_score_result(tied, registry) is None

    behind = with_score(tied, OPPONENT, (1,), registry)
    assert unique_lowest_score_result(behind, registry) == direct_card_effect_win(ACTIVE)


def test_unique_most_visible_icon_matches_bioengineering_style_win_text() -> None:
    registry = card_registry()
    state = playable_state(registry)
    state = place(state, ACTIVE, ("pottery",), registry)
    assert visible_icon_counts(state, Icon.LEAF, registry) == {ACTIVE: 3, OPPONENT: 0}
    assert unique_most_visible_icon_result(state, Icon.LEAF, registry) == direct_card_effect_win(
        ACTIVE
    )

    tied = place(state, OPPONENT, ("reformation",), registry)
    assert visible_icon_counts(tied, Icon.LEAF, registry) == {ACTIVE: 3, OPPONENT: 3}
    assert unique_most_visible_icon_result(tied, Icon.LEAF, registry) is None


def test_board_color_counts_and_top_card_lookup_support_win_conditions() -> None:
    registry = card_registry()
    state = playable_state(registry)
    state = place(state, ACTIVE, ("the-wheel", "clothing"), registry)
    state = place(state, OPPONENT, ("sailing",), registry)

    assert board_color_counts(state, Color.GREEN, registry) == {ACTIVE: 2, OPPONENT: 1}
    assert board_color_counts(state, Color.BLUE, registry) == {ACTIVE: 0, OPPONENT: 0}
    assert card_is_top_anywhere(state, CardId("clothing"))
    assert card_is_top_anywhere(state, CardId("sailing"))
    assert not card_is_top_anywhere(state, CardId("the-wheel"))
    assert not card_is_top_anywhere(state, CardId("pottery"))


def test_a_i_style_win_needs_both_top_cards_and_a_unique_lowest_score() -> None:
    registry = card_registry()
    state = playable_state(registry)
    state, _ = meld_card(state, ACTIVE, CardId("robotics"), registry)
    state, _ = meld_card(state, OPPONENT, CardId("software"), registry)
    assert card_is_top_anywhere(state, CardId("robotics"))
    assert card_is_top_anywhere(state, CardId("software"))

    # Decision 9: split boards qualify, but a score tie makes the win effect do nothing.
    assert unique_lowest_score_result(state, registry) is None
    ahead = with_score(state, ACTIVE, (5,), registry)
    assert unique_lowest_score_result(ahead, registry) == direct_card_effect_win(OPPONENT)
