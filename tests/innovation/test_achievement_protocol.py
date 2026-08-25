"""WP6 integration tests for achievement and terminal handling inside the turn protocol."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest
from achievement_fixtures import (
    ACTIVE,
    FIVE_NORMAL_ACHIEVEMENTS,
    OPPONENT,
    card_registry,
    monument_counters,
    place,
    playable_state,
    universe_board,
    with_achievements,
    with_score,
    wonder_board,
    world_board,
)

from innovation_ai.innovation.achievements import MonumentCountKind, check_atomic_boundary
from innovation_ai.innovation.actions import AchieveAction, DrawAction, MeldAction
from innovation_ai.innovation.protocol import (
    apply_action,
    current_decision,
    finish_effect_resolution,
    terminal_transition,
)
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    TerminalReason,
    state_hash,
)
from innovation_ai.innovation.terminal import direct_card_effect_win
from innovation_ai.innovation.types import (
    CardId,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
)
from innovation_ai.innovation.zones import CardLocation, ZoneOperationError, move_card


def _exhaust_supplies(state: GameState) -> GameState:
    supplied = tuple(card_id for pile in state.supply.piles for card_id in pile)
    return replace(
        state,
        supply=replace(state.supply, piles=tuple(() for _ in range(10))),
        removed_cards=(*state.removed_cards, *supplied),
    )


def test_achieve_action_is_enumerated_only_when_both_conditions_hold() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("tools",), registry)

    decision = current_decision(state, registry)
    assert decision is not None
    assert not any(isinstance(action, AchieveAction) for action in decision.legal_actions)

    scored = with_score(state, ACTIVE, (5,), registry)
    decision = current_decision(scored, registry)
    assert decision is not None
    achieve = [action for action in decision.legal_actions if isinstance(action, AchieveAction)]
    assert [action.achievement_id for action in achieve] == [NormalAchievementId.AGE_1]

    transition = apply_action(scored, achieve[0], registry)
    assert transition.state.player(ACTIVE).normal_achievements == (NormalAchievementId.AGE_1,)
    assert transition.state.paid_actions_remaining == 1


def test_achieve_action_producing_a_sixth_achievement_ends_the_game() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("tools",), registry)
    state = with_score(state, ACTIVE, (5,), registry)
    state = with_achievements(
        state,
        ACTIVE,
        normal=tuple(NormalAchievementId)[1:6],
    )

    decision = current_decision(state, registry)
    assert decision is not None
    achieve = next(action for action in decision.legal_actions if isinstance(action, AchieveAction))
    transition = apply_action(state, achieve, registry)

    assert transition.terminal is not None
    assert transition.terminal.reason is TerminalReason.ACHIEVEMENT_VICTORY
    assert transition.terminal.winners == (ACTIVE,)
    assert transition.state.phase is GamePhase.TERMINAL
    assert transition.decision is None


def test_paid_action_boundary_claims_a_special_achievement_automatically() -> None:
    registry = card_registry()
    # Board fixtures write zones directly, so Wonder is eligible but still unclaimed; the
    # boundary check inside the next paid action must claim it.
    state = wonder_board(playable_state(registry), ACTIVE, registry)
    assert state.player(ACTIVE).special_achievements == ()
    hand_card = state.supply.pile(1)[0]
    state, _ = move_card(state, hand_card, CardLocation.hand(ACTIVE), registry)

    decision = current_decision(state, registry)
    assert decision is not None
    meld = next(
        action
        for action in decision.legal_actions
        if isinstance(action, MeldAction) and action.card_id == hand_card
    )
    transition = apply_action(state, meld, registry)
    assert transition.state.player(ACTIVE).special_achievements == (SpecialAchievementId.WONDER,)
    assert transition.terminal is None


def test_paid_action_boundary_can_end_the_game_on_a_sixth_special_achievement() -> None:
    registry = card_registry()
    state = universe_board(playable_state(registry), ACTIVE, registry)
    state = with_achievements(state, ACTIVE, normal=FIVE_NORMAL_ACHIEVEMENTS)
    hand_card = state.supply.pile(1)[0]
    state, _ = move_card(state, hand_card, CardLocation.hand(ACTIVE), registry)

    decision = current_decision(state, registry)
    assert decision is not None
    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
    transition = apply_action(state, draw, registry)

    assert transition.terminal is not None
    assert transition.terminal.reason is TerminalReason.ACHIEVEMENT_VICTORY
    assert transition.terminal.winners == (ACTIVE,)
    assert transition.state.player(ACTIVE).special_achievements == (SpecialAchievementId.UNIVERSE,)


def test_monument_counters_reset_when_the_turn_rotates() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("tools",), registry)
    state = replace(state, paid_actions_remaining=1)
    state = monument_counters(state, ACTIVE, MonumentCountKind.TUCK, 5)

    decision = current_decision(state, registry)
    assert decision is not None
    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
    transition = apply_action(state, draw, registry)

    assert transition.state.active_player is OPPONENT
    assert transition.state.turn_counters.for_player(ACTIVE).tucked == 0
    assert SpecialAchievementId.MONUMENT not in transition.state.player(ACTIVE).special_achievements


def test_monument_is_claimed_before_the_turn_rotates_and_counters_reset() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("tools",), registry)
    state = replace(state, paid_actions_remaining=1)
    state = monument_counters(state, ACTIVE, MonumentCountKind.SCORE, 6)

    decision = current_decision(state, registry)
    assert decision is not None
    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
    transition = apply_action(state, draw, registry)

    assert SpecialAchievementId.MONUMENT in transition.state.player(ACTIVE).special_achievements
    assert transition.state.active_player is OPPONENT
    assert transition.state.turn_counters.for_player(ACTIVE).scored == 0


@pytest.mark.parametrize(
    ("active_scores", "opponent_scores", "extra", "winners"),
    [
        ((5,), (), (), (PlayerId.PLAYER_1,)),
        ((), (5,), (), (PlayerId.PLAYER_2,)),
        ((5,), (5,), (NormalAchievementId.AGE_1,), (PlayerId.PLAYER_1,)),
        ((5,), (5,), (), ()),
    ],
)
def test_draw_action_above_age_ten_ends_the_game_with_the_documented_tie_break(
    active_scores: tuple[int, ...],
    opponent_scores: tuple[int, ...],
    extra: tuple[NormalAchievementId, ...],
    winners: tuple[PlayerId, ...],
) -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("tools",), registry)
    state = with_score(state, ACTIVE, active_scores, registry)
    state = with_score(state, OPPONENT, opponent_scores, registry)
    state = with_achievements(state, ACTIVE, normal=extra)
    exhausted = _exhaust_supplies(state)

    decision = current_decision(exhausted, registry)
    assert decision is not None
    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
    transition = apply_action(exhausted, draw, registry)

    assert transition.terminal is not None
    assert transition.terminal.reason is TerminalReason.DRAW_BEYOND_AGE_10
    assert transition.terminal.winners == winners
    assert transition.state.phase is GamePhase.TERMINAL


def test_effect_resolution_handoff_runs_the_achievement_boundary_check() -> None:
    registry = card_registry()
    state = world_board(playable_state(registry), ACTIVE, registry)

    resumed = finish_effect_resolution(state, registry)
    assert resumed.state.player(ACTIVE).special_achievements == (SpecialAchievementId.WORLD,)
    assert resumed.decision is not None


def test_boundary_check_does_not_mutate_the_input_state_or_its_hash() -> None:
    registry = card_registry()
    state = universe_board(playable_state(registry), ACTIVE, registry)
    before = state_hash(state)

    result = check_atomic_boundary(state, registry)
    assert state_hash(state) == before
    assert state_hash(result.state) != before
    assert state.player(ACTIVE).special_achievements == ()


def test_top_card_lookup_still_reflects_the_registry_after_a_claim() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("robotics",), registry)
    result = check_atomic_boundary(state, registry)
    assert result.state.player(ACTIVE).board.stack(
        registry.card(CardId("robotics")).color
    ).top == CardId("robotics")


def test_card_effect_victory_handoff_ends_the_turn_immediately() -> None:
    registry = card_registry()
    state = place(playable_state(registry), ACTIVE, ("empiricism",), registry)

    transition = terminal_transition(state, direct_card_effect_win(ACTIVE))
    assert transition.terminal is not None
    assert transition.terminal.reason is TerminalReason.CARD_EFFECT
    assert transition.terminal.winners == (ACTIVE,)
    assert transition.decision is None
    assert current_decision(transition.state, registry) is None

    # An immediate win stops every remaining effect: the terminal state refuses more work.
    with pytest.raises(ZoneOperationError, match="terminal game state cannot be mutated"):
        move_card(transition.state, CardId("pottery"), CardLocation.hand(ACTIVE), registry)


def test_boundary_check_and_claim_order_are_hash_seed_independent() -> None:
    script = """
import json
from dataclasses import replace
from innovation_ai.innovation.achievements import check_atomic_boundary
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.state import GamePhase, build_setup_state_from_piles
from innovation_ai.innovation.types import CardId, PlayerId, SplayDirection
from innovation_ai.innovation.zones import meld_card, set_splay

registry = load_card_registry()
piles = tuple(
    tuple(sorted((card.id for card in registry.cards if card.age == age), key=str))
    for age in range(1, 11)
)
state = build_setup_state_from_piles(piles, seed=0, registry=registry)
state = replace(
    state,
    phase=GamePhase.PLAY,
    active_player=PlayerId.PLAYER_1,
    turn_number=3,
    paid_actions_remaining=2,
)
stacks = [
    (PlayerId.PLAYER_1, ("computers", "corporations", "empiricism", "flight", "skyscrapers"), None),
    (PlayerId.PLAYER_2, ("quantum-theory", "rocketry", "software"), SplayDirection.UP),
    (PlayerId.PLAYER_2, ("satellites", "databases"), SplayDirection.UP),
]
for player, names, splay in stacks:
    for name in names:
        state, _ = meld_card(state, player, CardId(name), registry)
    if splay is not None:
        state, _ = set_splay(state, player, registry.card(CardId(names[0])).color, splay, registry)
state = replace(state, turn_counters=state.turn_counters.increment(PlayerId.PLAYER_1, scored=6))
result = check_atomic_boundary(state, registry)
claims = [[c.player_id.value, c.achievement_id.value, c.route.value] for c in result.claims]
print(json.dumps(claims))
"""
    outputs = []
    for seed in ("1", "999"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script], text=True, env=environment
            ).strip()
        )
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0]) == [
        ["player-1", "monument", "automatic"],
        ["player-1", "universe", "automatic"],
        ["player-2", "world", "automatic"],
    ]
