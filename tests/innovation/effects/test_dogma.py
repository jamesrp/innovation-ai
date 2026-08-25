"""WP5 dogma orchestration: frozen icon counts, sharing order, demands, and abort.

The scenario matrix here is the WP5 acceptance gate: (opponent < / == / > activator icons) x
{demand-only, non-demand-only, two-effect} x {opponent no-op, opponent changes} x
{icon change mid-dogma} x {abort mid-effect}.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from support import choose_card, resolve_dogma, scenario

from innovation_ai.innovation.board import visible_icons
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    EffectEventKind,
    EffectStatus,
    dogma_schedule,
    frozen_icon_counts,
    load_effect_programs,
    start_dogma,
    step_effect,
)
from innovation_ai.innovation.effects.dogma import DOGMA_FRAME
from innovation_ai.innovation.effects.model import EffectInvariantError
from innovation_ai.innovation.serialization import SerializationError, dumps_state, loads_state
from innovation_ai.innovation.state import EffectVariable, state_hash
from innovation_ai.innovation.types import CardId, Color, Icon, PlayerId, SplayDirection
from innovation_ai.innovation.zones import set_splay

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_featured_icon_counts_are_frozen_once_and_never_recounted() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.RED, ("metalworking",))
        .build()
    )
    started = start_dogma(
        state, CardId("the-wheel"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    )
    frozen = frozen_icon_counts(started.state)
    assert frozen == (Icon.CASTLE, 3, 3)
    schedule = dogma_schedule(started.state, started.state.pending_effects[0], PROGRAMS)
    # Equality shares, so the opponent executes effect 1 before the activator.
    assert schedule == ((1, True, True), (1, False, False))


def test_splaying_mid_dogma_does_not_change_frozen_eligibility() -> None:
    """Rule 8.1: counts are fixed for the whole action even if the board changes."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.RED, ("archery", "metalworking"))
        .build()
    )
    started = start_dogma(
        state, CardId("the-wheel"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    )
    frame = started.state.pending_effects[0]
    before = dogma_schedule(started.state, frame, PROGRAMS)

    # Splaying red right exposes two more icon slots per covered card.
    splayed, change = set_splay(started.state, P2, Color.RED, SplayDirection.RIGHT, REGISTRY)
    assert change.changed
    assert (
        visible_icons(splayed.player(P2).board, REGISTRY)[Icon.CASTLE]
        > visible_icons(started.state.player(P2).board, REGISTRY)[Icon.CASTLE]
    )
    assert frozen_icon_counts(splayed) == frozen_icon_counts(started.state)
    assert dogma_schedule(splayed, splayed.pending_effects[0], PROGRAMS) == before


@pytest.mark.parametrize(
    ("opponent_stack", "expected_shares"),
    [
        # metalworking: three castles -> equal to the-wheel's three; equality shares.
        (("metalworking",), True),
        # archery: two castles -> fewer; no sharing.
        (("archery",), False),
        # pottery: no castles -> fewer; no sharing.
        (("pottery",), False),
    ],
)
def test_sharing_eligibility_matrix(opponent_stack: tuple[str, ...], expected_shares: bool) -> None:
    color = REGISTRY.card(CardId(opponent_stack[0])).color
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, color, opponent_stack)
        .build()
    )
    started = start_dogma(
        state, CardId("the-wheel"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    )
    schedule = dogma_schedule(started.state, started.state.pending_effects[0], PROGRAMS)
    opponent_entries = tuple(entry for entry in schedule if entry[1])
    assert bool(opponent_entries) is expected_shares


def test_opponent_executes_a_shared_effect_before_the_activator() -> None:
    """Rule 8.2: for each printed effect the sharing opponent goes first."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.RED, ("metalworking",))
        .supply(1, ("agriculture", "clothing", "domestication", "masonry", "mysticism"))
        .build()
    )
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    executors = tuple(event.executor for event in result.events if event.changed)
    # Two opponent draws, then two activator draws, then the activator's free Draw.
    assert executors == (P2, P2, P1, P1, P1)


def test_sharing_bonus_grants_exactly_one_free_draw_outside_the_paid_actions() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.RED, ("metalworking",))
        .active(P1, paid_actions=1)
        .build()
    )
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    assert len(result.state.player(P1).hand) == 3
    assert len(result.state.player(P2).hand) == 2
    # Rule 4/8.4: the bonus Draw is not one of the turn's actions.
    assert result.state.paid_actions_remaining == state.paid_actions_remaining


def test_no_sharing_means_no_free_draw() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.GREEN, ("the-wheel",))
        .board(P2, Color.BLUE, ("pottery",))
        .build()
    )
    result = resolve_dogma(state, "the-wheel", registry=REGISTRY, programs=PROGRAMS)
    assert len(result.state.player(P1).hand) == 2
    assert not result.state.player(P2).hand


def test_a_shared_execution_that_changes_nothing_gives_no_free_draw() -> None:
    """Rule 8.4 and decision 2: sharing with no gameplay change earns nothing."""

    # CODE OF LAWS' featured icon is crown. City States has two crowns to Code of Laws' two, so
    # the opponent shares, but neither player can tuck without a matching colour in hand.
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.PURPLE, ("city-states",))
        .build()
    )
    started = start_dogma(
        state, CardId("code-of-laws"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    )
    schedule = dogma_schedule(started.state, started.state.pending_effects[0], PROGRAMS)
    assert any(entry[2] for entry in schedule), "the opponent must be sharing for this case"

    result = resolve_dogma(state, "code-of-laws", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.qualifying_changes == 0
    assert not result.state.player(P1).hand
    assert not result.state.player(P2).hand


def test_a_demand_never_earns_the_activator_a_sharing_bonus() -> None:
    """Rule 8.4: only a shared *non-demand* execution can justify the free Draw."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("archery",))
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P2, ("tools",))
        .supply(1, ("writing",))
        .build()
    )
    result = resolve_dogma(
        state, "archery", choose_card("tools"), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.status is EffectStatus.COMPLETE
    assert result.qualifying_changes > 0, "the demand did change the game"
    # The activator's hand holds only the transferred card: no free Draw was added.
    assert result.state.player(P1).hand == (CardId("tools"),)


def test_an_immune_opponent_ignores_the_demand_entirely() -> None:
    """Rule 8.3: equal featured icons grant immunity."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("archery",))
        .board(P2, Color.RED, ("metalworking",))
        .hand(P2, ("tools",))
        .build()
    )
    started = start_dogma(
        state, CardId("archery"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    )
    frozen = frozen_icon_counts(started.state)
    assert frozen is not None and frozen[2] >= frozen[1]
    assert dogma_schedule(started.state, started.state.pending_effects[0], PROGRAMS) == ()

    result = resolve_dogma(state, "archery", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.state.player(P2).hand == (CardId("tools"),)
    assert not result.state.player(P1).hand
    assert result.qualifying_changes == 0


def test_each_printed_effect_completes_for_both_players_before_the_next_starts() -> None:
    """Rule 8.2's ordering, checked on a real two-effect card."""

    # POTTERY's featured icon is leaf; Agriculture also has three leaves, so the opponent shares.
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("pottery",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .build()
    )
    started = start_dogma(
        state, CardId("pottery"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    )
    schedule = dogma_schedule(started.state, started.state.pending_effects[0], PROGRAMS)
    assert schedule == (
        (1, True, True),
        (1, False, False),
        (2, True, True),
        (2, False, False),
    )


def test_partial_execution_needs_no_decision_when_nothing_is_selectable() -> None:
    """Rule 8.5: perform what is possible and ignore the impossible remainder."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("pottery",))
        .board(P2, Color.YELLOW, ("agriculture",))
        .build()
    )
    result = resolve_dogma(state, "pottery", registry=REGISTRY, programs=PROGRAMS)
    # Both hands are empty, so effect 1's bounded return raises no decision at all.
    assert result.decisions == ()
    assert result.status is EffectStatus.COMPLETE
    # Effect 2 still draws for both players, plus the sharing bonus for the activator.
    assert len(result.state.player(P1).hand) == 2
    assert len(result.state.player(P2).hand) == 1


def test_taking_dogma_that_can_change_nothing_is_still_legal() -> None:
    """Rule 8.5's closing sentence."""

    # Code of Laws with no matching colour in hand does nothing at all, for either player.
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.PURPLE, ("city-states",))
        .hand(P1, ("tools",))
        .build()
    )
    result = resolve_dogma(state, "code-of-laws", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.qualifying_changes == 0
    assert result.decisions == ()
    assert result.state.player(P1).hand == (CardId("tools"),)


def test_abort_skips_remaining_work_and_the_sharing_bonus_but_keeps_the_turn() -> None:
    """Decision 7: Fission's abort ends the action, not the turn."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("fission",))
        .board(P2, Color.BLUE, ("pottery",))
        .score(P2, ("tools",))
        .supply(10, ("robotics",))
        .active(P1, paid_actions=1)
        .build()
    )
    result = resolve_dogma(state, "fission", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.ABORT_DOGMA
    assert result.events[-1].kind is EffectEventKind.ABORT_DOGMA
    # Effect 2 never ran and no free Draw happened.
    assert not result.state.pending_effects
    assert all(not player.hand for player in result.state.players)
    assert all(not stack.cards for player in result.state.players for stack in player.board.stacks)
    assert result.state.paid_actions_remaining == 1


def test_abort_preserves_achievements_and_does_not_assume_the_source_card_survived() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("fission",))
        .board(P2, Color.BLUE, ("pottery",))
        .supply(10, ("robotics",))
        .build()
    )
    before = tuple(
        (player.normal_achievements, player.special_achievements) for player in state.players
    )
    result = resolve_dogma(state, "fission", registry=REGISTRY, programs=PROGRAMS)
    after = tuple(
        (player.normal_achievements, player.special_achievements) for player in result.state.players
    )
    assert after == before
    assert CardId("fission") in result.state.removed_cards


def test_a_paused_dogma_action_round_trips_at_every_step() -> None:
    from innovation_ai.innovation.serialization import dumps_state, loads_state

    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("archery",))
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P2, ("tools", "sailing"))
        .supply(1, ("writing",))
        .build()
    )
    checkpoint = start_dogma(
        state, CardId("archery"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    ).state
    seen_dogma_frame = False
    for _ in range(40):
        restored = loads_state(dumps_state(checkpoint), REGISTRY)
        assert state_hash(restored) == state_hash(checkpoint)
        seen_dogma_frame = seen_dogma_frame or any(
            frame.kind == DOGMA_FRAME for frame in checkpoint.pending_effects
        )
        result = step_effect(restored, PROGRAMS, REGISTRY)
        checkpoint = result.state
        if result.status is not EffectStatus.CONTINUE:
            break
    assert seen_dogma_frame


def test_a_second_dogma_action_cannot_start_while_one_is_pending() -> None:
    state = scenario(REGISTRY).board(P1, Color.GREEN, ("the-wheel",)).build()
    paused = start_dogma(
        state, CardId("the-wheel"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    ).state
    with pytest.raises(EffectInvariantError, match="empty effect runtime"):
        start_dogma(paused, CardId("the-wheel"), P1, PROGRAMS, REGISTRY)


def test_deserialization_rejects_an_incomplete_dogma_frame() -> None:
    state = scenario(REGISTRY).board(P1, Color.GREEN, ("the-wheel",)).build()
    paused = start_dogma(
        state, CardId("the-wheel"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    ).state
    payload = json.loads(dumps_state(paused))
    frame = payload["pending_effects"][0]
    frame["variables"] = [item for item in frame["variables"] if item["name"] != "activator_icons"]
    with pytest.raises(SerializationError, match="dogma frame"):
        loads_state(json.dumps(payload), REGISTRY)


def test_a_root_dogma_rejects_orphaned_runtime_variables() -> None:
    state = scenario(REGISTRY).board(P1, Color.GREEN, ("the-wheel",)).build()
    dirty = replace(state, effect_variables=(EffectVariable("orphan:value", 1),))
    with pytest.raises(EffectInvariantError, match="empty effect runtime"):
        start_dogma(dirty, CardId("the-wheel"), P1, PROGRAMS, REGISTRY)


def test_activating_an_unimplemented_card_fails_loudly() -> None:
    from innovation_ai.innovation.effects import UnimplementedCardError

    state = scenario(REGISTRY).board(P1, Color.BLUE, ("tools",)).build()
    with pytest.raises(UnimplementedCardError):
        start_dogma(state, CardId("tools"), P1, PROGRAMS, REGISTRY)
