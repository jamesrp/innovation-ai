"""FISSION: mass removal as one atom, abort, self-removal, and cross-player targeting."""

from __future__ import annotations

from support import ScenarioBuilder, assert_conserved, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    EffectEventKind,
    EffectStatus,
    load_effect_programs,
    start_dogma,
    step_effect,
)
from innovation_ai.innovation.serialization import dumps_state, loads_state
from innovation_ai.innovation.state import state_hash
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()
RED_TEN = CardId("robotics")
NON_RED_TEN = CardId("databases")


def _vulnerable() -> ScenarioBuilder:
    """Return a position where the opponent has fewer clocks than Fission's three."""

    return scenario(REGISTRY).board(P1, Color.RED, ("fission",)).board(P2, Color.BLUE, ("pottery",))


def test_a_red_ten_removes_every_card_in_play_and_aborts() -> None:
    state = (
        _vulnerable().hand(P1, ("tools",)).score(P2, ("writing",)).supply(10, (RED_TEN,)).build()
    )
    assert REGISTRY.card(RED_TEN).color is Color.RED
    result = resolve_dogma(state, "fission", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.ABORT_DOGMA
    for player in result.state.players:
        assert not player.hand
        assert not player.score_pile
        assert all(not stack.cards for stack in player.board.stacks)
    assert_conserved(result.state, REGISTRY)


def test_the_mass_removal_is_one_atomic_operation() -> None:
    """Decision 4: a bulk atom exposes no intermediate state and one change event."""

    state = _vulnerable().hand(P1, ("tools",)).supply(10, (RED_TEN,)).build()
    result = resolve_dogma(state, "fission", registry=REGISTRY, programs=PROGRAMS)
    removals = tuple(
        event
        for event in result.events
        if event.change is not None and event.change.kind.value == "remove"
    )
    assert len(removals) == 1, "mass removal must be a single change record"
    assert len(removals[0].change.card_moves) > 1  # type: ignore[union-attr]
    assert removals[0].atomic_group_id is not None


def test_fission_removes_itself_and_the_unwind_survives_it() -> None:
    state = _vulnerable().supply(10, (RED_TEN,)).build()
    result = resolve_dogma(state, "fission", registry=REGISTRY, programs=PROGRAMS)
    assert CardId("fission") in result.state.removed_cards
    assert result.status is EffectStatus.ABORT_DOGMA
    assert not result.state.pending_effects
    assert not result.state.effect_variables


def test_a_non_red_ten_neither_removes_nor_aborts() -> None:
    state = _vulnerable().supply(10, (NON_RED_TEN,)).build()
    assert REGISTRY.card(NON_RED_TEN).color is not Color.RED
    result = resolve_dogma(
        state, "fission", choose_card("pottery"), registry=REGISTRY, programs=PROGRAMS
    )
    assert result.status is EffectStatus.COMPLETE
    assert NON_RED_TEN in result.state.player(P2).hand
    assert not result.state.removed_cards


def test_effect_two_targets_a_top_card_on_any_players_board() -> None:
    state = _vulnerable().supply(10, (NON_RED_TEN, "a-i")).build()
    result = resolve_dogma(
        state, "fission", choose_card("pottery"), registry=REGISTRY, programs=PROGRAMS
    )
    offered = {
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    }
    # Both boards are eligible, but Fission itself is excluded by its printed text.
    assert offered == {CardId("pottery")}
    assert CardId("fission") not in offered
    assert not result.state.player(P2).board.stack(Color.BLUE).cards


def test_abort_skips_the_sharing_bonus_and_leaves_the_paid_action_spent() -> None:
    """Decision 7: no bonus Draw, and any second paid action is still available."""

    state = _vulnerable().supply(10, (RED_TEN,)).active(P1, paid_actions=1).build()
    result = resolve_dogma(state, "fission", registry=REGISTRY, programs=PROGRAMS)
    assert result.state.paid_actions_remaining == 1
    assert all(not player.hand for player in result.state.players)
    assert result.events[-1].kind is EffectEventKind.ABORT_DOGMA


def test_a_checkpoint_immediately_before_the_removal_resumes_identically() -> None:
    state = _vulnerable().hand(P1, ("tools",)).supply(10, (RED_TEN,)).build()
    checkpoint = start_dogma(
        state, CardId("fission"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    ).state
    live = checkpoint
    for _ in range(60):
        restored = loads_state(dumps_state(live), REGISTRY)
        assert state_hash(restored) == state_hash(live)
        direct = step_effect(live, PROGRAMS, REGISTRY)
        resumed = step_effect(restored, PROGRAMS, REGISTRY)
        assert state_hash(direct.state) == state_hash(resumed.state)
        live = direct.state
        if direct.status is not EffectStatus.CONTINUE:
            assert direct.status is EffectStatus.ABORT_DOGMA
            break


def test_an_immune_opponent_skips_the_demand_but_still_shares_effect_two() -> None:
    """Rule 8.1/8.3: equal clocks grant demand immunity and simultaneously allow sharing."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.RED, ("fission",))
        .board(P2, Color.BLUE, ("rocketry",))
        .supply(10, (NON_RED_TEN, "a-i"))
        .build()
    )
    from innovation_ai.innovation.effects.dogma import dogma_schedule

    started = start_dogma(
        state, CardId("fission"), P1, PROGRAMS, REGISTRY, pause_before_first_step=True
    )
    schedule = dogma_schedule(started.state, started.state.pending_effects[0], PROGRAMS)
    # Rocketry has three clocks, matching Fission, so the demand is skipped entirely.
    assert schedule == ((2, True, True), (2, False, False))

    result = resolve_dogma(
        state,
        "fission",
        # The sharing opponent executes effect 2 first, then the activator.
        choose_card("rocketry"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    # Rocketry was returned by the opponent, leaving only Fission as a top card for the
    # activator's own execution, which its printed text excludes.
    assert not result.state.player(P2).board.stack(Color.BLUE).cards
    assert result.state.player(P1).board.stack(Color.RED).top == CardId("fission")
