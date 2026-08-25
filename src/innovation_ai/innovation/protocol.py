"""Deterministic setup, paid-turn, and mid-dogma decision protocol.

``current_decisions`` and ``apply_action`` are the single public transition boundary for every
player choice, including choices nested inside a dogma effect. Public action transitions always
resume deterministic effect work to a decision or terminal result. Low-level diagnostic
checkpoints produced by ``step_effect`` are resumed explicitly with ``resume_pending_effects``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from innovation_ai.innovation.achievements import (
    check_atomic_boundary,
    claim_normal_achievement,
    eligible_normal_achievements,
)
from innovation_ai.innovation.actions import (
    AchieveAction,
    ChooseStartingMeldAction,
    Decision,
    DecisionKind,
    DogmaAction,
    DrawAction,
    MeldAction,
    SemanticAction,
)
from innovation_ai.innovation.board import top_cards
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.dogma import start_dogma
from innovation_ai.innovation.effects.engine import (
    current_effect_decision,
    resume_effect,
    submit_effect_action,
)
from innovation_ai.innovation.effects.model import EffectStatus
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.errors import (
    EngineInvariantError,
    IllegalAction,
    InnovationEngineError,
)
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    TerminalState,
    TurnCounters,
)
from innovation_ai.innovation.terminal import apply_terminal, draw_beyond_age_ten_result
from innovation_ai.innovation.types import PlayerId
from innovation_ai.innovation.zones import assert_state_invariants, draw_action, meld_card

__all__ = [
    "EngineInvariantError",
    "IllegalAction",
    "InnovationEngineError",
    "Transition",
    "apply_action",
    "current_decision",
    "current_decisions",
    "effect_programs",
    "other_player",
    "resume_pending_effects",
    "terminal_transition",
]


def effect_programs(programs: EffectProgramRegistry | None = None) -> EffectProgramRegistry:
    """Return the supplied registry, or the discovered production registry."""

    return programs if programs is not None else load_effect_programs()


@dataclass(frozen=True, slots=True)
class Transition:
    """Result of applying one semantic action.

    Exactly one of ``decision`` and ``terminal`` is present. There is deliberately no third
    "effect resolution pending" outcome: a mid-dogma state always exposes its next decision, so a
    runner, a fuzzer, and a replay all see the same boundary shape.
    """

    state: GameState
    decision: Decision | None = None
    terminal: TerminalState | None = None

    def __post_init__(self) -> None:
        if (self.decision is not None) == (self.terminal is not None):
            raise ValueError("a transition must have exactly one next outcome")
        if (self.terminal is not None) != (self.state.phase is GamePhase.TERMINAL):
            raise ValueError("terminal transition outcome does not match state phase")


def other_player(player_id: PlayerId) -> PlayerId:
    """Return the other player in the supported two-player game."""

    return PlayerId.PLAYER_2 if player_id is PlayerId.PLAYER_1 else PlayerId.PLAYER_1


def _starting_decisions(state: GameState, registry: CardRegistry) -> tuple[Decision, ...]:
    decisions: list[Decision] = []
    for player_id, decision_id, choice in zip(
        PlayerId,
        state.starting_meld_decision_ids,
        state.starting_meld_choices,
        strict=True,
    ):
        if choice is not None:
            continue
        legal_actions: tuple[SemanticAction, ...] = tuple(
            ChooseStartingMeldAction(decision_id, card_id)
            for card_id in state.player(player_id).hand
        )
        decisions.append(
            Decision(
                decision_id,
                DecisionKind.STARTING_MELD,
                player_id,
                player_id,
                observe(state, player_id, registry),
                legal_actions,
            )
        )
    return tuple(decisions)


def _turn_actions(
    state: GameState,
    player_id: PlayerId,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> tuple[SemanticAction, ...]:
    decision_id = state.next_decision_id
    player = state.player(player_id)
    implemented = programs.implemented_card_ids()
    actions: list[SemanticAction] = [DrawAction(decision_id)]
    actions.extend(MeldAction(decision_id, card_id) for card_id in player.hand)
    # A top card whose effects are not yet registered is not offered, because a Dogma action on
    # it must fail loudly rather than silently behave as a no-op.
    actions.extend(
        DogmaAction(decision_id, card_id)
        for card_id in top_cards(player.board)
        if card_id in implemented
    )
    actions.extend(
        AchieveAction(decision_id, achievement_id)
        for achievement_id in eligible_normal_achievements(state, player_id, registry)
    )
    return tuple(actions)


def current_decisions(
    state: GameState,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> tuple[Decision, ...]:
    """Return every currently pending decision in deterministic player order.

    At a public action boundary, a pending effect stack exposes exactly one player decision.
    A low-level deterministic checkpoint may expose none until :func:`resume_pending_effects`
    advances it.
    """

    registry = registry or load_card_registry()
    if state.phase is GamePhase.TERMINAL:
        return ()
    if state.pending_effects:
        decision = current_effect_decision(state, effect_programs(programs), registry)
        return (decision,) if decision is not None else ()
    if state.phase is GamePhase.STARTING_MELDS:
        decisions = _starting_decisions(state, registry)
        if not decisions:
            raise EngineInvariantError("setup has all choices but was not finalized")
        return decisions
    if state.phase is not GamePhase.PLAY:
        raise EngineInvariantError(f"unsupported game phase: {state.phase}")
    if state.active_player is None or state.paid_actions_remaining < 1:
        raise EngineInvariantError("play decision requires an active player and paid action")
    legal_actions = _turn_actions(state, state.active_player, registry, effect_programs(programs))
    return (
        Decision(
            state.next_decision_id,
            DecisionKind.TURN_ACTION,
            state.active_player,
            state.active_player,
            observe(state, state.active_player, registry),
            legal_actions,
        ),
    )


def current_decision(
    state: GameState,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> Decision | None:
    """Return the first pending decision, or ``None`` at a terminal/deterministic checkpoint."""

    decisions = current_decisions(state, registry, programs)
    return decisions[0] if decisions else None


def terminal_transition(state: GameState, result: TerminalState) -> Transition:
    """Finalize the game immediately with ``result``.

    This is the shared handoff used by the paid-action protocol and by card effects that award
    victory or end the game. The caller must abandon every remaining dogma effect, sharing
    bonus, and paid action once it returns.
    """

    return Transition(apply_terminal(state, result), terminal=result)


def _achievement_boundary(state: GameState, registry: CardRegistry) -> Transition | GameState:
    """Run the special-achievement boundary check, stopping on an immediate win.

    Returning a :class:`Transition` means the game ended and no further turn work may happen.
    """

    result = check_atomic_boundary(state, registry)
    if result.terminal is not None:
        return Transition(result.state, terminal=result.terminal)
    return result.state


def _advance_after_paid_action(state: GameState) -> GameState:
    if state.paid_actions_remaining > 0:
        return state
    if state.active_player is None:
        raise EngineInvariantError("cannot finish a turn without an active player")
    return replace(
        state,
        active_player=other_player(state.active_player),
        turn_number=state.turn_number + 1,
        paid_actions_remaining=2,
        turn_counters=TurnCounters.empty(),
    )


def _next_transition(
    state: GameState,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> Transition:
    """Return the next decision after one completed unit of protocol work.

    The achievement boundary check runs before the turn can rotate, so Monument's per-turn
    counters are still intact when it evaluates.
    """

    if state.phase is GamePhase.TERMINAL:
        assert state.terminal_result is not None
        return Transition(state, terminal=state.terminal_result)
    if state.pending_effects:
        decision = current_effect_decision(state, programs, registry)
        if decision is None:  # pragma: no cover - defensive
            raise EngineInvariantError("a paused effect exposes no decision")
        return Transition(state, decision=decision)
    checked = _achievement_boundary(state, registry)
    if isinstance(checked, Transition):
        return checked
    advanced = _advance_after_paid_action(checked)
    decision = current_decision(advanced, registry, programs)
    if decision is None:  # pragma: no cover - guarded by phase/pending checks
        raise EngineInvariantError("non-terminal state has no current decision")
    return Transition(advanced, decision=decision)


def _apply_starting_meld(
    state: GameState,
    decision: Decision,
    action: ChooseStartingMeldAction,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> Transition:
    chooser = decision.chooser
    choices = list(state.starting_meld_choices)
    choices[tuple(PlayerId).index(chooser)] = action.card_id
    selected = replace(
        state,
        starting_meld_choices=(choices[0], choices[1]),
    )
    if any(choice is None for choice in selected.starting_meld_choices):
        decisions = current_decisions(selected, registry, programs)
        if not decisions:  # pragma: no cover - defensive
            raise EngineInvariantError("incomplete setup has no decision")
        return Transition(selected, decision=decisions[0])

    first_card, second_card = selected.starting_meld_choices
    assert first_card is not None and second_card is not None
    finalized = replace(selected, starting_meld_choices=(None, None))
    finalized, _ = meld_card(finalized, PlayerId.PLAYER_1, first_card, registry)
    finalized, _ = meld_card(finalized, PlayerId.PLAYER_2, second_card, registry)
    first_name = registry.card(first_card).name.casefold()
    second_name = registry.card(second_card).name.casefold()
    if first_name == second_name:  # Unique card titles make this an engine defect.
        raise EngineInvariantError("starting cards have equal titles")
    first_player = PlayerId.PLAYER_1 if first_name < second_name else PlayerId.PLAYER_2
    finalized = replace(
        finalized,
        phase=GamePhase.PLAY,
        active_player=first_player,
        turn_number=1,
        paid_actions_remaining=1,
        starting_meld_choices=(None, None),
        turn_counters=TurnCounters.empty(),
    )
    assert_state_invariants(finalized, registry)
    # No special-achievement predicate can be satisfied by two one-card age-1 stacks, so setup
    # does not need an achievement boundary check.
    next_decision = current_decision(finalized, registry, programs)
    if next_decision is None:  # pragma: no cover - defensive
        raise EngineInvariantError("finalized setup has no first-turn decision")
    return Transition(finalized, decision=next_decision)


def _claim_normal_achievement(
    state: GameState,
    action: AchieveAction,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> Transition:
    assert state.active_player is not None
    spent = replace(
        state,
        paid_actions_remaining=state.paid_actions_remaining - 1,
        next_decision_id=state.next_decision_id + 1,
    )
    result = claim_normal_achievement(spent, state.active_player, action.achievement_id, registry)
    if result.terminal is not None:
        return Transition(result.state, terminal=result.terminal)
    return _next_transition(result.state, registry, programs)


def _effect_transition(
    state: GameState,
    status: EffectStatus,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
    *,
    decision: Decision | None,
) -> Transition:
    """Map one :class:`EffectStatus` onto the public transition contract.

    ``ABORT_DOGMA`` and ``COMPLETE`` both mean the dogma action is over; decision 7 requires the
    abort to leave the paid action spent and any second paid action intact, which is exactly what
    falling through to the normal post-action boundary does.
    """

    if status is EffectStatus.AWAIT_DECISION:
        if decision is None:  # pragma: no cover - guarded by EffectResolution
            raise EngineInvariantError("an awaiting effect returned no decision")
        return Transition(state, decision=decision)
    if status is EffectStatus.TERMINAL:
        if state.terminal_result is None:  # pragma: no cover - defensive
            raise EngineInvariantError("a terminal effect produced no terminal result")
        return Transition(state, terminal=state.terminal_result)
    return _next_transition(state, registry, programs)


def _start_dogma(
    state: GameState,
    action: DogmaAction,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> Transition:
    assert state.active_player is not None
    spent = replace(
        state,
        paid_actions_remaining=state.paid_actions_remaining - 1,
        next_decision_id=state.next_decision_id + 1,
        next_dogma_action_id=state.next_dogma_action_id + 1,
    )
    resolution = start_dogma(
        spent,
        action.card_id,
        state.active_player,
        programs,
        registry,
        dogma_action_id=state.next_dogma_action_id,
    )
    return _effect_transition(
        resolution.state,
        resolution.status,
        registry,
        programs,
        decision=resolution.decision,
    )


def apply_action(
    state: GameState,
    action: SemanticAction,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> Transition:
    """Apply one currently legal action without mutating ``state``.

    A mid-dogma effect choice is routed to the effect executor; every other action is a setup
    choice or a paid turn action. Either way the result is the next decision or a terminal result.
    """

    registry = registry or load_card_registry()
    resolved_programs = effect_programs(programs)
    decisions = current_decisions(state, registry, resolved_programs)
    if not decisions:
        if state.phase is GamePhase.TERMINAL:
            raise EngineInvariantError("state is terminal and cannot accept an action")
        if state.pending_effects:
            raise EngineInvariantError(
                "state is between deterministic effect steps; call resume_pending_effects first"
            )
        raise EngineInvariantError("non-terminal state exposes no legal decision")
    decision = next(
        (candidate for candidate in decisions if candidate.decision_id == action.decision_id),
        decisions[0],
    )
    if action not in decision.legal_actions:
        raise IllegalAction(action, decision)

    if state.pending_effects:
        resolution = submit_effect_action(state, action, resolved_programs, registry)
        return _effect_transition(
            resolution.state,
            resolution.status,
            registry,
            resolved_programs,
            decision=resolution.decision,
        )
    if isinstance(action, ChooseStartingMeldAction):
        return _apply_starting_meld(state, decision, action, registry, resolved_programs)
    if state.phase is not GamePhase.PLAY or state.active_player is None:
        raise EngineInvariantError("paid action submitted outside play")
    if isinstance(action, DrawAction):
        updated, _, result = draw_action(state, state.active_player, registry)
        updated = replace(
            updated,
            paid_actions_remaining=updated.paid_actions_remaining - 1,
            next_decision_id=updated.next_decision_id + 1,
        )
        if result.beyond_age_ten:
            return terminal_transition(updated, draw_beyond_age_ten_result(updated, registry))
        return _next_transition(updated, registry, resolved_programs)
    if isinstance(action, MeldAction):
        updated, _ = meld_card(state, state.active_player, action.card_id, registry)
        updated = replace(
            updated,
            paid_actions_remaining=updated.paid_actions_remaining - 1,
            next_decision_id=updated.next_decision_id + 1,
        )
        return _next_transition(updated, registry, resolved_programs)
    if isinstance(action, DogmaAction):
        return _start_dogma(state, action, registry, resolved_programs)
    if isinstance(action, AchieveAction):
        return _claim_normal_achievement(state, action, registry, resolved_programs)
    raise EngineInvariantError(f"turn decision contained unsupported action: {action.kind}")


def resume_pending_effects(
    state: GameState,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> Transition:
    """Advance a state paused between deterministic effect steps to its next boundary.

    This exists for checkpoint replay and diagnostics. Normal play never needs it, because
    :func:`apply_action` already resumes to a decision or terminal result.
    """

    registry = registry or load_card_registry()
    resolved_programs = effect_programs(programs)
    if state.phase is GamePhase.TERMINAL:
        assert state.terminal_result is not None
        return Transition(state, terminal=state.terminal_result)
    if not state.pending_effects:
        return _next_transition(state, registry, resolved_programs)
    resolution = resume_effect(state, resolved_programs, registry)
    return _effect_transition(
        resolution.state,
        resolution.status,
        registry,
        resolved_programs,
        decision=resolution.decision,
    )
