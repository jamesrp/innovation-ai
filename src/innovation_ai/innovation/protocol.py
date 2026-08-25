"""Deterministic setup and paid-turn decision protocol."""

from __future__ import annotations

from dataclasses import dataclass, replace

from innovation_ai.innovation.actions import (
    AchieveAction,
    Action,
    ChooseStartingMeldAction,
    Decision,
    DecisionKind,
    DogmaAction,
    DrawAction,
    MeldAction,
    SemanticAction,
)
from innovation_ai.innovation.board import highest_top_value, score_value, top_cards
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.state import (
    EffectFrameState,
    EffectVariable,
    GamePhase,
    GameState,
    TerminalReason,
    TerminalState,
    TurnCounters,
)
from innovation_ai.innovation.types import NormalAchievementId, PlayerId
from innovation_ai.innovation.zones import assert_state_invariants, draw_card, meld_card


class InnovationEngineError(RuntimeError):
    """Base class for recoverable protocol errors and engine defects."""


class IllegalAction(InnovationEngineError):
    """An agent submitted an action outside the current legal-action set."""

    def __init__(self, action: Action, decision: Decision) -> None:
        self.action = action
        self.decision = decision
        super().__init__(f"illegal {action.kind.value} action for decision {decision.decision_id}")


class EngineInvariantError(InnovationEngineError):
    """The engine reached a state that violates the transition protocol."""


@dataclass(frozen=True, slots=True)
class Transition:
    """Result of applying one semantic action."""

    state: GameState
    decision: Decision | None = None
    terminal: TerminalState | None = None
    effect_resolution_pending: bool = False

    def __post_init__(self) -> None:
        outcomes = sum(
            (
                self.decision is not None,
                self.terminal is not None,
                self.effect_resolution_pending,
            )
        )
        if outcomes != 1:
            raise ValueError("a transition must have exactly one next outcome")
        if (self.terminal is not None) != (self.state.phase is GamePhase.TERMINAL):
            raise ValueError("terminal transition outcome does not match state phase")
        if self.effect_resolution_pending and not self.state.pending_effects:
            raise ValueError("pending transition requires a serializable effect frame")


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


def _eligible_normal_achievements(
    state: GameState, player_id: PlayerId, registry: CardRegistry
) -> tuple[NormalAchievementId, ...]:
    player = state.player(player_id)
    claimed = {
        achievement for candidate in state.players for achievement in candidate.normal_achievements
    }
    score = score_value(player, registry)
    top_value = highest_top_value(player.board, registry)
    return tuple(
        achievement_id
        for age, achievement_id in enumerate(NormalAchievementId, start=1)
        if achievement_id not in claimed and score >= 5 * age and top_value >= age
    )


def _turn_actions(
    state: GameState, player_id: PlayerId, registry: CardRegistry
) -> tuple[SemanticAction, ...]:
    decision_id = state.next_decision_id
    player = state.player(player_id)
    actions: list[SemanticAction] = [DrawAction(decision_id)]
    actions.extend(MeldAction(decision_id, card_id) for card_id in player.hand)
    actions.extend(DogmaAction(decision_id, card_id) for card_id in top_cards(player.board))
    actions.extend(
        AchieveAction(decision_id, achievement_id)
        for achievement_id in _eligible_normal_achievements(state, player_id, registry)
    )
    return tuple(actions)


def current_decisions(
    state: GameState, registry: CardRegistry | None = None
) -> tuple[Decision, ...]:
    """Return every currently pending decision in deterministic player order."""

    registry = registry or load_card_registry()
    if state.phase is GamePhase.TERMINAL or state.pending_effects:
        return ()
    if state.phase is GamePhase.STARTING_MELDS:
        decisions = _starting_decisions(state, registry)
        if not decisions:
            raise EngineInvariantError("setup has all choices but was not finalized")
        return decisions
    if state.phase is not GamePhase.PLAY:
        raise EngineInvariantError(f"unsupported game phase: {state.phase}")
    if state.active_player is None or state.paid_actions_remaining < 1:
        raise EngineInvariantError("play decision requires an active player and paid action")
    legal_actions = _turn_actions(state, state.active_player, registry)
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


def current_decision(state: GameState, registry: CardRegistry | None = None) -> Decision | None:
    """Return the first pending decision, or ``None`` for terminal/pending states."""

    decisions = current_decisions(state, registry)
    return decisions[0] if decisions else None


def _terminal_transition(state: GameState, result: TerminalState) -> Transition:
    terminal = replace(state, phase=GamePhase.TERMINAL, terminal_result=result)
    return Transition(terminal, terminal=result)


def _draw_exhaustion_result(state: GameState, registry: CardRegistry) -> TerminalState:
    scores = {player_id: score_value(state.player(player_id), registry) for player_id in PlayerId}
    highest_score = max(scores.values())
    candidates = tuple(player_id for player_id in PlayerId if scores[player_id] == highest_score)
    if len(candidates) == 1:
        return TerminalState(TerminalReason.DRAW_BEYOND_AGE_10, candidates)
    achievement_counts = {
        player_id: state.player(player_id).achievement_count for player_id in candidates
    }
    most_achievements = max(achievement_counts.values())
    winners = tuple(
        player_id for player_id in candidates if achievement_counts[player_id] == most_achievements
    )
    return TerminalState(TerminalReason.DRAW_BEYOND_AGE_10, winners if len(winners) == 1 else ())


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


def _next_transition(state: GameState, registry: CardRegistry) -> Transition:
    if state.phase is GamePhase.TERMINAL:
        assert state.terminal_result is not None
        return Transition(state, terminal=state.terminal_result)
    if state.pending_effects:
        return Transition(state, effect_resolution_pending=True)
    advanced = _advance_after_paid_action(state)
    decision = current_decision(advanced, registry)
    if decision is None:  # pragma: no cover - guarded by phase/pending checks
        raise EngineInvariantError("non-terminal state has no current decision")
    return Transition(advanced, decision=decision)


def _apply_starting_meld(
    state: GameState,
    decision: Decision,
    action: ChooseStartingMeldAction,
    registry: CardRegistry,
) -> Transition:
    chooser = decision.chooser
    choices = list(state.starting_meld_choices)
    choices[tuple(PlayerId).index(chooser)] = action.card_id
    selected = replace(
        state,
        starting_meld_choices=(choices[0], choices[1]),
    )
    if any(choice is None for choice in selected.starting_meld_choices):
        decisions = current_decisions(selected, registry)
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
    next_decision = current_decision(finalized, registry)
    if next_decision is None:  # pragma: no cover - defensive
        raise EngineInvariantError("finalized setup has no first-turn decision")
    return Transition(finalized, decision=next_decision)


def _claim_normal_achievement(
    state: GameState, action: AchieveAction, registry: CardRegistry
) -> Transition:
    assert state.active_player is not None
    player = state.player(state.active_player)
    replacement = replace(
        player,
        normal_achievements=(*player.normal_achievements, action.achievement_id),
    )
    updated = state.replace_player(replacement)
    updated = replace(
        updated,
        paid_actions_remaining=updated.paid_actions_remaining - 1,
        next_decision_id=updated.next_decision_id + 1,
    )
    if replacement.achievement_count >= 6:
        return _terminal_transition(
            updated,
            TerminalState(TerminalReason.ACHIEVEMENT_VICTORY, (state.active_player,)),
        )
    assert_state_invariants(updated, registry)
    return _next_transition(updated, registry)


def _start_dogma(state: GameState, action: DogmaAction, registry: CardRegistry) -> Transition:
    assert state.active_player is not None
    frame = EffectFrameState(
        "dogma-action",
        source_card_id=action.card_id,
        variables=(
            EffectVariable("activator", state.active_player.value),
            EffectVariable("dogma_action_id", state.next_dogma_action_id),
        ),
    )
    updated = replace(
        state,
        paid_actions_remaining=state.paid_actions_remaining - 1,
        pending_effects=(frame,),
        next_decision_id=state.next_decision_id + 1,
        next_dogma_action_id=state.next_dogma_action_id + 1,
    )
    assert_state_invariants(updated, registry)
    return Transition(updated, effect_resolution_pending=True)


def apply_action(
    state: GameState,
    action: SemanticAction,
    registry: CardRegistry | None = None,
) -> Transition:
    """Apply one currently legal action without mutating ``state``.

    Dogma selection creates a serializable handoff frame for WP4. All other WP3 actions return
    the next decision or a terminal result directly.
    """

    registry = registry or load_card_registry()
    decisions = current_decisions(state, registry)
    if not decisions:
        raise EngineInvariantError("state is terminal or awaiting effect resolution")
    decision = next(
        (candidate for candidate in decisions if candidate.decision_id == action.decision_id),
        decisions[0],
    )
    if action not in decision.legal_actions:
        raise IllegalAction(action, decision)

    if isinstance(action, ChooseStartingMeldAction):
        return _apply_starting_meld(state, decision, action, registry)
    if state.phase is not GamePhase.PLAY or state.active_player is None:
        raise EngineInvariantError("paid action submitted outside play")
    if isinstance(action, DrawAction):
        requested_age = highest_top_value(state.player(state.active_player).board, registry)
        updated, _, result = draw_card(state, requested_age, state.active_player, registry)
        updated = replace(
            updated,
            paid_actions_remaining=updated.paid_actions_remaining - 1,
            next_decision_id=updated.next_decision_id + 1,
        )
        if result.beyond_age_ten:
            return _terminal_transition(updated, _draw_exhaustion_result(updated, registry))
        return _next_transition(updated, registry)
    if isinstance(action, MeldAction):
        updated, _ = meld_card(state, state.active_player, action.card_id, registry)
        updated = replace(
            updated,
            paid_actions_remaining=updated.paid_actions_remaining - 1,
            next_decision_id=updated.next_decision_id + 1,
        )
        return _next_transition(updated, registry)
    if isinstance(action, DogmaAction):
        return _start_dogma(state, action, registry)
    if isinstance(action, AchieveAction):
        return _claim_normal_achievement(state, action, registry)
    raise EngineInvariantError(f"turn decision contained unsupported action: {action.kind}")


def finish_effect_resolution(state: GameState, registry: CardRegistry | None = None) -> Transition:
    """Advance after WP4 has completely cleared a paid Dogma action's frames."""

    registry = registry or load_card_registry()
    if state.phase is not GamePhase.PLAY or state.pending_effects:
        raise EngineInvariantError("effect resolution is not complete")
    return _next_transition(state, registry)
