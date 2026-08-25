"""WP5 dogma orchestration: frozen icon counts, sharing, demands, and the sharing bonus.

The Dogma action is a third frame kind, ``dogma-action``, driven by a small step machine rather
than the declarative node graph. Every step is one serializable frame update, so a paused dogma
action round-trips through the WP9 state schema and resumes to an identical hash.

Rules mapping
-------------
* 8.1 - featured-icon counts are computed **once**, at step 0, and stored as integers in the
  frame. Nothing recomputes them, so splaying mid-dogma cannot change eligibility.
* 8.1 - equality both grants sharing and grants demand immunity.
* 8.2 - for each printed non-demand ordinal the sharing opponent executes first, then the
  activator, and that ordinal completes fully before the next one starts.
* 8.3 - a demand is executed only by an opponent with strictly fewer featured icons.
  ``PlayerRefKind.ACTIVATOR`` and ``EXECUTOR`` already give demand text its "I/my" and
  "you/your" pronouns, so there is no special-case code path.
* 8.4 - at most one free Draw, and only when a *shared non-demand* execution produced a
  qualifying change (rules decision 2, which counts reveals and achievement claims).
* 8.5 - partial execution needs no orchestration logic: the interpreter already advances past
  nodes whose selectors are empty.
* Decision 7 - Fission's abort skips all remaining work and the sharing bonus, while the paid
  action stays spent and any second paid action stays available.
"""

from __future__ import annotations

from dataclasses import replace

from innovation_ai.innovation.actions import Decision
from innovation_ai.innovation.board import visible_icons
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import (
    EffectFrameState,
    EffectVariable,
    GamePhase,
    GameState,
)
from innovation_ai.innovation.terminal import apply_terminal, draw_beyond_age_ten_result
from innovation_ai.innovation.types import CardId, DogmaEffectId, Icon, PlayerId
from innovation_ai.innovation.zones import draw_action

from .engine import (
    _bump_step,
    _event,
    _pop,
    _program_frame,
    _push,
    _replace_top,
    _root_context,
    current_effect_decision,
    other_player,
    qualifying_change_count,
    resume_effect,
)
from .model import (
    DOGMA_ACTIVATOR_ICONS,
    DOGMA_FRAME,
    DOGMA_OPPONENT_ICONS,
    EffectContext,
    EffectEvent,
    EffectEventKind,
    EffectInvariantError,
    EffectResolution,
    EffectStatus,
    clear_effect_scope,
    frame_context,
    frame_value,
    frozen_icon_counts,
    get_effect_variable,
    make_frame,
    set_effect_variable,
)
from .program import EffectProgramRegistry

DOGMA_SHARED_CHANGE = "shared_change"
DOGMA_ENTRY_COUNT = "entry_count"
DOGMA_ROOT_SCOPE = "dogma"
_SHARE_BASELINE = "share-baseline"


def _entry_schedule(
    ordinals: tuple[tuple[int, bool], ...],
    *,
    opponent_shares: bool,
    opponent_demand_bound: bool,
) -> tuple[tuple[int, bool, bool], ...]:
    """Return ``(ordinal, executor_is_opponent, shared)`` entries in strict rules order.

    Because each entry is its own step, printed effect *n* completes for both players before
    effect *n+1* begins, which is exactly rule 8.2's "complete that effect fully".
    """

    entries: list[tuple[int, bool, bool]] = []
    for ordinal, demand in ordinals:
        if demand:
            if opponent_demand_bound:
                entries.append((ordinal, True, False))
            continue
        if opponent_shares:
            entries.append((ordinal, True, True))
        entries.append((ordinal, False, False))
    return tuple(entries)


def dogma_schedule(
    state: GameState, frame: EffectFrameState, programs: EffectProgramRegistry
) -> tuple[tuple[int, bool, bool], ...]:
    """Recompute the frozen execution schedule for a paused dogma frame."""

    context = frame_context(frame)
    program = programs.program_for_card(context.source_card_id)
    ordinals = tuple((effect.effect_id.ordinal, effect.demand) for effect in program.effects)
    activator_icons = _integer(frame, DOGMA_ACTIVATOR_ICONS)
    opponent_icons = _integer(frame, DOGMA_OPPONENT_ICONS)
    return _entry_schedule(
        ordinals,
        opponent_shares=opponent_icons >= activator_icons,
        opponent_demand_bound=opponent_icons < activator_icons,
    )


def _integer(frame: EffectFrameState, name: str) -> int:
    value = frame_value(frame, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EffectInvariantError(f"dogma frame field {name!r} is not an integer")
    return value


def _flag(frame: EffectFrameState, name: str, default: bool = False) -> bool:
    value = frame_value(frame, name, default)
    if not isinstance(value, bool):
        raise EffectInvariantError(f"dogma frame field {name!r} is not a boolean")
    return value


def dogma_context(
    state: GameState,
    card_id: CardId,
    activator: PlayerId,
    dogma_action_id: int,
) -> EffectContext:
    """Build the root causal context for one paid Dogma action."""

    return EffectContext(
        actor=activator,
        chooser=activator,
        executor=activator,
        dogma_activator=activator,
        source_card_id=card_id,
        source_effect_id=None,
        turn_id=state.turn_number,
        dogma_action_id=dogma_action_id,
        scope=DOGMA_ROOT_SCOPE,
    )


def start_dogma(
    state: GameState,
    card_id: CardId,
    activator: PlayerId,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
    *,
    dogma_action_id: int | None = None,
    pause_before_first_step: bool = False,
) -> EffectResolution:
    """Install the dogma orchestration frame with frozen featured-icon counts.

    A missing card program raises ``UnimplementedCardError`` from the registry rather than acting
    as a no-op, so an unimplemented card can never be mistaken for a legal do-nothing dogma.
    """

    registry = registry or load_card_registry()
    if state.phase is not GamePhase.PLAY:
        raise EffectInvariantError("a dogma action requires the play phase")
    if state.pending_effects or state.effect_variables or state.revealed:
        raise EffectInvariantError("a root dogma action requires empty effect runtime")
    programs.program_for_card(card_id)
    card = registry.card(card_id)
    opponent = other_player(activator)
    featured = card.featured_icon
    # Rule 8.1: freeze both counts now, as integers, and never recount.
    activator_icons = visible_icons(state.player(activator).board, registry)[featured]
    opponent_icons = visible_icons(state.player(opponent).board, registry)[featured]
    action_id = dogma_action_id if dogma_action_id is not None else state.next_dogma_action_id
    context = dogma_context(state, card_id, activator, action_id)
    frame = make_frame(
        DOGMA_FRAME,
        context,
        activator=activator.value,
        dogma_action=action_id,
        featured_icon=featured.value,
        activator_icons=activator_icons,
        opponent_icons=opponent_icons,
        shared_change=False,
    )
    started = _push(state, frame)
    root = _root_context(context)
    started = set_effect_variable(started, root, "step-count", 0)
    started = set_effect_variable(started, root, "qualifying-change-count", 0)
    started = set_effect_variable(started, root, "nested-count", 0)
    if pause_before_first_step:
        return EffectResolution(started, EffectStatus.CONTINUE)
    return resume_effect(started, programs, registry)


def step_dogma(
    state: GameState,
    frame: EffectFrameState,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
) -> EffectResolution:
    """Advance the dogma step machine by exactly one scheduled entry.

    ``frame.step`` indexes the frozen schedule, then the sharing bonus, then the final pop, so a
    checkpoint exists between every executor and between every printed ordinal.
    """

    registry = registry or load_card_registry()
    context = frame_context(frame)
    state = _bump_step(state, context)
    frame = state.pending_effects[-1]
    schedule = dogma_schedule(state, frame, programs)
    activator = context.dogma_activator
    opponent = other_player(activator)
    program = programs.program_for_card(context.source_card_id)

    if frame.step > 0 and frame.step <= len(schedule):
        # The previous entry has finished; credit a shared change exactly once.
        state = _record_shared_change(state, frame, schedule[frame.step - 1], context)
        frame = state.pending_effects[-1]

    if frame.step < len(schedule):
        ordinal, executor_is_opponent, shared = schedule[frame.step]
        executor = opponent if executor_is_opponent else activator
        entry_context = replace(
            context,
            actor=executor,
            chooser=executor,
            executor=executor,
            shared=shared,
            scope=f"{DOGMA_ROOT_SCOPE}/entry-{frame.step + 1}",
        )
        baseline = qualifying_change_count(state, context)
        updated = _replace_top(state, replace(frame, step=frame.step + 1))
        updated = set_effect_variable(updated, entry_context, _SHARE_BASELINE, baseline)
        updated = _push(
            updated,
            _program_frame(
                program,
                entry_context,
                non_demand_only=False,
                selected_effect_ordinal=ordinal,
            ),
        )
        return EffectResolution(updated, EffectStatus.CONTINUE)

    if frame.step == len(schedule):
        updated = _replace_top(state, replace(frame, step=frame.step + 1))
        if not _flag(frame, DOGMA_SHARED_CHANGE):
            return EffectResolution(updated, EffectStatus.CONTINUE)
        # Rule 8.4: exactly one free Draw, and it is not one of the turn's two actions, so no
        # paid-action counter changes here.
        updated, change, result = draw_action(updated, activator, registry)
        events: list[EffectEvent] = []
        if change.changed:
            updated, event = _event(updated, context, EffectEventKind.CHANGE, change=change)
            events.append(event)
        if result.beyond_age_ten:
            terminal = draw_beyond_age_ten_result(updated, registry)
            changes = qualifying_change_count(updated, context)
            updated = apply_terminal(updated, terminal)
            return EffectResolution(
                updated,
                EffectStatus.TERMINAL,
                events=tuple(events),
                qualifying_changes=changes,
            )
        return EffectResolution(updated, EffectStatus.CONTINUE, events=tuple(events))

    changes = qualifying_change_count(state, context)
    updated = _pop(state)
    updated = clear_effect_scope(updated, _root_context(context))
    return EffectResolution(updated, EffectStatus.COMPLETE, qualifying_changes=changes)


def _record_shared_change(
    state: GameState,
    frame: EffectFrameState,
    entry: tuple[int, bool, bool],
    context: EffectContext,
) -> GameState:
    """Credit the sharing bonus when a shared non-demand execution changed the game.

    Only a *shared* entry counts, so a demand never earns the activator a free Draw and the
    activator's own execution never credits itself.
    """

    _, _, shared = entry
    if not shared or _flag(frame, DOGMA_SHARED_CHANGE):
        return state
    entry_context = replace(context, scope=f"{DOGMA_ROOT_SCOPE}/entry-{frame.step}")
    baseline = get_effect_variable(state, entry_context, _SHARE_BASELINE, 0)
    if not isinstance(baseline, int) or isinstance(baseline, bool):
        raise EffectInvariantError("serialized share baseline is invalid")
    if qualifying_change_count(state, context) <= baseline:
        return state
    variables = tuple(item for item in frame.variables if item.name != DOGMA_SHARED_CHANGE)
    replacement = replace(
        frame,
        variables=tuple(
            sorted(
                (*variables, EffectVariable(DOGMA_SHARED_CHANGE, True)),
                key=lambda item: item.name,
            )
        ),
    )
    return _replace_top(state, replacement)


def current_dogma_decision(
    state: GameState,
    programs: EffectProgramRegistry,
    registry: CardRegistry | None = None,
) -> Decision | None:
    """Return the current mid-dogma decision, if any."""

    return current_effect_decision(state, programs, registry)


def dogma_is_pending(state: GameState) -> bool:
    """Whether a dogma orchestration frame is still on the stack."""

    return any(frame.kind == DOGMA_FRAME for frame in state.pending_effects)


def dogma_effect_ids(card_id: CardId, programs: EffectProgramRegistry) -> tuple[DogmaEffectId, ...]:
    """Return the printed effect identities a card's program implements, in order."""

    return tuple(effect.effect_id for effect in programs.program_for_card(card_id).effects)


def frozen_featured_icon(state: GameState) -> Icon | None:
    """Return the current dogma action's frozen featured icon, if one is pending."""

    frozen = frozen_icon_counts(state)
    return None if frozen is None else frozen[0]


__all__ = [
    "DOGMA_FRAME",
    "DOGMA_ROOT_SCOPE",
    "DOGMA_SHARED_CHANGE",
    "current_dogma_decision",
    "dogma_context",
    "dogma_effect_ids",
    "dogma_is_pending",
    "dogma_schedule",
    "frozen_featured_icon",
    "start_dogma",
    "step_dogma",
]
