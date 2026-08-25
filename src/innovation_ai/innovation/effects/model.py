"""Serializable effect execution contracts and causal provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum

from innovation_ai.innovation.actions import Decision, SemanticAction
from innovation_ai.innovation.protocol import IllegalAction
from innovation_ai.innovation.state import (
    EffectFrameState,
    EffectVariable,
    GameState,
    StateValue,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, PlayerId
from innovation_ai.innovation.zones import ChangeRecord

EFFECT_RUNTIME_SCHEMA_VERSION = 1
EFFECT_EVENT_SCHEMA_VERSION = 1
_SCOPE_PART = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class EffectStatus(StrEnum):
    """Stable outcomes from advancing the explicit effect stack."""

    CONTINUE = "continue"
    AWAIT_DECISION = "await-decision"
    COMPLETE = "complete"
    ABORT_DOGMA = "abort-dogma"
    TERMINAL = "terminal"


class EffectEventKind(StrEnum):
    """Kinds of causal events emitted by shared effect primitives."""

    CHANGE = "change"
    REVEAL = "reveal"
    KEEP = "keep"
    ABORT_DOGMA = "abort-dogma"


class EffectRuntimeError(RuntimeError):
    """Base class for invalid or inconsistent effect execution."""


class EffectInvariantError(EffectRuntimeError):
    """The effect VM encountered an invalid frame or impossible program state."""


class IllegalEffectAction(IllegalAction, EffectRuntimeError):
    """An action was not legal for the currently paused effect decision."""

    def __init__(self, action: SemanticAction, decision: Decision) -> None:
        super().__init__(action, decision)


@dataclass(frozen=True, slots=True)
class EffectContext:
    """Causal identities and flags inherited by every nested primitive."""

    actor: PlayerId
    chooser: PlayerId
    executor: PlayerId
    dogma_activator: PlayerId
    source_card_id: CardId
    source_effect_id: DogmaEffectId | None
    turn_id: int
    dogma_action_id: int
    scope: str = "root"
    demand: bool = False
    shared: bool = False
    nested: bool = False
    step_limit: int = 10_000

    def __post_init__(self) -> None:
        if self.turn_id < 0:
            raise ValueError("turn ID cannot be negative")
        if self.dogma_action_id < 1:
            raise ValueError("dogma action ID must be positive")
        if (
            self.source_effect_id is not None
            and self.source_effect_id.card_id != self.source_card_id
        ):
            raise ValueError("source effect must belong to the source card")
        if self.step_limit < 1:
            raise ValueError("effect step limit must be positive")
        if any(_SCOPE_PART.fullmatch(part) is None for part in self.scope.split("/")):
            raise ValueError(f"invalid effect scope: {self.scope!r}")

    def for_effect(self, effect_id: DogmaEffectId, *, demand: bool) -> EffectContext:
        """Return context for one printed effect while preserving outer causality."""

        return replace(
            self,
            actor=self.executor,
            chooser=self.executor,
            source_card_id=effect_id.card_id,
            source_effect_id=effect_id,
            scope=f"{self.scope}/effect-{effect_id.ordinal}",
            demand=demand,
        )

    def for_nested(self, source_card_id: CardId, suffix: str) -> EffectContext:
        """Derive an isolated nested scope with sharing/demand disabled."""

        if _SCOPE_PART.fullmatch(suffix) is None:
            raise ValueError(f"invalid nested scope suffix: {suffix!r}")
        return replace(
            self,
            actor=self.executor,
            chooser=self.executor,
            source_card_id=source_card_id,
            source_effect_id=None,
            scope=f"{self.scope}/{suffix}",
            demand=False,
            nested=True,
        )


@dataclass(frozen=True, slots=True)
class EffectEvent:
    """One versioned event tying a reveal or state change to its full cause."""

    event_id: int
    kind: EffectEventKind
    actor: PlayerId
    chooser: PlayerId
    executor: PlayerId
    dogma_activator: PlayerId
    source_card_id: CardId
    source_effect_id: DogmaEffectId | None
    turn_id: int
    dogma_action_id: int
    demand: bool
    shared: bool
    nested: bool
    change: ChangeRecord | None = None
    card_ids: tuple[CardId, ...] = ()
    atomic_group_id: int | None = None
    schema_version: int = EFFECT_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.event_id < 1:
            raise ValueError("event ID must be positive")
        if self.dogma_action_id < 1:
            raise ValueError("dogma action ID must be positive")
        if (
            self.source_effect_id is not None
            and self.source_effect_id.card_id != self.source_card_id
        ):
            raise ValueError("event source effect must belong to its card")
        if self.atomic_group_id is not None and self.atomic_group_id < 1:
            raise ValueError("atomic group ID must be positive")
        if len(set(self.card_ids)) != len(self.card_ids):
            raise ValueError("event card IDs cannot contain duplicates")

    @property
    def changed(self) -> bool:
        """Whether this event is a qualifying physical or geometry change."""

        return self.change is not None and self.change.changed


@dataclass(frozen=True, slots=True)
class EffectResolution:
    """Result of one VM advance or one submitted effect choice."""

    state: GameState
    status: EffectStatus
    decision: Decision | None = None
    events: tuple[EffectEvent, ...] = ()
    qualifying_changes: int = 0

    def __post_init__(self) -> None:
        if (self.status is EffectStatus.AWAIT_DECISION) != (self.decision is not None):
            raise ValueError("only an awaiting resolution may contain a decision")
        if self.status is EffectStatus.CONTINUE and not self.state.pending_effects:
            raise ValueError("a continuing resolution requires pending frames")
        if self.qualifying_changes < 0:
            raise ValueError("qualifying change count cannot be negative")


class ScopedVariables:
    """Immutable helper for namespaced serializable effect variables."""

    def __init__(self, variables: tuple[EffectVariable, ...]) -> None:
        names = tuple(variable.name for variable in variables)
        if len(set(names)) != len(names):
            raise EffectInvariantError("effect variable names must be unique")
        self._variables = variables

    @staticmethod
    def key(scope: str, name: str) -> str:
        if not name or "/" in name or ":" in name:
            raise ValueError(f"invalid scoped variable name: {name!r}")
        return f"{scope}:{name}"

    def get(self, scope: str, name: str, default: StateValue = None) -> StateValue:
        key = self.key(scope, name)
        return next((item.value for item in self._variables if item.name == key), default)

    def set(self, scope: str, name: str, value: StateValue) -> tuple[EffectVariable, ...]:
        key = self.key(scope, name)
        replacement = EffectVariable(key, value)
        found = False
        result: list[EffectVariable] = []
        for item in self._variables:
            if item.name == key:
                result.append(replacement)
                found = True
            else:
                result.append(item)
        if not found:
            result.append(replacement)
        return tuple(sorted(result, key=lambda item: item.name))

    def delete(self, scope: str, name: str) -> tuple[EffectVariable, ...]:
        key = self.key(scope, name)
        return tuple(item for item in self._variables if item.name != key)

    def clear_scope(self, scope: str) -> tuple[EffectVariable, ...]:
        prefix = f"{scope}:"
        child_prefix = f"{scope}/"
        return tuple(
            item
            for item in self._variables
            if not item.name.startswith(prefix) and not item.name.startswith(child_prefix)
        )


def get_effect_variable(
    state: GameState, context: EffectContext, name: str, default: StateValue = None
) -> StateValue:
    """Read one variable from the context's isolated scope."""

    return ScopedVariables(state.effect_variables).get(context.scope, name, default)


def set_effect_variable(
    state: GameState, context: EffectContext, name: str, value: StateValue
) -> GameState:
    """Return state with one scoped serializable value replaced."""

    variables = ScopedVariables(state.effect_variables).set(context.scope, name, value)
    return replace(state, effect_variables=variables)


def delete_effect_variable(state: GameState, context: EffectContext, name: str) -> GameState:
    """Return state without one value from the context's scope."""

    variables = ScopedVariables(state.effect_variables).delete(context.scope, name)
    return replace(state, effect_variables=variables)


def clear_effect_scope(state: GameState, context: EffectContext) -> GameState:
    """Remove a scope and all nested child scopes."""

    variables = ScopedVariables(state.effect_variables).clear_scope(context.scope)
    return replace(state, effect_variables=variables)


def _context_variables(context: EffectContext, **extra: StateValue) -> tuple[EffectVariable, ...]:
    effect_ordinal = context.source_effect_id.ordinal if context.source_effect_id is not None else 0
    values: dict[str, StateValue] = {
        "actor": context.actor.value,
        "chooser": context.chooser.value,
        "executor": context.executor.value,
        "dogma_activator": context.dogma_activator.value,
        "source_effect_ordinal": effect_ordinal,
        "turn_id": context.turn_id,
        "dogma_action_id": context.dogma_action_id,
        "scope": context.scope,
        "demand": context.demand,
        "shared": context.shared,
        "nested": context.nested,
        "step_limit": context.step_limit,
    }
    overlap = set(values) & set(extra)
    if overlap:
        raise ValueError(f"frame variables override reserved context fields: {sorted(overlap)}")
    values.update(extra)
    return tuple(EffectVariable(name, value) for name, value in sorted(values.items()))


def make_frame(
    kind: str,
    context: EffectContext,
    *,
    step: int = 0,
    **variables: StateValue,
) -> EffectFrameState:
    """Build a generic serializable frame containing an explicit context."""

    return EffectFrameState(
        kind,
        step=step,
        source_card_id=context.source_card_id,
        variables=_context_variables(context, **variables),
    )


def frame_value(frame: EffectFrameState, name: str, default: StateValue = None) -> StateValue:
    """Read one field from a generic frame."""

    return next((item.value for item in frame.variables if item.name == name), default)


def frame_context(frame: EffectFrameState) -> EffectContext:
    """Restore the typed causal context embedded in a frame."""

    if frame.source_card_id is None:
        raise EffectInvariantError("effect frame is missing its source card")

    def text(name: str) -> str:
        value = frame_value(frame, name)
        if not isinstance(value, str):
            raise EffectInvariantError(f"effect frame field {name!r} is not text")
        return value

    def integer(name: str) -> int:
        value = frame_value(frame, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise EffectInvariantError(f"effect frame field {name!r} is not an integer")
        return value

    def flag(name: str) -> bool:
        value = frame_value(frame, name)
        if not isinstance(value, bool):
            raise EffectInvariantError(f"effect frame field {name!r} is not a boolean")
        return value

    ordinal = integer("source_effect_ordinal")
    effect_id = DogmaEffectId(frame.source_card_id, ordinal) if ordinal else None
    return EffectContext(
        actor=PlayerId(text("actor")),
        chooser=PlayerId(text("chooser")),
        executor=PlayerId(text("executor")),
        dogma_activator=PlayerId(text("dogma_activator")),
        source_card_id=frame.source_card_id,
        source_effect_id=effect_id,
        turn_id=integer("turn_id"),
        dogma_action_id=integer("dogma_action_id"),
        scope=text("scope"),
        demand=flag("demand"),
        shared=flag("shared"),
        nested=flag("nested"),
        step_limit=integer("step_limit"),
    )


def _payload_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, CardId):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _payload_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_payload_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"effect contract contains non-serializable value: {type(value).__name__}")


def effect_event_payload(event: EffectEvent) -> dict[str, object]:
    """Return a canonical JSON-compatible provenance event."""

    payload = _payload_value(event)
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise TypeError("effect event did not serialize to an object")
    return payload


def effect_runtime_payload(state: GameState) -> dict[str, object]:
    """Serialize only the resumable VM fields owned by WP4."""

    return {
        "schema_version": EFFECT_RUNTIME_SCHEMA_VERSION,
        "pending_effects": _payload_value(state.pending_effects),
        "effect_variables": _payload_value(state.effect_variables),
    }


def _state_value(value: object) -> StateValue:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, list):
        return tuple(_state_value(item) for item in value)
    raise ValueError(f"invalid serialized effect value: {value!r}")


def restore_effect_runtime(
    state: GameState,
    payload: dict[str, object],
    programs: object | None = None,
) -> GameState:
    """Restore VM frames/variables from a JSON-decoded runtime payload."""

    if set(payload) != {"schema_version", "pending_effects", "effect_variables"}:
        raise ValueError("effect runtime payload has missing or unknown fields")
    if payload.get("schema_version") != EFFECT_RUNTIME_SCHEMA_VERSION:
        raise ValueError("unsupported effect runtime schema version")
    raw_frames = payload.get("pending_effects")
    raw_variables = payload.get("effect_variables")
    if not isinstance(raw_frames, list) or not isinstance(raw_variables, list):
        raise ValueError("effect runtime payload must contain frame and variable lists")

    frames: list[EffectFrameState] = []
    for raw in raw_frames:
        if not isinstance(raw, dict):
            raise ValueError("effect frame payload must be an object")
        if set(raw) != {"kind", "step", "source_card_id", "variables"}:
            raise ValueError("effect frame payload has missing or unknown fields")
        kind = raw["kind"]
        step = raw["step"]
        source = raw["source_card_id"]
        variables = raw["variables"]
        if not isinstance(kind, str) or not kind:
            raise ValueError("effect frame kind must be non-empty text")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("effect frame step must be a non-negative integer")
        if source is not None and not isinstance(source, str):
            raise ValueError("effect frame source card must be text or null")
        if not isinstance(variables, list):
            raise ValueError("effect frame variables must be a list")
        parsed_variables: list[EffectVariable] = []
        for item in variables:
            if not isinstance(item, dict) or set(item) != {"name", "value"}:
                raise ValueError("effect frame variable must contain name and value")
            if not isinstance(item["name"], str):
                raise ValueError("effect frame variable name must be text")
            parsed_variables.append(EffectVariable(item["name"], _state_value(item["value"])))
        frames.append(
            EffectFrameState(
                kind,
                step,
                CardId(source) if source is not None else None,
                tuple(parsed_variables),
            )
        )
    parsed_effect_variables: list[EffectVariable] = []
    for item in raw_variables:
        if not isinstance(item, dict) or set(item) != {"name", "value"}:
            raise ValueError("effect variable must contain name and value")
        if not isinstance(item["name"], str):
            raise ValueError("effect variable name must be text")
        parsed_effect_variables.append(EffectVariable(item["name"], _state_value(item["value"])))
    restored = replace(
        state,
        pending_effects=tuple(frames),
        effect_variables=tuple(parsed_effect_variables),
    )
    variable_names = {variable.name for variable in restored.effect_variables}
    for frame in restored.pending_effects:
        if frame.kind != "effect-program" or frame_value(frame, "root_program") is not True:
            continue
        context = frame_context(frame)
        root_scope = context.scope.split("/", maxsplit=1)[0]
        required = {
            f"{root_scope}:step-count",
            f"{root_scope}:qualifying-change-count",
            f"{root_scope}:nested-count",
        }
        if not required <= variable_names:
            raise ValueError("root effect runtime is missing serialized VM counters")
    if programs is not None:
        from innovation_ai.innovation.effects.program import EffectProgramRegistry

        if not isinstance(programs, EffectProgramRegistry):
            raise TypeError("programs must be an EffectProgramRegistry")
        for frame in restored.pending_effects:
            if frame.kind not in {"effect-program", "effect-node"}:
                continue
            program_id = frame_value(frame, "program_id")
            if not isinstance(program_id, str):
                raise ValueError("restored effect frame is missing a program ID")
            program = programs.program(program_id)
            if frame.kind == "effect-node":
                node_id = frame_value(frame, "node_id")
                if not isinstance(node_id, str):
                    raise ValueError("restored node frame is missing a node ID")
                program.node(node_id)
    return restored
