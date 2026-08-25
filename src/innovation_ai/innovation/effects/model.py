"""Serializable effect execution contracts and causal provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum

from innovation_ai.innovation.actions import Decision, SemanticAction
from innovation_ai.innovation.errors import IllegalAction
from innovation_ai.innovation.state import (
    EffectFrameState,
    EffectVariable,
    GameState,
    StateValue,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
)
from innovation_ai.innovation.zones import ChangeRecord

EFFECT_RUNTIME_SCHEMA_VERSION = 2
EFFECT_EVENT_SCHEMA_VERSION = 3
_SCOPE_PART = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

DOGMA_FRAME = "dogma-action"
"""Frame kind installed by the paid Dogma action and stepped by WP5's orchestrator."""

DOGMA_FEATURED_ICON = "featured_icon"
DOGMA_ACTIVATOR_ICONS = "activator_icons"
DOGMA_OPPONENT_ICONS = "opponent_icons"


def frozen_icon_counts(state: GameState) -> tuple[Icon, int, int] | None:
    """Return the dogma action's frozen ``(featured icon, activator, opponent)`` counts.

    The counts are frozen once at the start of the Dogma action (rules 8.1) and stored in the
    orchestration frame, so this is a pure read: splaying mid-dogma cannot change them.
    """

    for frame in state.pending_effects:
        if frame.kind != DOGMA_FRAME:
            continue
        icon = frame_value(frame, DOGMA_FEATURED_ICON)
        activator = frame_value(frame, DOGMA_ACTIVATOR_ICONS)
        opponent = frame_value(frame, DOGMA_OPPONENT_ICONS)
        if (
            isinstance(icon, str)
            and isinstance(activator, int)
            and not isinstance(activator, bool)
            and isinstance(opponent, int)
            and not isinstance(opponent, bool)
        ):
            return Icon(icon), activator, opponent
    return None


class VariableScope(StrEnum):
    """Scope used for values that must survive one printed-effect boundary.

    ``CARD_EXECUTION`` persists across the printed ordinals of one card execution. For a paid
    Dogma action that means the dogma root shared by its scheduled entries; for nested execution
    it means only the innermost ``nested-N`` program scope, so skipped demands cannot read an
    unrelated outer card's causal result.
    """

    LOCAL = "local"
    CARD_EXECUTION = "card-execution"
    ROOT = "root"


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
    ACHIEVEMENT = "achievement"
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
        """Derive an isolated nested scope while preserving outer shared attribution."""

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


def variable_context(context: EffectContext, scope: VariableScope) -> EffectContext:
    """Return the context that owns one persisted declarative value.

    Direct scheduled ordinals communicate through the dogma root. Nested cards instead use their
    innermost ``nested-N`` scope, isolating causal results from outer and sibling executions.
    """

    if scope is VariableScope.LOCAL:
        return context
    if scope is VariableScope.ROOT or not context.nested:
        return replace(context, scope=context.scope.split("/", maxsplit=1)[0])
    parts = context.scope.split("/")
    nested_index = max(
        (index for index, part in enumerate(parts) if part.startswith("nested-")),
        default=-1,
    )
    if nested_index < 0:
        raise EffectInvariantError("nested context has no nested execution scope")
    return replace(context, scope="/".join(parts[: nested_index + 1]))


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
    revealed_colors: tuple[Color, ...] = ()
    achievement_player: PlayerId | None = None
    achievement_id: NormalAchievementId | SpecialAchievementId | None = None
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
        if len(set(self.revealed_colors)) != len(self.revealed_colors):
            raise ValueError("event revealed colours cannot contain duplicates")
        if self.revealed_colors and self.kind is not EffectEventKind.REVEAL:
            raise ValueError("only a reveal event carries public colours")
        has_achievement = self.achievement_player is not None or self.achievement_id is not None
        if has_achievement != (self.kind is EffectEventKind.ACHIEVEMENT):
            raise ValueError("only an achievement event carries claim provenance")
        if (self.achievement_player is None) != (self.achievement_id is None):
            raise ValueError("achievement player and ID must be supplied together")

    @property
    def changed(self) -> bool:
        """Whether this event represents a sharing-qualifying gameplay change."""

        if self.kind in {EffectEventKind.REVEAL, EffectEventKind.ACHIEVEMENT}:
            return True
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


def validate_effect_runtime_structure(state: GameState) -> None:
    """Reject malformed serialized frame stacks before the VM attempts to execute them."""

    if not state.pending_effects:
        if state.effect_variables or state.revealed:
            raise ValueError("effect variables and reveals require pending frames")
        return

    known_kinds = {DOGMA_FRAME, "effect-program", "effect-node"}
    root_scopes: set[str] = set()
    live_roots: set[str] = set()
    for frame in state.pending_effects:
        if frame.kind not in known_kinds:
            raise ValueError(f"unknown effect frame kind: {frame.kind!r}")
        context = frame_context(frame)
        root_scope = context.scope.split("/", maxsplit=1)[0]
        live_roots.add(root_scope)
        nested_depth = frame_value(frame, "nested_depth", 0)
        if not isinstance(nested_depth, int) or isinstance(nested_depth, bool) or nested_depth < 0:
            raise ValueError("effect frame has an invalid nested depth")
        if frame.kind == DOGMA_FRAME:
            root_scopes.add(root_scope)
            featured = frame_value(frame, DOGMA_FEATURED_ICON)
            activator_icons = frame_value(frame, DOGMA_ACTIVATOR_ICONS)
            opponent_icons = frame_value(frame, DOGMA_OPPONENT_ICONS)
            shared_change = frame_value(frame, "shared_change")
            if not isinstance(featured, str):
                raise ValueError("dogma frame is missing its featured icon")
            Icon(featured)
            for name, value in (
                (DOGMA_ACTIVATOR_ICONS, activator_icons),
                (DOGMA_OPPONENT_ICONS, opponent_icons),
            ):
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"dogma frame field {name!r} is invalid")
            if not isinstance(shared_change, bool):
                raise ValueError("dogma frame is missing its shared-change flag")
            continue
        program_id = frame_value(frame, "program_id")
        if not isinstance(program_id, str) or not program_id:
            raise ValueError("effect frame is missing its program ID")
        if frame.kind == "effect-program":
            for name in ("non_demand_only", "root_program"):
                if not isinstance(frame_value(frame, name), bool):
                    raise ValueError(f"program frame field {name!r} is invalid")
            ordinal = frame_value(frame, "selected_effect_ordinal")
            if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
                raise ValueError("program frame has an invalid selected effect ordinal")
            if frame_value(frame, "root_program") is True:
                root_scopes.add(root_scope)
        else:
            node_id = frame_value(frame, "node_id")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("node frame is missing its node ID")

    if not root_scopes:
        raise ValueError("effect runtime has no root orchestration frame")
    variable_names = {variable.name for variable in state.effect_variables}
    for root_scope in root_scopes:
        required = {
            f"{root_scope}:step-count",
            f"{root_scope}:qualifying-change-count",
            f"{root_scope}:nested-count",
        }
        if not required <= variable_names:
            raise ValueError("root effect runtime is missing serialized VM counters")
    for marker in state.revealed:
        if marker.scope.split("/", maxsplit=1)[0] not in live_roots:
            raise ValueError("reveal marker belongs to a dead effect scope")


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
    validate_effect_runtime_structure(restored)
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
