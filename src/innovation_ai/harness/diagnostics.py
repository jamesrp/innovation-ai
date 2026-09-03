"""Deterministic trusted-private diagnostic traces for pathological games.

The trace format intentionally sits outside player-safe logs and training data.  It records exact
semantic actions, legal sets, policy/search telemetry, and (only behind a second explicit marker)
authoritative snapshots.  A separate redactor emits aggregate-only summaries.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, fields
from io import BytesIO
from pathlib import Path
from typing import cast

from innovation_ai.harness.policy_scheduler import PolicyDecisionAudit
from innovation_ai.innovation.actions import (
    ActionKind,
    Decision,
    DecisionKind,
    SemanticAction,
    action_payload,
    decision_payload,
)
from innovation_ai.innovation.serialization import (
    SerializationError,
    action_from_payload,
    state_from_payload,
    terminal_from_payload,
    terminal_payload,
)
from innovation_ai.innovation.state import (
    GameState,
    TerminalResult,
    state_hash,
    state_payload,
)
from innovation_ai.innovation.strategic import strategic_state_digest
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

DIAGNOSTIC_TRACE_FORMAT = "innovation-ai-private-diagnostic-trace"
DIAGNOSTIC_TRACE_SCHEMA_VERSION = 1
DIAGNOSTIC_TRACE_PRIVACY = "trusted-private"
DIAGNOSTIC_REDACTED_FORMAT = "innovation-ai-redacted-diagnostic-summary"
DIAGNOSTIC_REDACTED_SCHEMA_VERSION = 1
DEFAULT_REPEATED_PAID_ACTION_WINDOW = 4
FIXED_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
_SHA256_PREFIX = "sha256:"
_REQUIRED_VERSION_KEYS = frozenset({"policy", "checkpoint", "fallback", "search", "sampler"})
_ALLOWED_HANDLINGS = frozenset(
    {
        "learned",
        "heuristic",
        "baseline",
        "sampler-fallback",
        "evaluator-fallback",
        "search",
        "learned-search-fallback",
    }
)

type DiagnosticJsonScalar = str | int | float | bool | None
type DiagnosticJsonValue = (
    DiagnosticJsonScalar | list["DiagnosticJsonValue"] | dict[str, "DiagnosticJsonValue"]
)
type DiagnosticJsonObject = dict[str, DiagnosticJsonValue]


class DiagnosticTraceError(ValueError):
    """A private diagnostic trace is malformed or internally inconsistent."""


class DiagnosticTraceSchemaError(DiagnosticTraceError):
    """A trace record differs from the exact supported schema."""


@dataclass(frozen=True, slots=True)
class DiagnosticTraceHeader:
    """Identity and provenance for one trusted-private trace."""

    source_revision: str
    game_id: str
    setup_id: str
    setup_seed: int
    manifest_digest: str
    config_digest: str
    versions: Mapping[str, str | None]
    rng_seed_digests: Mapping[str, str]
    initial_state_hash: str
    initial_strategic_digest: str
    authoritative_snapshots: bool = False
    private_debug: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_revision, "source revision"),
            (self.game_id, "game ID"),
            (self.setup_id, "setup ID"),
        ):
            _nonempty(value, label)
        if isinstance(self.setup_seed, bool) or not isinstance(self.setup_seed, int):
            raise ValueError("setup seed must be an integer")
        _digest(self.manifest_digest, "manifest digest")
        _digest(self.config_digest, "config digest")
        _digest(self.initial_state_hash, "initial state hash")
        _digest(self.initial_strategic_digest, "initial strategic digest")
        versions = dict(self.versions)
        if set(versions) != _REQUIRED_VERSION_KEYS:
            raise ValueError(
                "diagnostic versions must name policy/checkpoint/fallback/search/sampler"
            )
        if any(value is not None and not value for value in versions.values()):
            raise ValueError("diagnostic version values cannot be empty")
        seeds = dict(self.rng_seed_digests)
        if any(not key for key in seeds):
            raise ValueError("RNG seed digest labels cannot be empty")
        for value in seeds.values():
            _digest(value, "RNG seed digest")
        object.__setattr__(self, "versions", versions)
        object.__setattr__(self, "rng_seed_digests", seeds)
        if self.authoritative_snapshots and not self.private_debug:
            raise ValueError("authoritative snapshots require the explicit private-debug marker")

    @classmethod
    def for_state(
        cls,
        state: GameState,
        *,
        source_revision: str,
        game_id: str,
        setup_id: str,
        manifest_digest: str,
        config_digest: str,
        versions: Mapping[str, str | None],
        rng_seed_digests: Mapping[str, str],
        authoritative_snapshots: bool = False,
        private_debug: bool = False,
    ) -> DiagnosticTraceHeader:
        """Build a header whose initial chain markers match ``state``."""

        return cls(
            source_revision=source_revision,
            game_id=game_id,
            setup_id=setup_id,
            setup_seed=state.setup.seed,
            manifest_digest=manifest_digest,
            config_digest=config_digest,
            versions=versions,
            rng_seed_digests=rng_seed_digests,
            initial_state_hash=state_hash(state),
            initial_strategic_digest=strategic_state_digest(state),
            authoritative_snapshots=authoritative_snapshots,
            private_debug=private_debug,
        )


@dataclass(frozen=True, slots=True)
class LearnedDecisionSummary:
    """Complete learned selection values aligned to the decision's stable legal-action order."""

    sample_values: tuple[tuple[float, ...], ...]
    mean_values: tuple[float, ...]
    selector_scores: tuple[float, ...]
    selected_action_index: int
    tied_action_indices: tuple[int, ...]
    margin: float

    def __post_init__(self) -> None:
        count = len(self.mean_values)
        if count < 1 or len(self.sample_values) != count or len(self.selector_scores) != count:
            raise ValueError("learned values must align to a non-empty legal-action set")
        if any(not values for values in self.sample_values):
            raise ValueError("every learned action requires per-determinization values")
        widths = {len(values) for values in self.sample_values}
        if len(widths) != 1:
            raise ValueError("learned actions must have equal determinization counts")
        for value in (*self.mean_values, *self.selector_scores, self.margin):
            _finite(value, "learned value")
        for values in self.sample_values:
            for value in values:
                _finite(value, "learned sample value")
        if not 0 <= self.selected_action_index < count:
            raise ValueError("learned selected action index is out of range")
        if tuple(sorted(set(self.tied_action_indices))) != self.tied_action_indices:
            raise ValueError("learned tied action indices must be unique stable order")
        if any(index < 0 or index >= count for index in self.tied_action_indices):
            raise ValueError("learned tied action index is out of range")
        if self.margin < 0.0:
            raise ValueError("learned margin cannot be negative")
        for actual, values in zip(self.mean_values, self.sample_values, strict=True):
            expected = math.fsum(values) / len(values)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("learned mean does not match its per-determinization values")

    @classmethod
    def from_values(
        cls,
        sample_values: Sequence[Sequence[float]],
        selector_scores: Sequence[float] | None,
        selected_action_index: int,
    ) -> LearnedDecisionSummary:
        """Compute exact means, stable ties, and winner margin from raw values."""

        samples = tuple(tuple(float(value) for value in values) for values in sample_values)
        means = tuple(math.fsum(values) / len(values) if values else math.nan for values in samples)
        scores = (
            means if selector_scores is None else tuple(float(value) for value in selector_scores)
        )
        if scores:
            best = max(scores)
            tied = tuple(index for index, value in enumerate(scores) if value == best)
            distinct = sorted(set(scores), reverse=True)
            margin = best - distinct[1] if len(distinct) > 1 else 0.0
        else:
            tied = ()
            margin = 0.0
        return cls(samples, means, scores, selected_action_index, tied, margin)


@dataclass(frozen=True, slots=True)
class CardMovement:
    """One authoritative card-zone change (private because it includes identity)."""

    card_id: CardId
    before_zone: str
    after_zone: str


@dataclass(frozen=True, slots=True)
class SplayChange:
    player: PlayerId
    color: Color
    before: str
    after: str

    def __post_init__(self) -> None:
        SplayDirection(self.before)
        SplayDirection(self.after)
        if self.before == self.after:
            raise ValueError("a splay change must change direction")


@dataclass(frozen=True, slots=True)
class AchievementChange:
    player: PlayerId
    achievement_id: str
    achievement_kind: str

    def __post_init__(self) -> None:
        _nonempty(self.achievement_id, "achievement ID")
        if self.achievement_kind not in {"normal", "special"}:
            raise ValueError("achievement kind must be normal or special")


@dataclass(frozen=True, slots=True)
class SupplyChange:
    age: int
    before_count: int
    after_count: int
    before_top: CardId | None
    after_top: CardId | None

    def __post_init__(self) -> None:
        if not 1 <= self.age <= 10:
            raise ValueError("supply change age must be 1-10")
        if self.before_count < 0 or self.after_count < 0:
            raise ValueError("supply change counts cannot be negative")
        if self.before_count == self.after_count and self.before_top == self.after_top:
            raise ValueError("supply change must alter a count or top card")


@dataclass(frozen=True, slots=True)
class RepeatedPaidActionWindow:
    window_size: int
    patterns: tuple[DiagnosticJsonObject, ...]
    matching_prior_count: int
    repeated: bool

    def __post_init__(self) -> None:
        if self.window_size < 1 or not 1 <= len(self.patterns) <= self.window_size:
            raise ValueError("repeated paid-action window has invalid size")
        if self.matching_prior_count < 0 or self.matching_prior_count >= len(self.patterns):
            raise ValueError("repeated paid-action match count is invalid")
        if self.repeated != (self.matching_prior_count > 0):
            raise ValueError("repeated paid-action flag differs from match count")
        for pattern in self.patterns:
            _validate_action_pattern(pattern)


@dataclass(frozen=True, slots=True)
class NoProgressTelemetry:
    """Authoritative movement diagnostics for one committed semantic action."""

    card_movements: tuple[CardMovement, ...]
    splay_changes: tuple[SplayChange, ...]
    achievements: tuple[AchievementChange, ...]
    supply_changes: tuple[SupplyChange, ...]
    score_count: int
    meld_count: int
    tuck_count: int
    return_count: int
    no_op_dogma: bool
    repeated_paid_action_window: RepeatedPaidActionWindow | None

    def __post_init__(self) -> None:
        counts = (self.score_count, self.meld_count, self.tuck_count, self.return_count)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
        ):
            raise ValueError("no-progress counts must be non-negative integers")
        movement_ids = tuple(item.card_id for item in self.card_movements)
        if len(set(movement_ids)) != len(movement_ids):
            raise ValueError("no-progress card movements cannot repeat a card")


@dataclass(frozen=True, slots=True)
class DiagnosticStep:
    """One committed action and its complete before/after audit."""

    sequence: int
    decision_id: int
    decision_hash: str
    before_state_hash: str
    after_state_hash: str
    before_strategic_digest: str
    after_strategic_digest: str
    chooser: PlayerId
    executor: PlayerId
    dogma_activator: PlayerId | None
    active_player: PlayerId | None
    decision_kind: DecisionKind
    paid_actions_remaining: int
    legal_actions: tuple[DiagnosticJsonObject, ...]
    selected_action: DiagnosticJsonObject
    handling: str
    failure: DiagnosticJsonObject | None
    learned: DiagnosticJsonObject | None
    search: DiagnosticJsonObject | None
    no_progress: NoProgressTelemetry
    before_snapshot: DiagnosticJsonObject | None = None
    after_snapshot: DiagnosticJsonObject | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1 or self.decision_id < 1 or self.paid_actions_remaining < 0:
            raise ValueError("diagnostic step counters are invalid")
        for value, label in (
            (self.decision_hash, "decision hash"),
            (self.before_state_hash, "before state hash"),
            (self.after_state_hash, "after state hash"),
            (self.before_strategic_digest, "before strategic digest"),
            (self.after_strategic_digest, "after strategic digest"),
        ):
            _digest(value, label)
        if not self.legal_actions:
            raise ValueError("diagnostic step requires legal actions")
        try:
            legal = tuple(action_from_payload(item) for item in self.legal_actions)
            selected = action_from_payload(self.selected_action)
        except SerializationError as error:
            raise ValueError(f"diagnostic step contains an invalid action: {error}") from error
        if any(action.decision_id != self.decision_id for action in legal):
            raise ValueError("diagnostic legal action decision IDs differ from the step")
        if selected.decision_id != self.decision_id or selected not in legal:
            raise ValueError("diagnostic selected action is not legal")
        if self.handling not in _ALLOWED_HANDLINGS:
            raise ValueError("diagnostic step has unknown policy handling")
        if (self.handling == "learned") != (self.learned is not None):
            raise ValueError("diagnostic learned telemetry differs from policy handling")
        search_handling = self.handling in {"search", "learned-search-fallback"}
        if search_handling != (self.search is not None):
            raise ValueError("diagnostic search telemetry differs from policy handling")


@dataclass(frozen=True, slots=True)
class DiagnosticTraceFooter:
    """Terminal/failure disposition and integrity seal for preceding records."""

    step_count: int
    final_state_hash: str
    final_strategic_digest: str
    outcome: str
    terminal_result: DiagnosticJsonObject | None
    failure: DiagnosticJsonObject | None
    no_progress_totals: Mapping[str, int]
    records_digest: str

    def __post_init__(self) -> None:
        if self.step_count < 0:
            raise ValueError("diagnostic footer step count cannot be negative")
        _digest(self.final_state_hash, "final state hash")
        _digest(self.final_strategic_digest, "final strategic digest")
        _digest(self.records_digest, "records digest")
        _nonempty(self.outcome, "diagnostic outcome")
        if self.terminal_result is not None and self.failure is not None:
            raise ValueError("diagnostic footer terminal result and failure are mutually exclusive")
        totals = dict(self.no_progress_totals)
        if set(totals) != _TOTAL_KEYS or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in totals.values()
        ):
            raise ValueError("diagnostic no-progress totals are malformed")
        object.__setattr__(self, "no_progress_totals", totals)


@dataclass(frozen=True, slots=True)
class DiagnosticTrace:
    header: DiagnosticTraceHeader
    steps: tuple[DiagnosticStep, ...]
    footer: DiagnosticTraceFooter

    def __post_init__(self) -> None:
        _validate_trace(self)


_TOTAL_KEYS = frozenset(
    {
        "card_movements",
        "splay_changes",
        "achievements",
        "supply_changes",
        "scored",
        "melded",
        "tucked",
        "returned",
        "no_op_dogmas",
        "repeated_paid_actions",
    }
)


class PrivateDiagnosticTraceRecorder:
    """Accumulate a chain from committed ``PolicyDecisionAudit`` submissions."""

    def __init__(self, header: DiagnosticTraceHeader, initial_state: GameState) -> None:
        if state_hash(initial_state) != header.initial_state_hash:
            raise DiagnosticTraceError("header initial state hash does not match initial state")
        if strategic_state_digest(initial_state) != header.initial_strategic_digest:
            raise DiagnosticTraceError(
                "header initial strategic digest does not match initial state"
            )
        if initial_state.setup.seed != header.setup_seed:
            raise DiagnosticTraceError("header setup seed does not match initial state")
        self._header = header
        self._state = initial_state
        self._steps: list[DiagnosticStep] = []
        self._recent_paid_actions: dict[PlayerId, list[SemanticAction]] = {
            player: [] for player in PlayerId
        }
        self._sealed = False

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def steps(self) -> tuple[DiagnosticStep, ...]:
        return tuple(self._steps)

    def record_step(
        self,
        after: GameState,
        decision: Decision,
        audit: PolicyDecisionAudit,
        *,
        learned: LearnedDecisionSummary | None = None,
        no_op_dogma: bool | None = None,
    ) -> DiagnosticStep:
        """Record one already-committed policy audit and resulting state."""

        if self._sealed:
            raise DiagnosticTraceError("cannot record a step after sealing")
        if audit.submission.game_id != self._header.game_id:
            raise DiagnosticTraceError("policy audit belongs to another game")
        if audit.chooser is not decision.chooser:
            raise DiagnosticTraceError("policy audit chooser differs from decision chooser")
        action = audit.submission.action
        if action.decision_id != decision.decision_id or action not in decision.legal_actions:
            raise DiagnosticTraceError("policy audit action does not match the decision")
        if learned is None and audit.selection is not None and audit.selection.action_sample_values:
            learned = LearnedDecisionSummary(
                audit.selection.action_sample_values,
                audit.selection.action_mean_values,
                audit.selection.selector_scores,
                audit.selection.selected_action_index,
                audit.selection.tied_best_action_indices,
                0.0
                if audit.selection.selection_margin is None
                else audit.selection.selection_margin,
            )
        recent_actions = self._recent_paid_actions[decision.chooser]
        step = build_diagnostic_step(
            sequence=len(self._steps) + 1,
            before=self._state,
            after=after,
            decision=decision,
            audit=audit,
            learned=learned,
            recent_paid_actions=recent_actions,
            repeated_window_size=DEFAULT_REPEATED_PAID_ACTION_WINDOW,
            include_snapshots=self._header.authoritative_snapshots,
            no_op_dogma=no_op_dogma,
        )
        self._steps.append(step)
        self._state = after
        if decision.kind is DecisionKind.TURN_ACTION:
            recent_actions.append(action)
            del recent_actions[:-DEFAULT_REPEATED_PAID_ACTION_WINDOW]
        return step

    def finish(
        self,
        outcome: str,
        *,
        terminal_result: TerminalResult | None = None,
        failure: BaseException | None = None,
    ) -> DiagnosticTrace:
        """Seal the in-memory trace.  Failures remain failures and are never changed to draws."""

        if self._sealed:
            raise DiagnosticTraceError("diagnostic trace is already sealed")
        if terminal_result is not None and failure is not None:
            raise DiagnosticTraceError("terminal result and failure are mutually exclusive")
        self._sealed = True
        records = [_header_payload(self._header), *(_step_payload(step) for step in self._steps)]
        footer = DiagnosticTraceFooter(
            step_count=len(self._steps),
            final_state_hash=state_hash(self._state),
            final_strategic_digest=strategic_state_digest(self._state),
            outcome=outcome,
            terminal_result=(
                None
                if terminal_result is None
                else cast(DiagnosticJsonObject, terminal_payload(terminal_result))
            ),
            failure=_failure_payload(failure),
            no_progress_totals=_no_progress_totals(self._steps),
            records_digest=_records_digest(records),
        )
        return DiagnosticTrace(self._header, tuple(self._steps), footer)


DiagnosticTraceRecorder = PrivateDiagnosticTraceRecorder


def build_diagnostic_step(
    *,
    sequence: int,
    before: GameState,
    after: GameState,
    decision: Decision,
    audit: PolicyDecisionAudit,
    learned: LearnedDecisionSummary | None = None,
    recent_paid_actions: Sequence[SemanticAction] = (),
    repeated_window_size: int = DEFAULT_REPEATED_PAID_ACTION_WINDOW,
    include_snapshots: bool = False,
    no_op_dogma: bool | None = None,
) -> DiagnosticStep:
    """Build one trace step, deriving movement telemetry from authoritative states."""

    action = audit.submission.action
    if action.decision_id != decision.decision_id or action not in decision.legal_actions:
        raise DiagnosticTraceError("audit selected action is not a legal action for decision")
    if audit.chooser is not decision.chooser:
        raise DiagnosticTraceError("audit and decision chooser differ")
    legal = tuple(
        cast(DiagnosticJsonObject, action_payload(item)) for item in decision.legal_actions
    )
    selected = cast(DiagnosticJsonObject, action_payload(action))
    learned_payload = _learned_payload(learned, decision, audit)
    search_payload = _search_payload(audit)
    if audit.handling == "learned" and learned is None:
        raise DiagnosticTraceError("learned handling requires complete learned decision telemetry")
    if audit.handling != "learned" and learned is not None:
        raise DiagnosticTraceError("non-learned handling cannot carry learned decision telemetry")
    no_progress = derive_no_progress_telemetry(
        before,
        after,
        decision,
        action,
        recent_paid_actions=recent_paid_actions,
        repeated_window_size=repeated_window_size,
        no_op_dogma=no_op_dogma,
    )
    return DiagnosticStep(
        sequence=sequence,
        decision_id=decision.decision_id,
        decision_hash=_tagged_sha256(
            _canonical_json(cast(DiagnosticJsonValue, decision_payload(decision)))
        ),
        before_state_hash=state_hash(before),
        after_state_hash=state_hash(after),
        before_strategic_digest=strategic_state_digest(before),
        after_strategic_digest=strategic_state_digest(after),
        chooser=decision.chooser,
        executor=decision.executor,
        dogma_activator=decision.dogma_activator,
        active_player=before.active_player,
        decision_kind=decision.kind,
        paid_actions_remaining=before.paid_actions_remaining,
        legal_actions=legal,
        selected_action=selected,
        handling=audit.handling,
        failure=_failure_payload(audit.failure),
        learned=learned_payload,
        search=search_payload,
        no_progress=no_progress,
        before_snapshot=(
            cast(DiagnosticJsonObject, state_payload(before)) if include_snapshots else None
        ),
        after_snapshot=(
            cast(DiagnosticJsonObject, state_payload(after)) if include_snapshots else None
        ),
    )


def derive_no_progress_telemetry(
    before: GameState,
    after: GameState,
    decision: Decision,
    action: SemanticAction,
    *,
    recent_paid_actions: Sequence[SemanticAction] = (),
    repeated_window_size: int = DEFAULT_REPEATED_PAID_ACTION_WINDOW,
    no_op_dogma: bool | None = None,
) -> NoProgressTelemetry:
    """Compare exact zones while avoiding false movement from indices shifting within a zone."""

    if repeated_window_size < 1:
        raise ValueError("repeated paid-action window must be positive")
    before_zones = _card_zones(before)
    after_zones = _card_zones(after)
    movements = tuple(
        CardMovement(card_id, before_zones[card_id], after_zones[card_id])
        for card_id in sorted(before_zones, key=str)
        if before_zones[card_id] != after_zones[card_id]
    )
    splays = tuple(
        SplayChange(player.player_id, left.color, left.splay.value, right.splay.value)
        for player, after_player in zip(before.players, after.players, strict=True)
        for left, right in zip(player.board.stacks, after_player.board.stacks, strict=True)
        if left.splay is not right.splay
    )
    achievements: list[AchievementChange] = []
    for left, right in zip(before.players, after.players, strict=True):
        achievements.extend(
            AchievementChange(left.player_id, item.value, "normal")
            for item in right.normal_achievements
            if item not in left.normal_achievements
        )
        achievements.extend(
            AchievementChange(left.player_id, item.value, "special")
            for item in right.special_achievements
            if item not in left.special_achievements
        )
    supplies = tuple(
        SupplyChange(
            age,
            len(left),
            len(right),
            left[0] if left else None,
            right[0] if right else None,
        )
        for age, (left, right) in enumerate(
            zip(before.supply.piles, after.supply.piles, strict=True), 1
        )
        if left != right
    )
    score_count = sum(item.after_zone.endswith(".score") for item in movements)
    return_count = sum(item.after_zone.startswith("supply.age-") for item in movements)
    board_entries = [item for item in movements if ".board." in item.after_zone]
    positional_tucks = sum(
        not _is_board_top(after, item.card_id, item.after_zone) for item in board_entries
    )
    tuck_delta = _positive_counter_delta(before, after, "tucked")
    score_delta = _positive_counter_delta(before, after, "scored")
    tuck_count = max(positional_tucks, min(len(board_entries), tuck_delta))
    meld_count = max(0, len(board_entries) - tuck_count)
    score_count = max(score_count, score_delta)
    progress = bool(movements or splays or achievements or supplies)
    inferred_no_op = action.kind is ActionKind.DOGMA and not progress and not after.pending_effects
    repeated: RepeatedPaidActionWindow | None = None
    if decision.kind is DecisionKind.TURN_ACTION:
        current = _action_pattern(action)
        prior = tuple(_action_pattern(item) for item in recent_paid_actions[-repeated_window_size:])
        patterns = (*prior, current)[-repeated_window_size:]
        matches = sum(item == current for item in prior)
        repeated = RepeatedPaidActionWindow(
            window_size=repeated_window_size,
            patterns=patterns,
            matching_prior_count=matches,
            repeated=matches > 0,
        )
    return NoProgressTelemetry(
        card_movements=movements,
        splay_changes=splays,
        achievements=tuple(achievements),
        supply_changes=supplies,
        score_count=score_count,
        meld_count=meld_count,
        tuck_count=tuck_count,
        return_count=return_count,
        no_op_dogma=inferred_no_op if no_op_dogma is None else no_op_dogma,
        repeated_paid_action_window=repeated,
    )


def write_diagnostic_trace(path: Path, trace: DiagnosticTrace) -> str:
    """Atomically write canonical JSONL in one fixed-header gzip member."""

    _validate_trace(trace)
    records = [
        _header_payload(trace.header),
        *(_step_payload(step) for step in trace.steps),
        _footer_payload(trace.footer),
    ]
    encoded = "".join(
        _canonical_json(cast(DiagnosticJsonValue, item)) + "\n" for item in records
    ).encode("ascii")
    compressed = _deterministic_gzip(encoded)
    _atomic_write(path, compressed)
    return _tagged_sha256_bytes(compressed)


def read_diagnostic_trace(path: Path) -> DiagnosticTrace:
    """Strictly decode canonical, fixed-gzip JSONL and verify every chain/integrity marker."""

    try:
        compressed = path.read_bytes()
    except OSError as error:
        raise DiagnosticTraceError(f"could not read diagnostic trace {path}: {error}") from error
    raw = _read_fixed_gzip(compressed)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise DiagnosticTraceError("diagnostic JSONL must be canonical ASCII JSON") from error
    if not text or not text.endswith("\n"):
        raise DiagnosticTraceError("diagnostic JSONL must end in exactly one newline")
    lines = text[:-1].split("\n")
    if len(lines) < 2 or any(not line for line in lines):
        raise DiagnosticTraceError("diagnostic JSONL records are missing or empty")
    records: list[DiagnosticJsonObject] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise DiagnosticTraceError(f"invalid diagnostic JSONL record: {error}") from error
        record = _object(value, "record")
        if _canonical_json(cast(DiagnosticJsonValue, record)) != line:
            raise DiagnosticTraceError("diagnostic JSONL record is not canonical JSON")
        records.append(record)
    header = _header_from_payload(records[0])
    steps = tuple(_step_from_payload(record) for record in records[1:-1])
    footer = _footer_from_payload(records[-1])
    trace = DiagnosticTrace(header, steps, footer)
    expected = _records_digest(records[:-1])
    if footer.records_digest != expected:
        raise DiagnosticTraceError("diagnostic records digest does not match preceding records")
    return trace


def redacted_diagnostic_summary(trace: DiagnosticTrace) -> DiagnosticJsonObject:
    """Return an aggregate summary without actions, legal sets, card IDs, or snapshots."""

    _validate_trace(trace)
    steps: list[DiagnosticJsonValue] = []
    for step in trace.steps:
        progress = step.no_progress
        learned = _redacted_learned(step.learned)
        search = _redacted_search(step.search)
        repeated = progress.repeated_paid_action_window
        steps.append(
            {
                "sequence": step.sequence,
                "decision_hash": step.decision_hash,
                "before_state_hash": step.before_state_hash,
                "after_state_hash": step.after_state_hash,
                "before_strategic_digest": step.before_strategic_digest,
                "after_strategic_digest": step.after_strategic_digest,
                "chooser": step.chooser.value,
                "executor": step.executor.value,
                "dogma_activator": (
                    None if step.dogma_activator is None else step.dogma_activator.value
                ),
                "active_player": None if step.active_player is None else step.active_player.value,
                "decision_kind": step.decision_kind.value,
                "paid_actions_remaining": step.paid_actions_remaining,
                "handling": step.handling,
                "learned_aggregates": cast(DiagnosticJsonValue, learned),
                "search_aggregates": cast(DiagnosticJsonValue, search),
                "no_progress": {
                    "card_movement_count": len(progress.card_movements),
                    "splay_change_count": len(progress.splay_changes),
                    "achievement_count": len(progress.achievements),
                    "supply_change_count": len(progress.supply_changes),
                    "score_count": progress.score_count,
                    "meld_count": progress.meld_count,
                    "tuck_count": progress.tuck_count,
                    "return_count": progress.return_count,
                    "no_op_dogma": progress.no_op_dogma,
                    "repeated_paid_action": None if repeated is None else repeated.repeated,
                    "matching_prior_count": (
                        None if repeated is None else repeated.matching_prior_count
                    ),
                },
            }
        )
    payload: DiagnosticJsonObject = {
        "format": DIAGNOSTIC_REDACTED_FORMAT,
        "schema_version": DIAGNOSTIC_REDACTED_SCHEMA_VERSION,
        "source_revision": trace.header.source_revision,
        "game_id": trace.header.game_id,
        "setup_id": trace.header.setup_id,
        "manifest_digest": trace.header.manifest_digest,
        "config_digest": trace.header.config_digest,
        "versions": cast(DiagnosticJsonValue, dict(trace.header.versions)),
        "rng_seed_digests": cast(DiagnosticJsonValue, dict(trace.header.rng_seed_digests)),
        "initial_state_hash": trace.header.initial_state_hash,
        "initial_strategic_digest": trace.header.initial_strategic_digest,
        "steps": steps,
        "footer": {
            "step_count": trace.footer.step_count,
            "final_state_hash": trace.footer.final_state_hash,
            "final_strategic_digest": trace.footer.final_strategic_digest,
            "outcome": trace.footer.outcome,
            "no_progress_totals": cast(DiagnosticJsonValue, dict(trace.footer.no_progress_totals)),
            "records_digest": trace.footer.records_digest,
        },
    }
    _assert_redacted(payload)
    return payload


def write_redacted_diagnostic_summary(path: Path, trace: DiagnosticTrace) -> str:
    """Atomically write the canonical one-line redacted summary."""

    encoded = (
        _canonical_json(cast(DiagnosticJsonValue, redacted_diagnostic_summary(trace))) + "\n"
    ).encode("ascii")
    _atomic_write(path, encoded)
    return _tagged_sha256_bytes(encoded)


def _learned_payload(
    summary: LearnedDecisionSummary | None,
    decision: Decision,
    audit: PolicyDecisionAudit,
) -> DiagnosticJsonObject | None:
    if summary is None:
        return None
    selection = audit.selection
    if selection is None:
        raise DiagnosticTraceError("learned telemetry requires a learned policy selection")
    if len(summary.mean_values) != len(decision.legal_actions):
        raise DiagnosticTraceError("learned telemetry does not align to legal actions")
    selected_index = decision.legal_actions.index(audit.submission.action)
    if summary.selected_action_index != selected_index:
        raise DiagnosticTraceError("learned telemetry selected index differs from policy audit")
    if not math.isclose(
        summary.mean_values[selected_index], selection.mean_value, rel_tol=0.0, abs_tol=1e-12
    ):
        raise DiagnosticTraceError("learned telemetry selected mean differs from policy audit")
    return {
        "policy_id": selection.policy_id,
        "temperature": float(selection.temperature),
        "selector_version": selection.selector_version,
        "sampler_seed_digest": selection.sampler_seed_digest,
        "selector_seed_digest": selection.selector_seed_digest,
        "sample_values": [list(values) for values in summary.sample_values],
        "mean_values": list(summary.mean_values),
        "selector_scores": list(summary.selector_scores),
        "selected_action_index": summary.selected_action_index,
        "tied_action_indices": list(summary.tied_action_indices),
        "margin": float(summary.margin),
    }


def _search_payload(audit: PolicyDecisionAudit) -> DiagnosticJsonObject | None:
    if audit.search_selection is None:
        return None
    selection = audit.search_selection
    statistics = {
        field.name: getattr(selection.statistics, field.name)
        for field in fields(selection.statistics)
    }
    return {
        "telemetry": cast(DiagnosticJsonValue, selection.telemetry.payload()),
        "statistics": cast(DiagnosticJsonValue, statistics),
    }


def _failure_payload(error: BaseException | None) -> DiagnosticJsonObject | None:
    if error is None:
        return None
    return {"type": type(error).__name__, "message": str(error)}


def _card_zones(state: GameState) -> dict[CardId, str]:
    zones: dict[CardId, str] = {}
    for age, pile in enumerate(state.supply.piles, 1):
        zones.update((card_id, f"supply.age-{age}") for card_id in pile)
    for player in state.players:
        prefix = player.player_id.value
        zones.update((card_id, f"{prefix}.hand") for card_id in player.hand)
        zones.update((card_id, f"{prefix}.score") for card_id in player.score_pile)
        for stack in player.board.stacks:
            zones.update(
                (card_id, f"{prefix}.board.{stack.color.value}") for card_id in stack.cards
            )
    zones.update((card_id, "normal-achievement") for card_id in state.normal_achievements.cards)
    zones.update((card_id, "removed") for card_id in state.removed_cards)
    return zones


def _is_board_top(state: GameState, card_id: CardId, zone: str) -> bool:
    for player in state.players:
        for stack in player.board.stacks:
            if f"{player.player_id.value}.board.{stack.color.value}" == zone:
                return stack.top == card_id
    raise DiagnosticTraceError(f"card {card_id} has an unknown board zone {zone!r}")


def _positive_counter_delta(before: GameState, after: GameState, field_name: str) -> int:
    total = 0
    for player in PlayerId:
        left = getattr(before.turn_counters.for_player(player), field_name)
        right = getattr(after.turn_counters.for_player(player), field_name)
        total += max(0, cast(int, right) - cast(int, left))
    return total


def _validate_action_pattern(payload: DiagnosticJsonObject) -> None:
    if "schema_version" in payload or "decision_id" in payload:
        raise ValueError("paid-action pattern must omit schema and decision IDs")
    expanded: DiagnosticJsonObject = {
        "schema_version": 1,
        "decision_id": 1,
        **payload,
    }
    try:
        action = action_from_payload(expanded)
    except SerializationError as error:
        raise ValueError(f"invalid paid-action pattern: {error}") from error
    if _action_pattern(action) != payload:
        raise ValueError("paid-action pattern is not canonical")


def _action_pattern(action: SemanticAction) -> DiagnosticJsonObject:
    payload = cast(DiagnosticJsonObject, action_payload(action))
    return {
        key: value for key, value in payload.items() if key not in {"schema_version", "decision_id"}
    }


def _header_payload(header: DiagnosticTraceHeader) -> DiagnosticJsonObject:
    return {
        "record_type": "header",
        "format": DIAGNOSTIC_TRACE_FORMAT,
        "schema_version": DIAGNOSTIC_TRACE_SCHEMA_VERSION,
        "privacy": DIAGNOSTIC_TRACE_PRIVACY,
        "source_revision": header.source_revision,
        "game_id": header.game_id,
        "setup_id": header.setup_id,
        "setup_seed": header.setup_seed,
        "manifest_digest": header.manifest_digest,
        "config_digest": header.config_digest,
        "versions": cast(DiagnosticJsonValue, dict(header.versions)),
        "rng_seed_digests": cast(DiagnosticJsonValue, dict(header.rng_seed_digests)),
        "initial_state_hash": header.initial_state_hash,
        "initial_strategic_digest": header.initial_strategic_digest,
        "authoritative_snapshots": header.authoritative_snapshots,
        "private_debug": header.private_debug,
    }


def _movement_payload(item: CardMovement) -> DiagnosticJsonObject:
    return {
        "card_id": item.card_id.value,
        "before_zone": item.before_zone,
        "after_zone": item.after_zone,
    }


def _no_progress_payload(progress: NoProgressTelemetry) -> DiagnosticJsonObject:
    repeated = progress.repeated_paid_action_window
    return {
        "card_movements": [_movement_payload(item) for item in progress.card_movements],
        "splay_changes": [
            {
                "player": item.player.value,
                "color": item.color.value,
                "before": item.before,
                "after": item.after,
            }
            for item in progress.splay_changes
        ],
        "achievements": [
            {
                "player": item.player.value,
                "achievement_id": item.achievement_id,
                "achievement_kind": item.achievement_kind,
            }
            for item in progress.achievements
        ],
        "supply_changes": [
            {
                "age": item.age,
                "before_count": item.before_count,
                "after_count": item.after_count,
                "before_top": None if item.before_top is None else item.before_top.value,
                "after_top": None if item.after_top is None else item.after_top.value,
            }
            for item in progress.supply_changes
        ],
        "score_count": progress.score_count,
        "meld_count": progress.meld_count,
        "tuck_count": progress.tuck_count,
        "return_count": progress.return_count,
        "no_op_dogma": progress.no_op_dogma,
        "repeated_paid_action_window": (
            None
            if repeated is None
            else {
                "window_size": repeated.window_size,
                "patterns": list(repeated.patterns),
                "matching_prior_count": repeated.matching_prior_count,
                "repeated": repeated.repeated,
            }
        ),
    }


def _step_payload(step: DiagnosticStep) -> DiagnosticJsonObject:
    return {
        "record_type": "step",
        "schema_version": DIAGNOSTIC_TRACE_SCHEMA_VERSION,
        "sequence": step.sequence,
        "decision_id": step.decision_id,
        "decision_hash": step.decision_hash,
        "before_state_hash": step.before_state_hash,
        "after_state_hash": step.after_state_hash,
        "before_strategic_digest": step.before_strategic_digest,
        "after_strategic_digest": step.after_strategic_digest,
        "chooser": step.chooser.value,
        "executor": step.executor.value,
        "dogma_activator": None if step.dogma_activator is None else step.dogma_activator.value,
        "active_player": None if step.active_player is None else step.active_player.value,
        "decision_kind": step.decision_kind.value,
        "paid_actions_remaining": step.paid_actions_remaining,
        "legal_actions": list(step.legal_actions),
        "selected_action": step.selected_action,
        "handling": step.handling,
        "failure": step.failure,
        "learned": step.learned,
        "search": step.search,
        "no_progress": _no_progress_payload(step.no_progress),
        "before_snapshot": step.before_snapshot,
        "after_snapshot": step.after_snapshot,
    }


def _footer_payload(footer: DiagnosticTraceFooter) -> DiagnosticJsonObject:
    return {
        "record_type": "footer",
        "schema_version": DIAGNOSTIC_TRACE_SCHEMA_VERSION,
        "step_count": footer.step_count,
        "final_state_hash": footer.final_state_hash,
        "final_strategic_digest": footer.final_strategic_digest,
        "outcome": footer.outcome,
        "terminal_result": footer.terminal_result,
        "failure": footer.failure,
        "no_progress_totals": cast(DiagnosticJsonValue, dict(footer.no_progress_totals)),
        "records_digest": footer.records_digest,
    }


def _header_from_payload(payload: DiagnosticJsonObject) -> DiagnosticTraceHeader:
    _exact_keys(payload, set(_header_payload(_dummy_header())))
    if _string(payload["record_type"], "header.record_type") != "header":
        raise DiagnosticTraceSchemaError("first diagnostic record must be a header")
    if _string(payload["format"], "header.format") != DIAGNOSTIC_TRACE_FORMAT:
        raise DiagnosticTraceSchemaError("unsupported diagnostic trace format")
    _schema(payload, "header")
    if _string(payload["privacy"], "header.privacy") != DIAGNOSTIC_TRACE_PRIVACY:
        raise DiagnosticTraceSchemaError("diagnostic trace is not marked trusted-private")
    return DiagnosticTraceHeader(
        source_revision=_string(payload["source_revision"], "header.source_revision"),
        game_id=_string(payload["game_id"], "header.game_id"),
        setup_id=_string(payload["setup_id"], "header.setup_id"),
        setup_seed=_integer(payload["setup_seed"], "header.setup_seed"),
        manifest_digest=_string(payload["manifest_digest"], "header.manifest_digest"),
        config_digest=_string(payload["config_digest"], "header.config_digest"),
        versions=_optional_string_mapping(payload["versions"], "header.versions"),
        rng_seed_digests=_string_mapping(payload["rng_seed_digests"], "header.rng_seed_digests"),
        initial_state_hash=_string(payload["initial_state_hash"], "header.initial_state_hash"),
        initial_strategic_digest=_string(
            payload["initial_strategic_digest"], "header.initial_strategic_digest"
        ),
        authoritative_snapshots=_boolean(
            payload["authoritative_snapshots"], "header.authoritative_snapshots"
        ),
        private_debug=_boolean(payload["private_debug"], "header.private_debug"),
    )


def _step_from_payload(payload: DiagnosticJsonObject) -> DiagnosticStep:
    _exact_keys(payload, set(_step_payload(_dummy_step())))
    if _string(payload["record_type"], "step.record_type") != "step":
        raise DiagnosticTraceSchemaError("middle diagnostic records must be steps")
    _schema(payload, "step")
    legal_raw = _array(payload["legal_actions"], "step.legal_actions")
    legal = tuple(_validated_action_payload(item, "step.legal_actions[]") for item in legal_raw)
    selected = _validated_action_payload(payload["selected_action"], "step.selected_action")
    progress = _no_progress_from_payload(_object(payload["no_progress"], "step.no_progress"))
    before_snapshot = _optional_object(payload["before_snapshot"], "step.before_snapshot")
    after_snapshot = _optional_object(payload["after_snapshot"], "step.after_snapshot")
    learned = _optional_object(payload["learned"], "step.learned")
    search = _optional_object(payload["search"], "step.search")
    _validate_learned_payload(learned, len(legal))
    _validate_search_payload(search)
    return DiagnosticStep(
        sequence=_integer(payload["sequence"], "step.sequence"),
        decision_id=_integer(payload["decision_id"], "step.decision_id"),
        decision_hash=_string(payload["decision_hash"], "step.decision_hash"),
        before_state_hash=_string(payload["before_state_hash"], "step.before_state_hash"),
        after_state_hash=_string(payload["after_state_hash"], "step.after_state_hash"),
        before_strategic_digest=_string(
            payload["before_strategic_digest"], "step.before_strategic_digest"
        ),
        after_strategic_digest=_string(
            payload["after_strategic_digest"], "step.after_strategic_digest"
        ),
        chooser=PlayerId(_string(payload["chooser"], "step.chooser")),
        executor=PlayerId(_string(payload["executor"], "step.executor")),
        dogma_activator=_optional_player(payload["dogma_activator"], "step.dogma_activator"),
        active_player=_optional_player(payload["active_player"], "step.active_player"),
        decision_kind=DecisionKind(_string(payload["decision_kind"], "step.decision_kind")),
        paid_actions_remaining=_integer(
            payload["paid_actions_remaining"], "step.paid_actions_remaining"
        ),
        legal_actions=legal,
        selected_action=selected,
        handling=_string(payload["handling"], "step.handling"),
        failure=_optional_failure(payload["failure"], "step.failure"),
        learned=learned,
        search=search,
        no_progress=progress,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
    )


def _footer_from_payload(payload: DiagnosticJsonObject) -> DiagnosticTraceFooter:
    _exact_keys(payload, set(_footer_payload(_dummy_footer())))
    if _string(payload["record_type"], "footer.record_type") != "footer":
        raise DiagnosticTraceSchemaError("last diagnostic record must be a footer")
    _schema(payload, "footer")
    terminal = _optional_object(payload["terminal_result"], "footer.terminal_result")
    if terminal is not None:
        try:
            terminal_from_payload(terminal)
        except SerializationError as error:
            raise DiagnosticTraceSchemaError(f"invalid footer terminal result: {error}") from error
    totals_raw = _object(payload["no_progress_totals"], "footer.no_progress_totals")
    totals = {
        key: _integer(value, f"footer.no_progress_totals.{key}")
        for key, value in totals_raw.items()
    }
    return DiagnosticTraceFooter(
        step_count=_integer(payload["step_count"], "footer.step_count"),
        final_state_hash=_string(payload["final_state_hash"], "footer.final_state_hash"),
        final_strategic_digest=_string(
            payload["final_strategic_digest"], "footer.final_strategic_digest"
        ),
        outcome=_string(payload["outcome"], "footer.outcome"),
        terminal_result=terminal,
        failure=_optional_failure(payload["failure"], "footer.failure"),
        no_progress_totals=totals,
        records_digest=_string(payload["records_digest"], "footer.records_digest"),
    )


def _no_progress_from_payload(payload: DiagnosticJsonObject) -> NoProgressTelemetry:
    expected = {
        "card_movements",
        "splay_changes",
        "achievements",
        "supply_changes",
        "score_count",
        "meld_count",
        "tuck_count",
        "return_count",
        "no_op_dogma",
        "repeated_paid_action_window",
    }
    _exact_keys(payload, expected)
    movements = tuple(
        CardMovement(
            CardId(_string(item["card_id"], "movement.card_id")),
            _string(item["before_zone"], "movement.before_zone"),
            _string(item["after_zone"], "movement.after_zone"),
        )
        for raw in _array(payload["card_movements"], "no_progress.card_movements")
        for item in [_object(raw, "movement")]
        if not _exact_keys_return(item, {"card_id", "before_zone", "after_zone"})
    )
    splays = tuple(
        SplayChange(
            PlayerId(_string(item["player"], "splay.player")),
            Color(_string(item["color"], "splay.color")),
            _string(item["before"], "splay.before"),
            _string(item["after"], "splay.after"),
        )
        for raw in _array(payload["splay_changes"], "no_progress.splay_changes")
        for item in [_object(raw, "splay")]
        if not _exact_keys_return(item, {"player", "color", "before", "after"})
    )
    achievements = tuple(
        AchievementChange(
            PlayerId(_string(item["player"], "achievement.player")),
            _string(item["achievement_id"], "achievement.achievement_id"),
            _string(item["achievement_kind"], "achievement.achievement_kind"),
        )
        for raw in _array(payload["achievements"], "no_progress.achievements")
        for item in [_object(raw, "achievement")]
        if not _exact_keys_return(item, {"player", "achievement_id", "achievement_kind"})
    )
    supplies = tuple(
        SupplyChange(
            _integer(item["age"], "supply.age"),
            _integer(item["before_count"], "supply.before_count"),
            _integer(item["after_count"], "supply.after_count"),
            _optional_card(item["before_top"], "supply.before_top"),
            _optional_card(item["after_top"], "supply.after_top"),
        )
        for raw in _array(payload["supply_changes"], "no_progress.supply_changes")
        for item in [_object(raw, "supply")]
        if not _exact_keys_return(
            item, {"age", "before_count", "after_count", "before_top", "after_top"}
        )
    )
    repeated_raw = payload["repeated_paid_action_window"]
    repeated: RepeatedPaidActionWindow | None = None
    if repeated_raw is not None:
        item = _object(repeated_raw, "repeated_paid_action_window")
        _exact_keys(item, {"window_size", "patterns", "matching_prior_count", "repeated"})
        repeated = RepeatedPaidActionWindow(
            _integer(item["window_size"], "repeat.window_size"),
            tuple(
                _object(raw, "repeat.pattern")
                for raw in _array(item["patterns"], "repeat.patterns")
            ),
            _integer(item["matching_prior_count"], "repeat.matching_prior_count"),
            _boolean(item["repeated"], "repeat.repeated"),
        )
    counts = [
        _integer(payload[key], f"no_progress.{key}")
        for key in ("score_count", "meld_count", "tuck_count", "return_count")
    ]
    if any(value < 0 for value in counts):
        raise DiagnosticTraceSchemaError("no-progress counts cannot be negative")
    return NoProgressTelemetry(
        card_movements=movements,
        splay_changes=splays,
        achievements=achievements,
        supply_changes=supplies,
        score_count=counts[0],
        meld_count=counts[1],
        tuck_count=counts[2],
        return_count=counts[3],
        no_op_dogma=_boolean(payload["no_op_dogma"], "no_progress.no_op_dogma"),
        repeated_paid_action_window=repeated,
    )


def _validate_trace(trace: DiagnosticTrace) -> None:
    if trace.footer.step_count != len(trace.steps):
        raise DiagnosticTraceError("footer step count differs from trace")
    prior_state = trace.header.initial_state_hash
    prior_strategic = trace.header.initial_strategic_digest
    snapshots = trace.header.authoritative_snapshots
    for expected, step in enumerate(trace.steps, 1):
        if step.sequence != expected:
            raise DiagnosticTraceError("diagnostic step sequence is not contiguous")
        if step.before_state_hash != prior_state or step.before_strategic_digest != prior_strategic:
            raise DiagnosticTraceError("diagnostic before/after hash chain is broken")
        if snapshots != (step.before_snapshot is not None and step.after_snapshot is not None):
            raise DiagnosticTraceError("snapshot presence differs from the explicit header marker")
        if step.before_snapshot is not None:
            _verify_snapshot(
                step.before_snapshot, step.before_state_hash, step.before_strategic_digest
            )
            assert step.after_snapshot is not None
            _verify_snapshot(
                step.after_snapshot, step.after_state_hash, step.after_strategic_digest
            )
        prior_state = step.after_state_hash
        prior_strategic = step.after_strategic_digest
    if trace.footer.final_state_hash != prior_state:
        raise DiagnosticTraceError("footer final state hash differs from chain")
    if trace.footer.final_strategic_digest != prior_strategic:
        raise DiagnosticTraceError("footer final strategic digest differs from chain")
    if dict(trace.footer.no_progress_totals) != _no_progress_totals(trace.steps):
        raise DiagnosticTraceError("footer no-progress totals differ from steps")
    expected_digest = _records_digest(
        [_header_payload(trace.header), *(_step_payload(step) for step in trace.steps)]
    )
    if trace.footer.records_digest != expected_digest:
        raise DiagnosticTraceError("footer records digest differs from trace records")


def _verify_snapshot(
    payload: DiagnosticJsonObject, expected_hash: str, expected_strategic: str
) -> None:
    try:
        state = state_from_payload(payload)
    except SerializationError as error:
        raise DiagnosticTraceSchemaError(f"invalid authoritative snapshot: {error}") from error
    if state_hash(state) != expected_hash or strategic_state_digest(state) != expected_strategic:
        raise DiagnosticTraceError("authoritative snapshot differs from its hash markers")


def _no_progress_totals(steps: Sequence[DiagnosticStep]) -> dict[str, int]:
    return {
        "card_movements": sum(len(step.no_progress.card_movements) for step in steps),
        "splay_changes": sum(len(step.no_progress.splay_changes) for step in steps),
        "achievements": sum(len(step.no_progress.achievements) for step in steps),
        "supply_changes": sum(len(step.no_progress.supply_changes) for step in steps),
        "scored": sum(step.no_progress.score_count for step in steps),
        "melded": sum(step.no_progress.meld_count for step in steps),
        "tucked": sum(step.no_progress.tuck_count for step in steps),
        "returned": sum(step.no_progress.return_count for step in steps),
        "no_op_dogmas": sum(step.no_progress.no_op_dogma for step in steps),
        "repeated_paid_actions": sum(
            step.no_progress.repeated_paid_action_window is not None
            and step.no_progress.repeated_paid_action_window.repeated
            for step in steps
        ),
    }


def _redacted_learned(payload: DiagnosticJsonObject | None) -> DiagnosticJsonObject | None:
    if payload is None:
        return None
    means = tuple(
        _number(value, "learned.mean")
        for value in cast(list[DiagnosticJsonValue], payload["mean_values"])
    )
    samples = cast(list[DiagnosticJsonValue], payload["sample_values"])
    selected = cast(int, payload["selected_action_index"])
    return {
        "policy_id": payload["policy_id"],
        "temperature": payload["temperature"],
        "candidate_count": len(means),
        "determinization_count": len(cast(list[DiagnosticJsonValue], samples[0])) if samples else 0,
        "minimum_mean": min(means),
        "maximum_mean": max(means),
        "selected_mean": means[selected],
        "tie_count": len(cast(list[DiagnosticJsonValue], payload["tied_action_indices"])),
        "margin": payload["margin"],
    }


def _redacted_search(payload: DiagnosticJsonObject | None) -> DiagnosticJsonObject | None:
    if payload is None:
        return None
    telemetry = _object(payload["telemetry"], "search.telemetry")
    means = tuple(
        _number(value, "search.mean")
        for value in _array(telemetry["action_mean_values"], "search.means")
    )
    selected = _integer(telemetry["selected_action_index"], "search.selected")
    statistics = _object(payload["statistics"], "search.statistics")
    return {
        "search_descriptor_id": telemetry["search_descriptor_id"],
        "candidate_count": len(means),
        "minimum_mean": min(means),
        "maximum_mean": max(means),
        "selected_mean": means[selected],
        "tie_count": len(_array(telemetry["tied_action_indices"], "search.ties")),
        "route_count": statistics["routes"],
        "nodes": statistics["nodes"],
        "recursive_engine_transitions": statistics["recursive_engine_transitions"],
        "transposition_hits": statistics["transposition_hits"],
        "repeated_position_cutoffs": statistics["repeated_position_cutoffs"],
        "budget_cutoff_routes": statistics["budget_cutoff_routes"],
    }


def _validate_learned_payload(payload: DiagnosticJsonObject | None, legal_count: int) -> None:
    if payload is None:
        return
    expected = {
        "policy_id",
        "temperature",
        "sample_values",
        "mean_values",
        "selector_scores",
        "selected_action_index",
        "tied_action_indices",
        "margin",
    }
    _exact_keys(payload, expected)
    summary = LearnedDecisionSummary(
        tuple(
            tuple(_number(value, "learned.sample") for value in _array(raw, "learned.samples[]"))
            for raw in _array(payload["sample_values"], "learned.samples")
        ),
        tuple(
            _number(value, "learned.mean")
            for value in _array(payload["mean_values"], "learned.means")
        ),
        tuple(
            _number(value, "learned.score")
            for value in _array(payload["selector_scores"], "learned.scores")
        ),
        _integer(payload["selected_action_index"], "learned.selected"),
        tuple(
            _integer(value, "learned.tie")
            for value in _array(payload["tied_action_indices"], "learned.ties")
        ),
        _number(payload["margin"], "learned.margin"),
    )
    if len(summary.mean_values) != legal_count:
        raise DiagnosticTraceSchemaError("learned telemetry does not align to legal actions")
    _nonempty(_string(payload["policy_id"], "learned.policy_id"), "learned policy ID")
    _finite(_number(payload["temperature"], "learned.temperature"), "learned temperature")


def _validate_search_payload(payload: DiagnosticJsonObject | None) -> None:
    if payload is None:
        return
    _exact_keys(payload, {"telemetry", "statistics"})
    telemetry = _object(payload["telemetry"], "search.telemetry")
    required = {
        "schema_version",
        "search_descriptor_id",
        "action_keys",
        "action_mean_values",
        "selected_action_index",
        "selected_action_key",
        "tied_action_indices",
        "selector_seed_digest",
        "routes",
    }
    _exact_keys(telemetry, required)
    if _integer(telemetry["schema_version"], "search.schema_version") != 1:
        raise DiagnosticTraceSchemaError("unsupported search telemetry schema version")
    _digest(
        _string(telemetry["search_descriptor_id"], "search.descriptor"),
        "search descriptor",
    )
    keys = tuple(
        _string(value, "search.action_keys[]")
        for value in _array(telemetry["action_keys"], "search.action_keys")
    )
    means = tuple(
        _number(value, "search.action_mean_values[]")
        for value in _array(telemetry["action_mean_values"], "search.action_mean_values")
    )
    if not keys or len(keys) != len(means) or len(set(keys)) != len(keys):
        raise DiagnosticTraceSchemaError("search action keys and means are inconsistent")
    selected = _integer(telemetry["selected_action_index"], "search.selected_action_index")
    if not 0 <= selected < len(keys):
        raise DiagnosticTraceSchemaError("search selected action index is out of range")
    if _string(telemetry["selected_action_key"], "search.selected_action_key") != keys[selected]:
        raise DiagnosticTraceSchemaError("search selected action key differs from its index")
    ties = tuple(
        _integer(value, "search.tied_action_indices[]")
        for value in _array(telemetry["tied_action_indices"], "search.tied_action_indices")
    )
    if ties != tuple(sorted(set(ties))) or any(index < 0 or index >= len(keys) for index in ties):
        raise DiagnosticTraceSchemaError("search tied action indices are invalid")
    selector_digest = telemetry["selector_seed_digest"]
    if selector_digest is not None:
        _digest(_string(selector_digest, "search.selector_seed_digest"), "selector seed digest")
    routes = _array(telemetry["routes"], "search.routes")
    for raw in routes:
        _validate_search_route(_object(raw, "search.route"), len(keys))
    statistics = _object(payload["statistics"], "search.statistics")
    expected_statistics = {
        "routes",
        "nodes",
        "recursive_engine_transitions",
        "root_transitions",
        "mandatory_setup_transitions",
        "transposition_hits",
        "repeated_position_cutoffs",
        "budget_cutoff_routes",
        "immediate_leaf_fallback_routes",
    }
    _exact_keys(statistics, expected_statistics)
    for key, value in statistics.items():
        if _integer(value, f"search.statistics.{key}") < 0:
            raise DiagnosticTraceSchemaError("search statistics cannot be negative")
    if _integer(statistics["routes"], "search.statistics.routes") != len(routes):
        raise DiagnosticTraceSchemaError("search route statistic differs from route telemetry")


def _validate_search_route(payload: DiagnosticJsonObject, action_count: int) -> None:
    expected = {
        "schema_version",
        "root_action_index",
        "determinization_index",
        "value",
        "completed_turn_depth",
        "nodes",
        "engine_transitions",
        "transposition_hits",
        "repeated_position_cutoffs",
        "budget_cutoff",
        "immediate_leaf_fallback",
        "principal_variation",
    }
    _exact_keys(payload, expected, "search.route")
    if _integer(payload["schema_version"], "search.route.schema_version") != 1:
        raise DiagnosticTraceSchemaError("unsupported search route schema version")
    root_index = _integer(payload["root_action_index"], "search.route.root_action_index")
    if not 0 <= root_index < action_count:
        raise DiagnosticTraceSchemaError("search route root action index is out of range")
    _number(payload["value"], "search.route.value")
    for key in (
        "determinization_index",
        "completed_turn_depth",
        "nodes",
        "engine_transitions",
        "transposition_hits",
        "repeated_position_cutoffs",
    ):
        if _integer(payload[key], f"search.route.{key}") < 0:
            raise DiagnosticTraceSchemaError("search route counts cannot be negative")
    _boolean(payload["budget_cutoff"], "search.route.budget_cutoff")
    _boolean(payload["immediate_leaf_fallback"], "search.route.immediate_leaf_fallback")
    for value in _array(payload["principal_variation"], "search.route.principal_variation"):
        _nonempty(_string(value, "search.route.principal_variation[]"), "PV entry")


def _assert_redacted(value: DiagnosticJsonValue, path: str = "summary") -> None:
    forbidden = {
        "legal_actions",
        "selected_action",
        "action_keys",
        "principal_variation",
        "card_id",
        "card_ids",
        "before_snapshot",
        "after_snapshot",
        "terminal_result",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden or key.endswith("_snapshot"):
                raise DiagnosticTraceError(
                    f"redacted summary contains forbidden field {path}.{key}"
                )
            _assert_redacted(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_redacted(child, f"{path}[{index}]")


def _records_digest(records: Sequence[DiagnosticJsonObject]) -> str:
    encoded = "".join(
        _canonical_json(cast(DiagnosticJsonValue, item)) + "\n" for item in records
    ).encode("ascii")
    return _tagged_sha256_bytes(encoded)


def _deterministic_gzip(payload: bytes) -> bytes:
    stream = BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    compressed = stream.getvalue()
    if not compressed.startswith(FIXED_GZIP_HEADER):  # pragma: no cover - platform defense
        raise DiagnosticTraceError("runtime did not produce the required fixed gzip header")
    return compressed


def _read_fixed_gzip(compressed: bytes) -> bytes:
    if not compressed.startswith(FIXED_GZIP_HEADER):
        raise DiagnosticTraceError("diagnostic gzip header is not the required fixed header")
    try:
        decoder = zlib.decompressobj(wbits=31)
        raw = decoder.decompress(compressed) + decoder.flush()
    except zlib.error as error:
        raise DiagnosticTraceError(f"invalid or truncated diagnostic gzip: {error}") from error
    if not decoder.eof:
        raise DiagnosticTraceError("truncated diagnostic gzip member")
    if decoder.unused_data:
        raise DiagnosticTraceError("diagnostic trace must contain exactly one gzip member")
    return raw


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except OSError as error:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
        raise DiagnosticTraceError(
            f"could not write diagnostic artifact {path}: {error}"
        ) from error


def _canonical_json(value: DiagnosticJsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _tagged_sha256(text: str) -> str:
    return _tagged_sha256_bytes(text.encode("ascii"))


def _tagged_sha256_bytes(value: bytes) -> str:
    return f"{_SHA256_PREFIX}{hashlib.sha256(value).hexdigest()}"


def _digest(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith(_SHA256_PREFIX):
        raise ValueError(f"{label} must be a tagged SHA-256 digest")
    try:
        int(value[len(_SHA256_PREFIX) :], 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a tagged SHA-256 digest") from error
    if value.lower() != value:
        raise ValueError(f"{label} must be a tagged SHA-256 digest")


def _nonempty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} cannot be empty")


def _finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _object(value: object, path: str) -> DiagnosticJsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DiagnosticTraceSchemaError(f"{path} must be an object")
    return cast(DiagnosticJsonObject, value)


def _optional_object(value: DiagnosticJsonValue, path: str) -> DiagnosticJsonObject | None:
    return None if value is None else _object(value, path)


def _array(value: DiagnosticJsonValue, path: str) -> list[DiagnosticJsonValue]:
    if not isinstance(value, list):
        raise DiagnosticTraceSchemaError(f"{path} must be an array")
    return value


def _string(value: DiagnosticJsonValue, path: str) -> str:
    if not isinstance(value, str):
        raise DiagnosticTraceSchemaError(f"{path} must be a string")
    return value


def _integer(value: DiagnosticJsonValue, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiagnosticTraceSchemaError(f"{path} must be an integer")
    return value


def _number(value: DiagnosticJsonValue, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticTraceSchemaError(f"{path} must be a number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise DiagnosticTraceSchemaError(f"{path} must be finite")
    return scalar


def _boolean(value: DiagnosticJsonValue, path: str) -> bool:
    if not isinstance(value, bool):
        raise DiagnosticTraceSchemaError(f"{path} must be a boolean")
    return value


def _optional_player(value: DiagnosticJsonValue, path: str) -> PlayerId | None:
    return None if value is None else PlayerId(_string(value, path))


def _optional_card(value: DiagnosticJsonValue, path: str) -> CardId | None:
    return None if value is None else CardId(_string(value, path))


def _string_mapping(value: DiagnosticJsonValue, path: str) -> dict[str, str]:
    payload = _object(value, path)
    return {key: _string(item, f"{path}.{key}") for key, item in payload.items()}


def _optional_string_mapping(value: DiagnosticJsonValue, path: str) -> dict[str, str | None]:
    payload = _object(value, path)
    return {
        key: None if item is None else _string(item, f"{path}.{key}")
        for key, item in payload.items()
    }


def _optional_failure(value: DiagnosticJsonValue, path: str) -> DiagnosticJsonObject | None:
    payload = _optional_object(value, path)
    if payload is not None:
        _exact_keys(payload, {"type", "message"})
        _string(payload["type"], f"{path}.type")
        _string(payload["message"], f"{path}.message")
    return payload


def _validated_action_payload(value: DiagnosticJsonValue, path: str) -> DiagnosticJsonObject:
    payload = _object(value, path)
    try:
        action = action_from_payload(payload)
    except SerializationError as error:
        raise DiagnosticTraceSchemaError(f"invalid {path}: {error}") from error
    if cast(DiagnosticJsonObject, action_payload(action)) != payload:
        raise DiagnosticTraceSchemaError(f"{path} is not canonical")
    return payload


def _exact_keys(payload: DiagnosticJsonObject, expected: set[str], path: str = "record") -> None:
    if set(payload) != expected:
        raise DiagnosticTraceSchemaError(
            f"{path} keys differ: missing={sorted(expected - payload.keys())}, "
            f"unexpected={sorted(payload.keys() - expected)}"
        )


def _exact_keys_return(payload: DiagnosticJsonObject, expected: set[str]) -> bool:
    _exact_keys(payload, expected)
    return False


def _schema(payload: DiagnosticJsonObject, path: str) -> None:
    version = _integer(payload["schema_version"], f"{path}.schema_version")
    if version != DIAGNOSTIC_TRACE_SCHEMA_VERSION:
        raise DiagnosticTraceSchemaError(f"unsupported {path} schema version {version}")


def _dummy_header() -> DiagnosticTraceHeader:
    digest = "sha256:" + "0" * 64
    return DiagnosticTraceHeader(
        "x",
        "x",
        "x",
        0,
        digest,
        digest,
        {key: None for key in _REQUIRED_VERSION_KEYS},
        {},
        digest,
        digest,
    )


def _dummy_step() -> DiagnosticStep:
    digest = "sha256:" + "0" * 64
    action = cast(DiagnosticJsonObject, {"schema_version": 1, "kind": "draw", "decision_id": 1})
    progress = NoProgressTelemetry((), (), (), (), 0, 0, 0, 0, False, None)
    return DiagnosticStep(
        1,
        1,
        digest,
        digest,
        digest,
        digest,
        digest,
        PlayerId.PLAYER_1,
        PlayerId.PLAYER_1,
        None,
        PlayerId.PLAYER_1,
        DecisionKind.TURN_ACTION,
        1,
        (action,),
        action,
        "baseline",
        None,
        None,
        None,
        progress,
    )


def _dummy_footer() -> DiagnosticTraceFooter:
    digest = "sha256:" + "0" * 64
    return DiagnosticTraceFooter(
        0, digest, digest, "x", None, None, {key: 0 for key in _TOTAL_KEYS}, digest
    )


__all__ = [
    "DEFAULT_REPEATED_PAID_ACTION_WINDOW",
    "DIAGNOSTIC_REDACTED_FORMAT",
    "DIAGNOSTIC_TRACE_FORMAT",
    "DIAGNOSTIC_TRACE_PRIVACY",
    "DIAGNOSTIC_TRACE_SCHEMA_VERSION",
    "AchievementChange",
    "CardMovement",
    "DiagnosticStep",
    "DiagnosticTrace",
    "DiagnosticTraceError",
    "DiagnosticTraceFooter",
    "DiagnosticTraceHeader",
    "DiagnosticTraceRecorder",
    "DiagnosticTraceSchemaError",
    "LearnedDecisionSummary",
    "NoProgressTelemetry",
    "PrivateDiagnosticTraceRecorder",
    "RepeatedPaidActionWindow",
    "SplayChange",
    "SupplyChange",
    "build_diagnostic_step",
    "derive_no_progress_telemetry",
    "read_diagnostic_trace",
    "redacted_diagnostic_summary",
    "write_diagnostic_trace",
    "write_redacted_diagnostic_summary",
]
