"""Repeatable Work Package R4 rollout feasibility benchmark.

This module is intentionally separate from :mod:`innovation_ai.search.benchmark`: the latter is
historical exhaustive-minimax evidence whose schema, behavior, and artifacts must remain unchanged.
The rollout benchmark reuses its committed eight authoritative corpus constructors, adds focused
rollout stress roots, and measures only the experimental deterministic turn-rollout algorithm.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import cast

from innovation_ai.innovation.actions import Decision, DogmaAction, SemanticAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.state import ExplicitPlayerPosition, GameState, build_explicit_state
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.innovation.zones import ValidationLevel, validation
from innovation_ai.search.benchmark import _representative_corpus
from innovation_ai.search.contracts import (
    EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING,
    RolloutSearchDescriptor,
)
from innovation_ai.search.information_sets import InformationSetSampler, InformationSetSpecBuilder
from innovation_ai.search.rollout import DeterministicSampledRollout, RolloutSelection
from innovation_ai.search.rollout_validation import (
    RolloutRouteLengthSoakResult,
    RolloutThroughputSmokeResult,
)
from innovation_ai.search.seeds import seed_digest

ROLLOUT_FEASIBILITY_SCHEMA_VERSION = 2
ROLLOUT_FEASIBILITY_TRUSTED_PRIVATE_FORMAT = "innovation-ai-rollout-feasibility-trusted-private"
ROLLOUT_FEASIBILITY_PUBLIC_REDACTED_FORMAT = "innovation-ai-rollout-feasibility-public-redacted"
ROLLOUT_FEASIBILITY_TRUSTED_PRIVATE = "trusted-private"
ROLLOUT_FEASIBILITY_PUBLIC_REDACTED = "public-redacted"
FROZEN_DETERMINIZATION_COUNTS = (1, 2, 4)
EXPERIMENTAL_ROUTE_TRANSITION_CEILING = EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING
FROZEN_TARGET_DETERMINIZATION_COUNT = 2
FROZEN_MAX_ROUTE_TRANSITIONS = 512
FROZEN_MIN_MEDIAN_ROOTS_PER_SECOND = 2.5
FROZEN_MAX_PER_ROOT_P95_SECONDS = 1.0
FROZEN_MIN_GENERATION_ACTIONS_PER_SECOND = 2.0
FROZEN_MIN_ARENA_ACTIONS_PER_SECOND = 1.0
FROZEN_CONCURRENCY = 1
FROZEN_TRACING_MODE = "aggregate-private-rollout-digests-v1"
FROZEN_TIMING_METHOD = "time.perf_counter-v1"

_HISTORICAL_CORPUS_NAMES = (
    "both-pending-starting-meld",
    "one-latent-starting-meld",
    "early-first-paid-turn",
    "explicit-second-paid-action",
    "own-turn-effect-choice",
    "opponent-demand-effect-choice",
    "opponent-shared-effect-choice",
    "late-high-branching-position",
)
_EXPANDED_CORPUS_NAMES = (
    "high-choice-first-paid-action",
    "nested-effect-decisions",
    "continuation-terminal",
    "no-op-no-progress-continuation",
)
DEFAULT_ROLLOUT_CORPUS_NAMES = (*_HISTORICAL_CORPUS_NAMES, *_EXPANDED_CORPUS_NAMES)


class RolloutFeasibilityError(RuntimeError):
    """A named rollout corpus/configuration case failed in strict mode."""


@dataclass(frozen=True, slots=True)
class RolloutFeasibilityConfig:
    """Reference R4 execution inputs, not a production rollout freeze.

    Tests and diagnostic smokes may select a corpus subset, fewer determinization counts, no
    warmups, or one repetition. Such runs can pass core-search checks but are ineligible for an
    overall gate pass. The 1,024 route ceiling is experimental pending an all-card/high-choice
    route-length soak.
    """

    determinization_counts: tuple[int, ...] = FROZEN_DETERMINIZATION_COUNTS
    route_transition_ceiling: int = EXPERIMENTAL_ROUTE_TRANSITION_CEILING
    corpus_names: tuple[str, ...] = DEFAULT_ROLLOUT_CORPUS_NAMES
    corpus_seed: int = 4404
    sampler_seed: int = 4406
    sampler_retry_limit: int = 32
    warmup_repetitions: int = 1
    measurement_repetitions: int = 5
    validation_level: ValidationLevel = ValidationLevel.FULL
    strict: bool = True
    concurrency: int = FROZEN_CONCURRENCY
    tracing_mode: str = FROZEN_TRACING_MODE
    timing_method: str = FROZEN_TIMING_METHOD

    def __post_init__(self) -> None:
        if not self.determinization_counts or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.determinization_counts
        ):
            raise ValueError("determinization counts must contain positive integers")
        if len(set(self.determinization_counts)) != len(self.determinization_counts):
            raise ValueError("determinization counts must be unique")
        if tuple(sorted(self.determinization_counts)) != self.determinization_counts:
            raise ValueError("determinization counts must use frozen ascending order")
        if any(value not in FROZEN_DETERMINIZATION_COUNTS for value in self.determinization_counts):
            raise ValueError(
                "rollout feasibility supports only frozen determinization counts 1/2/4"
            )
        if self.route_transition_ceiling != EXPERIMENTAL_ROUTE_TRANSITION_CEILING:
            raise ValueError(
                "rollout feasibility experiment requires the provisional 1024-transition ceiling"
            )
        if not self.corpus_names or any(not name for name in self.corpus_names):
            raise ValueError("corpus names must be non-empty")
        if len(set(self.corpus_names)) != len(self.corpus_names):
            raise ValueError("corpus names must be unique")
        unknown = set(self.corpus_names) - set(DEFAULT_ROLLOUT_CORPUS_NAMES)
        if unknown:
            raise ValueError(f"unknown rollout corpus entries: {sorted(unknown)}")
        for value, label in (
            (self.corpus_seed, "corpus seed"),
            (self.sampler_seed, "sampler seed"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
        if (
            isinstance(self.sampler_retry_limit, bool)
            or not isinstance(self.sampler_retry_limit, int)
            or self.sampler_retry_limit < 1
        ):
            raise ValueError("sampler retry limit must be positive")
        if (
            isinstance(self.warmup_repetitions, bool)
            or not isinstance(self.warmup_repetitions, int)
            or self.warmup_repetitions < 0
        ):
            raise ValueError("warmup repetitions must be non-negative")
        if (
            isinstance(self.measurement_repetitions, bool)
            or not isinstance(self.measurement_repetitions, int)
            or self.measurement_repetitions < 1
        ):
            raise ValueError("measurement repetitions must be positive")
        if not isinstance(self.validation_level, ValidationLevel):
            raise ValueError("validation level must be a ValidationLevel")
        if not isinstance(self.strict, bool):
            raise ValueError("strict mode must be boolean")
        if (
            isinstance(self.concurrency, bool)
            or not isinstance(self.concurrency, int)
            or self.concurrency != FROZEN_CONCURRENCY
        ):
            raise ValueError("rollout feasibility concurrency is frozen at one")
        if self.tracing_mode != FROZEN_TRACING_MODE:
            raise ValueError("rollout feasibility tracing mode differs from the frozen mode")
        if self.timing_method != FROZEN_TIMING_METHOD:
            raise ValueError("rollout feasibility timing method differs from the frozen method")

    def payload(self) -> dict[str, object]:
        return {
            "determinization_counts": list(self.determinization_counts),
            "route_transition_ceiling": self.route_transition_ceiling,
            "corpus_names": list(self.corpus_names),
            "corpus_seed": self.corpus_seed,
            "sampler_seed": self.sampler_seed,
            "sampler_retry_limit": self.sampler_retry_limit,
            "warmup_repetitions": self.warmup_repetitions,
            "measurement_repetitions": self.measurement_repetitions,
            "validation_level": self.validation_level.value,
            "strict": self.strict,
            "concurrency": self.concurrency,
            "tracing_mode": self.tracing_mode,
            "timing_method": self.timing_method,
        }

    @property
    def config_digest(self) -> str:
        return _payload_digest(self.payload())

    @property
    def is_reference_protocol(self) -> bool:
        """Whether every benchmark input matches the complete predeclared R4 invocation."""

        return self == type(self)()

    def public_redacted_payload(self) -> dict[str, object]:
        """Return non-secret configuration fields; sampling seeds stay trusted-private."""

        return {
            "determinization_counts": list(self.determinization_counts),
            "route_transition_ceiling": self.route_transition_ceiling,
            "route_transition_ceiling_status": "experimental-pending-all-card-route-soak",
            "corpus_names": list(self.corpus_names),
            "warmup_repetitions": self.warmup_repetitions,
            "measurement_repetitions": self.measurement_repetitions,
            "validation_level": self.validation_level.value,
            "strict": self.strict,
            "concurrency": self.concurrency,
            "timing_method": self.timing_method,
            "reference_protocol": self.is_reference_protocol,
        }


@dataclass(frozen=True, slots=True)
class _RolloutCorpusEntry:
    name: str
    category: str
    features: tuple[str, ...]
    state: GameState = field(repr=False)
    decision: Decision = field(repr=False)


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Raw warmed timings plus fixed median/nearest-rank-p95/maximum summaries."""

    wall_seconds: tuple[float, ...]
    median_seconds: float = field(init=False)
    p95_seconds: float = field(init=False)
    max_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.wall_seconds or any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0
            for value in self.wall_seconds
        ):
            raise ValueError("timing summary requires finite non-negative observations")
        ordered = sorted(self.wall_seconds)
        object.__setattr__(self, "median_seconds", float(median(ordered)))
        object.__setattr__(
            self,
            "p95_seconds",
            ordered[(95 * len(ordered) + 99) // 100 - 1],
        )
        object.__setattr__(self, "max_seconds", ordered[-1])

    def payload(self) -> dict[str, object]:
        return {
            "wall_seconds": list(self.wall_seconds),
            "median_seconds": self.median_seconds,
            "p95_seconds": self.p95_seconds,
            "max_seconds": self.max_seconds,
        }


@dataclass(frozen=True, slots=True)
class RateSummary:
    """Per-repetition aggregate throughput summary."""

    roots_per_second: tuple[float, ...]
    median_roots_per_second: float = field(init=False)
    min_roots_per_second: float = field(init=False)
    max_roots_per_second: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.roots_per_second or any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0
            for value in self.roots_per_second
        ):
            raise ValueError("rate summary requires finite non-negative observations")
        object.__setattr__(self, "median_roots_per_second", float(median(self.roots_per_second)))
        object.__setattr__(self, "min_roots_per_second", min(self.roots_per_second))
        object.__setattr__(self, "max_roots_per_second", max(self.roots_per_second))

    def payload(self) -> dict[str, object]:
        return {
            "roots_per_second": list(self.roots_per_second),
            "median_roots_per_second": self.median_roots_per_second,
            "min_roots_per_second": self.min_roots_per_second,
            "max_roots_per_second": self.max_roots_per_second,
        }


@dataclass(frozen=True, slots=True)
class ActionValueVariance:
    action_key: str
    sample_variance: float

    def payload(self) -> dict[str, object]:
        return {"action_key": self.action_key, "sample_variance": self.sample_variance}


@dataclass(frozen=True, slots=True)
class RolloutFeasibilityMeasurement:
    """Deterministic counters and warmed timing distribution for one root/sample count."""

    corpus_name: str
    category: str
    features: tuple[str, ...]
    decision_kind: str
    chooser: str
    determinization_count: int
    route_transition_ceiling: int
    rollout_descriptor_id: str
    continuation_agent_descriptor_id: str
    information_set_digest: str
    sampler_seed_digest: str
    selected_action_key: str
    selected_action_index: int
    action_keys: tuple[str, ...]
    action_mean_values: tuple[float, ...]
    action_value_variances: tuple[ActionValueVariance, ...]
    root_actions: int
    routes: int
    rollout_steps: int
    root_engine_transitions: int
    setup_response_engine_transitions: int
    continuation_engine_transitions: int
    total_engine_transitions: int
    completed_turn_routes: int
    terminal_routes: int
    completed_routes: int
    safety_ceiling_hits: int
    maximum_route_transitions: int
    deterministic_run_digest: str
    deterministic_rerun_digest: str
    deterministic_rerun_match: bool
    trusted_private_telemetry_json: str = field(repr=False)
    timing: TimingSummary
    median_roots_per_second: float
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        try:
            telemetry = json.loads(self.trusted_private_telemetry_json)
        except json.JSONDecodeError as error:
            raise ValueError("trusted-private rollout telemetry must be JSON") from error
        if _canonical_json(telemetry) != self.trusted_private_telemetry_json:
            raise ValueError("trusted-private rollout telemetry must be canonical JSON")

    def deterministic_payload(self) -> dict[str, object]:
        return {
            "corpus_name": self.corpus_name,
            "category": self.category,
            "features": list(self.features),
            "decision_kind": self.decision_kind,
            "chooser": self.chooser,
            "determinization_count": self.determinization_count,
            "route_transition_ceiling": self.route_transition_ceiling,
            "rollout_descriptor_id": self.rollout_descriptor_id,
            "continuation_agent_descriptor_id": self.continuation_agent_descriptor_id,
            "information_set_digest": self.information_set_digest,
            "sampler_seed_digest": self.sampler_seed_digest,
            "selected_action_key": self.selected_action_key,
            "selected_action_index": self.selected_action_index,
            "action_keys": list(self.action_keys),
            "action_mean_values": list(self.action_mean_values),
            "action_value_variances": [item.payload() for item in self.action_value_variances],
            "root_actions": self.root_actions,
            "routes": self.routes,
            "rollout_steps": self.rollout_steps,
            "engine_transitions": {
                "root": self.root_engine_transitions,
                "setup_response": self.setup_response_engine_transitions,
                "continuation": self.continuation_engine_transitions,
                "total": self.total_engine_transitions,
            },
            "completed_turn_routes": self.completed_turn_routes,
            "terminal_routes": self.terminal_routes,
            "completed_routes": self.completed_routes,
            "safety_ceiling_hits": self.safety_ceiling_hits,
            "maximum_route_transitions": self.maximum_route_transitions,
            "deterministic_run_digest": self.deterministic_run_digest,
            "deterministic_rerun_digest": self.deterministic_rerun_digest,
            "deterministic_rerun_match": self.deterministic_rerun_match,
            "selection_telemetry": json.loads(self.trusted_private_telemetry_json),
        }

    def public_redacted_deterministic_payload(self) -> dict[str, object]:
        """Return only aggregate counters; action/sample/path and private IDs are omitted."""

        variances = tuple(item.sample_variance for item in self.action_value_variances)
        return {
            "corpus_name": self.corpus_name,
            "category": self.category,
            "features": list(self.features),
            "decision_kind": self.decision_kind,
            "determinization_count": self.determinization_count,
            "route_transition_ceiling": self.route_transition_ceiling,
            "root_action_count": self.root_actions,
            "route_count": self.routes,
            "rollout_step_count": self.rollout_steps,
            "engine_transition_counts": {
                "root": self.root_engine_transitions,
                "setup_response": self.setup_response_engine_transitions,
                "continuation": self.continuation_engine_transitions,
                "total": self.total_engine_transitions,
            },
            "completed_turn_route_count": self.completed_turn_routes,
            "terminal_route_count": self.terminal_routes,
            "completed_route_count": self.completed_routes,
            "safety_ceiling_hit_count": self.safety_ceiling_hits,
            "maximum_route_transitions": self.maximum_route_transitions,
            "value_variance_aggregates": {
                "minimum": min(variances, default=0.0),
                "maximum": max(variances, default=0.0),
                "mean": math.fsum(variances) / len(variances) if variances else 0.0,
            },
            "deterministic_rerun_match": self.deterministic_rerun_match,
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.deterministic_payload(),
            "timing": self.timing.payload(),
            "median_roots_per_second": self.median_roots_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
        }

    def public_redacted_payload(self) -> dict[str, object]:
        return {
            **self.public_redacted_deterministic_payload(),
            "timing": self.timing.payload(),
            "median_roots_per_second": self.median_roots_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class RolloutFeasibilityFailure:
    corpus_name: str
    category: str
    determinization_count: int
    stage: str
    repetition: int | None
    error_type: str
    message: str

    def payload(self) -> dict[str, object]:
        return {
            "corpus_name": self.corpus_name,
            "category": self.category,
            "determinization_count": self.determinization_count,
            "stage": self.stage,
            "repetition": self.repetition,
            "error_type": self.error_type,
            "message": self.message,
        }

    def public_redacted_payload(self) -> dict[str, object]:
        """Omit exception text because it may contain paths or private identities."""

        return {
            "corpus_name": self.corpus_name,
            "category": self.category,
            "determinization_count": self.determinization_count,
            "stage": self.stage,
            "repetition": self.repetition,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class CrossCountActionVariance:
    action_key: str
    means_by_determinization_count: tuple[tuple[int, float], ...]
    variance: float

    def payload(self) -> dict[str, object]:
        return {
            "action_key": self.action_key,
            "means_by_determinization_count": {
                str(count): value for count, value in self.means_by_determinization_count
            },
            "variance": self.variance,
        }


@dataclass(frozen=True, slots=True)
class RolloutAgreementDiagnostic:
    """Action agreement and mean-value variance for one root across sample counts."""

    corpus_name: str
    selected_actions: tuple[tuple[int, str], ...]
    agrees_with_target: tuple[tuple[int, bool], ...]
    all_actions_agree: bool
    action_value_variances: tuple[CrossCountActionVariance, ...]

    def payload(self) -> dict[str, object]:
        return {
            "corpus_name": self.corpus_name,
            "selected_actions": {str(count): action for count, action in self.selected_actions},
            "agrees_with_target": {str(count): agrees for count, agrees in self.agrees_with_target},
            "all_actions_agree": self.all_actions_agree,
            "action_value_variances": [item.payload() for item in self.action_value_variances],
        }

    def public_redacted_payload(self) -> dict[str, object]:
        variances = tuple(item.variance for item in self.action_value_variances)
        return {
            "corpus_name": self.corpus_name,
            "all_actions_agree": self.all_actions_agree,
            "agreement_counts": {str(count): agrees for count, agrees in self.agrees_with_target},
            "cross_count_value_variance_aggregates": {
                "minimum": min(variances, default=0.0),
                "maximum": max(variances, default=0.0),
                "mean": math.fsum(variances) / len(variances) if variances else 0.0,
            },
        }


@dataclass(frozen=True, slots=True)
class DeterminizationAggregate:
    determinization_count: int
    roots: int
    root_actions: int
    routes: int
    rollout_steps: int
    total_engine_transitions: int
    completed_routes: int
    terminal_routes: int
    safety_ceiling_hits: int
    maximum_route_transitions: int
    per_root_timing: TimingSummary
    throughput: RateSummary
    max_per_root_p95_seconds: float
    peak_rss_bytes: int

    @property
    def completion_rate(self) -> float:
        return self.completed_routes / self.routes if self.routes else 0.0

    def payload(self) -> dict[str, object]:
        return {
            "determinization_count": self.determinization_count,
            "roots": self.roots,
            "root_actions": self.root_actions,
            "routes": self.routes,
            "rollout_steps": self.rollout_steps,
            "total_engine_transitions": self.total_engine_transitions,
            "completed_routes": self.completed_routes,
            "terminal_routes": self.terminal_routes,
            "completion_rate": self.completion_rate,
            "safety_ceiling_hits": self.safety_ceiling_hits,
            "maximum_route_transitions": self.maximum_route_transitions,
            "per_root_timing": self.per_root_timing.payload(),
            "throughput": self.throughput.payload(),
            "max_per_root_p95_seconds": self.max_per_root_p95_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    status: GateStatus
    actual: object
    requirement: str

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASS

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "passed": self.passed,
            "actual": self.actual,
            "requirement": self.requirement,
        }


@dataclass(frozen=True, slots=True)
class RolloutGateEvaluation:
    target_determinization_count: int
    core_search_passed: bool
    status: GateStatus
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        """Overall pass, not merely the core-search sub-gate result."""

        return self.status is GateStatus.PASS

    def payload(self) -> dict[str, object]:
        return {
            "target_determinization_count": self.target_determinization_count,
            "core_search_passed": self.core_search_passed,
            "status": self.status.value,
            "passed": self.passed,
            "checks": [item.payload() for item in self.checks],
        }


@dataclass(frozen=True, slots=True)
class RolloutFeasibilityResult:
    """R4 measurements with distinct trusted-private and public-redacted projections."""

    config: RolloutFeasibilityConfig
    corpus_digest: str
    measurements: tuple[RolloutFeasibilityMeasurement, ...]
    agreement_diagnostics: tuple[RolloutAgreementDiagnostic, ...]
    failures: tuple[RolloutFeasibilityFailure, ...] = ()
    route_length_soak: RolloutRouteLengthSoakResult | None = None
    generation_smoke: RolloutThroughputSmokeResult | None = None
    arena_smoke: RolloutThroughputSmokeResult | None = None
    schema_version: int = ROLLOUT_FEASIBILITY_SCHEMA_VERSION
    content_digest: str = field(init=False)
    public_content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != ROLLOUT_FEASIBILITY_SCHEMA_VERSION:
            raise ValueError("unsupported rollout feasibility schema version")
        if self.generation_smoke is not None and self.generation_smoke.smoke_kind != "generation":
            raise ValueError("generation smoke result has the wrong smoke kind")
        if self.arena_smoke is not None and self.arena_smoke.smoke_kind != "arena":
            raise ValueError("arena smoke result has the wrong smoke kind")
        object.__setattr__(
            self,
            "content_digest",
            _payload_digest(self.trusted_private_deterministic_payload()),
        )
        object.__setattr__(
            self,
            "public_content_digest",
            _payload_digest(self.public_redacted_deterministic_payload()),
        )

    def trusted_private_deterministic_payload(self) -> dict[str, object]:
        """Return deterministic private content, including actions, samples, and paths."""

        return {
            "format": ROLLOUT_FEASIBILITY_TRUSTED_PRIVATE_FORMAT,
            "privacy": ROLLOUT_FEASIBILITY_TRUSTED_PRIVATE,
            "schema_version": self.schema_version,
            "config": self.config.payload(),
            "config_digest": self.config.config_digest,
            "corpus_digest": self.corpus_digest,
            "measurements": [item.deterministic_payload() for item in self.measurements],
            "agreement_diagnostics": [item.payload() for item in self.agreement_diagnostics],
            "failures": [item.payload() for item in self.failures],
            "route_length_soak": (
                None
                if self.route_length_soak is None
                else self.route_length_soak.deterministic_payload()
            ),
            "generation_smoke": (
                None
                if self.generation_smoke is None
                else self.generation_smoke.deterministic_payload()
            ),
            "arena_smoke": (
                None if self.arena_smoke is None else self.arena_smoke.deterministic_payload()
            ),
        }

    def public_redacted_deterministic_payload(self) -> dict[str, object]:
        """Return deterministic aggregate content with no action/sample/path identities."""

        payload: dict[str, object] = {
            "format": ROLLOUT_FEASIBILITY_PUBLIC_REDACTED_FORMAT,
            "privacy": ROLLOUT_FEASIBILITY_PUBLIC_REDACTED,
            "schema_version": self.schema_version,
            "config": self.config.public_redacted_payload(),
            "corpus_digest": self.corpus_digest,
            "measurements": [
                item.public_redacted_deterministic_payload() for item in self.measurements
            ],
            "agreement_diagnostics": [
                item.public_redacted_payload() for item in self.agreement_diagnostics
            ],
            "failures": [item.public_redacted_payload() for item in self.failures],
            "route_length_soak": (
                None
                if self.route_length_soak is None
                else self.route_length_soak.deterministic_payload(public=True)
            ),
            "generation_smoke": (
                None
                if self.generation_smoke is None
                else self.generation_smoke.deterministic_payload(public=True)
            ),
            "arena_smoke": (
                None
                if self.arena_smoke is None
                else self.arena_smoke.deterministic_payload(public=True)
            ),
        }
        _assert_public_redacted(payload)
        return payload

    @property
    def aggregates(self) -> tuple[DeterminizationAggregate, ...]:
        return _aggregate_measurements(self.config, self.measurements)

    @property
    def gates(self) -> RolloutGateEvaluation:
        return evaluate_rollout_feasibility_gates(self)

    def trusted_private_payload(self) -> dict[str, object]:
        payload = {
            **self.trusted_private_deterministic_payload(),
            "content_digest": self.content_digest,
            "runtime": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "measurements": [item.payload() for item in self.measurements],
            "aggregates": [item.payload() for item in self.aggregates],
            "remaining_gate_measurements": {
                "route_length_soak": (
                    None if self.route_length_soak is None else self.route_length_soak.payload()
                ),
                "generation_smoke": (
                    None if self.generation_smoke is None else self.generation_smoke.payload()
                ),
                "arena_smoke": None if self.arena_smoke is None else self.arena_smoke.payload(),
            },
            "gates": self.gates.payload(),
        }
        return payload

    def public_redacted_payload(self) -> dict[str, object]:
        payload = {
            **self.public_redacted_deterministic_payload(),
            "public_content_digest": self.public_content_digest,
            "measurements": [item.public_redacted_payload() for item in self.measurements],
            "aggregates": [item.payload() for item in self.aggregates],
            "remaining_gate_measurements": {
                "route_length_soak": (
                    None
                    if self.route_length_soak is None
                    else self.route_length_soak.payload(public=True)
                ),
                "generation_smoke": (
                    None
                    if self.generation_smoke is None
                    else self.generation_smoke.payload(public=True)
                ),
                "arena_smoke": (
                    None if self.arena_smoke is None else self.arena_smoke.payload(public=True)
                ),
            },
            "gates": self.gates.payload(),
        }
        _assert_public_redacted(payload)
        return payload

    def to_trusted_private_json(self) -> str:
        return _canonical_json(self.trusted_private_payload())

    def to_public_redacted_json(self) -> str:
        return _canonical_json(self.public_redacted_payload())

    def trusted_private_json_file_digest(self) -> str:
        return _bytes_digest((self.to_trusted_private_json() + "\n").encode("utf-8"))

    def public_redacted_json_file_digest(self) -> str:
        return _bytes_digest((self.to_public_redacted_json() + "\n").encode("utf-8"))

    def to_trusted_private_markdown(self) -> str:
        gate = self.gates
        lines = [
            "# Milestone 4 rollout feasibility (trusted private)",
            "",
            f"Content digest: `{self.content_digest}`",
            f"Configuration digest: `{self.config.config_digest}`",
            f"Corpus digest: `{self.corpus_digest}`",
            f"Complete private JSON file digest: `{self.trusted_private_json_file_digest()}`",
            "",
            (
                "| corpus | category | dets | actions | routes | complete | terminal | steps | "
                "transitions | max route | median (s) | p95 (s) | max (s) | roots/s | RSS MiB | "
                "rerun |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for measurement in self.measurements:
            lines.append(
                f"| {measurement.corpus_name} | {measurement.category} | "
                f"{measurement.determinization_count} | {measurement.root_actions} | "
                f"{measurement.routes} | {measurement.completed_routes} | "
                f"{measurement.terminal_routes} | {measurement.rollout_steps} | "
                f"{measurement.total_engine_transitions} | "
                f"{measurement.maximum_route_transitions} | "
                f"{measurement.timing.median_seconds:.6f} | "
                f"{measurement.timing.p95_seconds:.6f} | "
                f"{measurement.timing.max_seconds:.6f} | "
                f"{measurement.median_roots_per_second:.3f} | "
                f"{measurement.peak_rss_bytes / (1024 * 1024):.1f} | "
                f"{'yes' if measurement.deterministic_rerun_match else 'NO'} |"
            )
        lines.extend(("", "## Aggregate scaling", "", _aggregate_markdown_header()))
        lines.extend(_aggregate_markdown_row(item) for item in self.aggregates)
        lines.extend(("", "## Action agreement and value variance", ""))
        for diagnostic in self.agreement_diagnostics:
            actions = ", ".join(
                f"{count}: `{action}`" for count, action in diagnostic.selected_actions
            )
            maximum_variance = max(
                (variance.variance for variance in diagnostic.action_value_variances),
                default=0.0,
            )
            lines.append(
                f"- **{diagnostic.corpus_name}:** {actions}; all agree: "
                f"{'yes' if diagnostic.all_actions_agree else 'no'}; "
                f"max cross-count value variance: {maximum_variance:.9g}."
            )
        _append_remaining_measurement_markdown(lines, self)
        _append_gate_markdown(lines, gate)
        if self.failures:
            lines.extend(("", "## Failures", ""))
            lines.extend(
                f"- **{item.corpus_name}** ({item.determinization_count} dets, {item.stage}): "
                f"`{item.error_type}: {item.message}`"
                for item in self.failures
            )
        return "\n".join(lines) + "\n"

    def to_public_redacted_markdown(self) -> str:
        gate = self.gates
        lines = [
            "# Milestone 4 rollout feasibility (public redacted)",
            "",
            f"Public content digest: `{self.public_content_digest}`",
            f"Corpus digest: `{self.corpus_digest}`",
            f"Complete public JSON file digest: `{self.public_redacted_json_file_digest()}`",
            "",
            _aggregate_markdown_header(),
        ]
        lines.extend(_aggregate_markdown_row(item) for item in self.aggregates)
        lines.extend(("", "## Aggregate action agreement and value variance", ""))
        for diagnostic in self.agreement_diagnostics:
            variances = tuple(item.variance for item in diagnostic.action_value_variances)
            lines.append(
                f"- **{diagnostic.corpus_name}:** all counts agree: "
                f"{'yes' if diagnostic.all_actions_agree else 'no'}; "
                f"max cross-count value variance: {max(variances, default=0.0):.9g}."
            )
        _append_remaining_measurement_markdown(lines, self)
        _append_gate_markdown(lines, gate)
        return "\n".join(lines) + "\n"

    def trusted_private_markdown_file_digest(self) -> str:
        return _bytes_digest(self.to_trusted_private_markdown().encode("utf-8"))

    def public_redacted_markdown_file_digest(self) -> str:
        return _bytes_digest(self.to_public_redacted_markdown().encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _TimedSelection:
    selection: RolloutSelection
    elapsed_seconds: float
    peak_rss_bytes: int


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _bytes_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _payload_digest(payload: object) -> str:
    return _bytes_digest(_canonical_json(payload).encode("ascii"))


_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "action_key",
        "action_keys",
        "action_mean_values",
        "action_sample_values",
        "chooser",
        "selected_action_index",
        "selected_action_key",
        "tied_action_indices",
        "selected_actions",
        "selection_telemetry",
        "rollout_path",
        "root_action_key",
        "setup_response_action_key",
        "information_set_digest",
        "sampler_seed_digest",
        "rollout_descriptor_id",
        "continuation_agent_descriptor_id",
        "decision_algorithm_descriptor_id",
        "message",
    }
)


def _assert_public_redacted(value: object, path: str = "public") -> None:
    """Reject accidental action/sample/path/private-identity fields in public projections."""

    if isinstance(value, dict):
        for key, item in value.items():
            if key in _PUBLIC_FORBIDDEN_KEYS or "path" in key:
                raise ValueError(f"public rollout benchmark contains forbidden field {path}.{key}")
            _assert_public_redacted(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public_redacted(item, f"{path}[{index}]")


def _aggregate_markdown_header() -> str:
    return (
        "| dets | roots | routes | completion | max route | median roots/s | "
        "max root p95 (s) | RSS MiB |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    )


def _aggregate_markdown_row(aggregate: DeterminizationAggregate) -> str:
    return (
        f"| {aggregate.determinization_count} | {aggregate.roots} | {aggregate.routes} | "
        f"{aggregate.completion_rate:.1%} | {aggregate.maximum_route_transitions} | "
        f"{aggregate.throughput.median_roots_per_second:.3f} | "
        f"{aggregate.max_per_root_p95_seconds:.6f} | "
        f"{aggregate.peak_rss_bytes / (1024 * 1024):.1f} |"
    )


def _append_remaining_measurement_markdown(
    lines: list[str],
    result: RolloutFeasibilityResult,
) -> None:
    lines.extend(("", "## Remaining R4 measurements", ""))
    soak = result.route_length_soak
    if soak is None:
        lines.append("- Route-length soak: not provided.")
    else:
        lines.append(
            f"- Route-length soak: {len(soak.measurements)}/{soak.planned_roots} roots; "
            f"{soak.completed_routes}/{soak.routes} routes complete; "
            f"max {soak.maximum_route_transitions} transitions; "
            f"{len(soak.failures)} failures; {soak.safety_ceiling_hits} ceiling hits."
        )
    for label, smoke in (
        ("Generation smoke", result.generation_smoke),
        ("Arena smoke", result.arena_smoke),
    ):
        if smoke is None:
            lines.append(f"- {label}: not provided.")
        else:
            lines.append(
                f"- {label}: {smoke.completed_games}/{smoke.expected_games} games; "
                f"{smoke.committed_actions} committed actions in "
                f"{smoke.elapsed_seconds:.3f}s "
                f"({smoke.actions_per_second:.3f} actions/s); "
                f"{len(smoke.failures)} failures."
            )


def _append_gate_markdown(lines: list[str], gate: RolloutGateEvaluation) -> None:
    lines.extend(("", "## Gate evaluation", ""))
    for check in gate.checks:
        lines.append(
            f"- {check.status.value.upper()} **{check.name}**: actual `{check.actual}`; "
            f"requires {check.requirement}."
        )
    lines.append(f"- **Core search:** {'PASS' if gate.core_search_passed else 'FAIL'}")
    lines.append(f"- **Overall:** {gate.status.value.upper()}")


def complete_file_digest(path: str | Path) -> str:
    """Return a tagged SHA-256 digest over every byte in an emitted artifact."""

    return _bytes_digest(Path(path).read_bytes())


def _peak_rss_bytes() -> int:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def _dogma_action(decision: Decision, card: str) -> SemanticAction:
    card_id = CardId(card)
    return next(
        action
        for action in decision.legal_actions
        if isinstance(action, DogmaAction) and action.card_id == card_id
    )


def _effect_entry(
    name: str,
    category: str,
    features: tuple[str, ...],
    state: GameState,
    card: str,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> _RolloutCorpusEntry:
    root = current_decisions(state, registry, programs)[0]
    transition = apply_action(state, _dogma_action(root, card), registry, programs)
    if transition.decision is None:
        raise RolloutFeasibilityError(f"corpus fixture {name} did not pause at an effect choice")
    return _RolloutCorpusEntry(name, category, features, transition.state, transition.decision)


def _expanded_corpus(
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> tuple[_RolloutCorpusEntry, ...]:
    high_choice_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(
                        CardId("agriculture"),
                        CardId("construction"),
                        CardId("alchemy"),
                        CardId("enterprise"),
                        CardId("statistics"),
                        CardId("publications"),
                    ),
                    board=((Color.PURPLE, (CardId("empiricism"),)),),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),)),
            ),
        ),
        supply_tops=((9, (CardId("collaboration"),)),),
        turn_number=21,
        paid_actions_remaining=2,
    )
    high_choice = _RolloutCorpusEntry(
        "high-choice-first-paid-action",
        "high-choice-first-action",
        ("first-paid-action", "ten-way-dogma-choice"),
        high_choice_state,
        current_decisions(high_choice_state, registry, programs)[0],
    )

    nested_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(CardId("tools"),),
                    board=(
                        (Color.RED, (CardId("archery"), CardId("engineering"))),
                        (Color.GREEN, (CardId("self-service"),)),
                        (Color.PURPLE, (CardId("philosophy"),)),
                    ),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),)),
            ),
        ),
        turn_number=25,
        paid_actions_remaining=2,
    )
    nested = _effect_entry(
        "nested-effect-decisions",
        "nested-effect",
        ("pending-effect-root", "nested-card-execution", "multiple-effect-decisions"),
        nested_state,
        "self-service",
        registry,
        programs,
    )

    terminal_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(CardId("databases"),),
                    board=((Color.YELLOW, (CardId("agriculture"),)),),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.RED, (CardId("archery"),)),)),
            ),
        ),
        turn_number=27,
        paid_actions_remaining=2,
    )
    terminal = _RolloutCorpusEntry(
        "continuation-terminal",
        "continuation-terminal",
        ("first-paid-action", "terminal-during-continuation"),
        terminal_state,
        current_decisions(terminal_state, registry, programs)[0],
    )

    no_progress_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(board=((Color.YELLOW, (CardId("machinery"),)),)),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.BLUE, (CardId("writing"),)),)),
            ),
        ),
        turn_number=29,
        paid_actions_remaining=2,
    )
    no_progress = _RolloutCorpusEntry(
        "no-op-no-progress-continuation",
        "no-progress",
        ("repeated-dogma", "no-op-root", "no-op-continuation"),
        no_progress_state,
        current_decisions(no_progress_state, registry, programs)[0],
    )
    return high_choice, nested, terminal, no_progress


def _rollout_corpus(
    seed: int,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> tuple[_RolloutCorpusEntry, ...]:
    historical = tuple(
        _RolloutCorpusEntry(
            item.name,
            item.category,
            ("historical-eight-root-corpus",),
            item.state,
            item.decision,
        )
        for item in _representative_corpus(seed, registry, programs)
    )
    if tuple(item.name for item in historical) != _HISTORICAL_CORPUS_NAMES:
        raise RolloutFeasibilityError("historical search corpus names changed unexpectedly")
    return (*historical, *_expanded_corpus(registry, programs))


def _sampler_seed(config: RolloutFeasibilityConfig, entry: _RolloutCorpusEntry) -> bytes:
    return f"rollout-feasibility:{config.sampler_seed}:{entry.name}".encode()


def _run_once(
    entry: _RolloutCorpusEntry,
    determinizations: int,
    config: RolloutFeasibilityConfig,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> _TimedSelection:
    sampler_seed = _sampler_seed(config, entry)
    started = perf_counter()
    spec = InformationSetSpecBuilder(registry, programs).build(entry.state, entry.decision)
    sampled = InformationSetSampler(
        registry,
        programs,
        retry_limit=config.sampler_retry_limit,
        strict=config.strict,
    ).sample_many(spec, determinizations, sampler_seed)
    if any(state is None for state in sampled):
        raise RolloutFeasibilityError("non-strict sampler returned no synthetic state")
    descriptor = RolloutSearchDescriptor(
        determinization_count=determinizations,
        route_transition_ceiling=config.route_transition_ceiling,
    )
    selection = DeterministicSampledRollout(
        descriptor,
        registry=registry,
        programs=programs,
    ).select(
        spec,
        cast(tuple[GameState, ...], sampled),
        sampler_seed_digest=seed_digest(sampler_seed),
    )
    return _TimedSelection(selection, perf_counter() - started, _peak_rss_bytes())


def _selection_digest(selection: RolloutSelection) -> str:
    return _payload_digest(selection.telemetry.payload())


def _population_variance(values: Iterable[float]) -> float:
    items = tuple(values)
    if not items:
        raise ValueError("variance requires at least one value")
    center = math.fsum(items) / len(items)
    return math.fsum((value - center) ** 2 for value in items) / len(items)


def _measurement_from_runs(
    entry: _RolloutCorpusEntry,
    determinizations: int,
    config: RolloutFeasibilityConfig,
    runs: tuple[_TimedSelection, ...],
    rerun: _TimedSelection,
) -> RolloutFeasibilityMeasurement:
    canonical = runs[0].selection
    canonical_digest = _selection_digest(canonical)
    measured_digests = tuple(_selection_digest(run.selection) for run in runs)
    rerun_digest = _selection_digest(rerun.selection)
    deterministic = all(item == canonical_digest for item in measured_digests) and (
        rerun_digest == canonical_digest
    )
    telemetry = canonical.telemetry
    statistics = canonical.statistics
    routes = telemetry.routes
    completed_routes = sum(
        route.completed_turn_count == 1 or route.terminal_reason is not None for route in routes
    )
    ceiling_hits = sum(route.safety_ceiling_hit for route in routes)
    timings = TimingSummary(tuple(run.elapsed_seconds for run in runs))
    return RolloutFeasibilityMeasurement(
        corpus_name=entry.name,
        category=entry.category,
        features=entry.features,
        decision_kind=entry.decision.kind.value,
        chooser=entry.decision.chooser.value,
        determinization_count=determinizations,
        route_transition_ceiling=config.route_transition_ceiling,
        rollout_descriptor_id=telemetry.rollout_descriptor_id,
        continuation_agent_descriptor_id=telemetry.continuation_agent_descriptor_id,
        information_set_digest=telemetry.information_set_digest,
        sampler_seed_digest=cast(str, telemetry.sampler_seed_digest),
        selected_action_key=telemetry.selected_action_key,
        selected_action_index=telemetry.selected_action_index,
        action_keys=telemetry.action_keys,
        action_mean_values=telemetry.action_mean_values,
        action_value_variances=tuple(
            ActionValueVariance(action_key, _population_variance(values))
            for action_key, values in zip(
                telemetry.action_keys, telemetry.action_sample_values, strict=True
            )
        ),
        root_actions=len(telemetry.action_keys),
        routes=statistics.routes,
        rollout_steps=statistics.continuation_engine_transitions,
        root_engine_transitions=statistics.root_engine_transitions,
        setup_response_engine_transitions=statistics.setup_response_engine_transitions,
        continuation_engine_transitions=statistics.continuation_engine_transitions,
        total_engine_transitions=statistics.total_engine_transitions,
        completed_turn_routes=statistics.completed_turn_routes,
        terminal_routes=statistics.terminal_routes,
        completed_routes=completed_routes,
        safety_ceiling_hits=ceiling_hits,
        maximum_route_transitions=statistics.maximum_route_transitions,
        deterministic_run_digest=canonical_digest,
        deterministic_rerun_digest=rerun_digest,
        deterministic_rerun_match=deterministic,
        trusted_private_telemetry_json=_canonical_json(telemetry.payload()),
        timing=timings,
        median_roots_per_second=(1.0 / timings.median_seconds if timings.median_seconds else 0.0),
        peak_rss_bytes=max(*(run.peak_rss_bytes for run in runs), rerun.peak_rss_bytes),
    )


def _corpus_digest(
    entries: tuple[_RolloutCorpusEntry, ...],
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> str:
    builder = InformationSetSpecBuilder(registry, programs)
    return _payload_digest(
        [
            {
                "name": entry.name,
                "category": entry.category,
                "features": list(entry.features),
                "decision_kind": entry.decision.kind.value,
                "chooser": entry.decision.chooser.value,
                "information_set_digest": builder.build(entry.state, entry.decision).spec_digest,
            }
            for entry in entries
        ]
    )


def _agreement_diagnostics(
    measurements: tuple[RolloutFeasibilityMeasurement, ...],
) -> tuple[RolloutAgreementDiagnostic, ...]:
    by_name: dict[str, list[RolloutFeasibilityMeasurement]] = {}
    for item in measurements:
        by_name.setdefault(item.corpus_name, []).append(item)
    diagnostics: list[RolloutAgreementDiagnostic] = []
    for corpus_name, unsorted_items in by_name.items():
        items = tuple(sorted(unsorted_items, key=lambda item: item.determinization_count))
        selected = tuple((item.determinization_count, item.selected_action_key) for item in items)
        target = next(
            (
                item.selected_action_key
                for item in items
                if item.determinization_count == FROZEN_TARGET_DETERMINIZATION_COUNT
            ),
            None,
        )
        agrees = tuple(
            (count, target is not None and action == target) for count, action in selected
        )
        common_actions = set(items[0].action_keys)
        for item in items[1:]:
            common_actions.intersection_update(item.action_keys)
        ordered_common = tuple(key for key in items[0].action_keys if key in common_actions)
        variances: list[CrossCountActionVariance] = []
        for action in ordered_common:
            means = tuple(
                (
                    item.determinization_count,
                    item.action_mean_values[item.action_keys.index(action)],
                )
                for item in items
            )
            variances.append(
                CrossCountActionVariance(
                    action,
                    means,
                    _population_variance(value for _, value in means),
                )
            )
        diagnostics.append(
            RolloutAgreementDiagnostic(
                corpus_name,
                selected,
                agrees,
                len({action for _, action in selected}) == 1,
                tuple(variances),
            )
        )
    return tuple(diagnostics)


def _aggregate_measurements(
    config: RolloutFeasibilityConfig,
    measurements: tuple[RolloutFeasibilityMeasurement, ...],
) -> tuple[DeterminizationAggregate, ...]:
    aggregates: list[DeterminizationAggregate] = []
    for determinizations in config.determinization_counts:
        items = tuple(
            item for item in measurements if item.determinization_count == determinizations
        )
        if not items:
            continue
        repetition_rates: list[float] = []
        for repetition in range(config.measurement_repetitions):
            elapsed = math.fsum(item.timing.wall_seconds[repetition] for item in items)
            repetition_rates.append(len(items) / elapsed if elapsed else 0.0)
        all_timings = tuple(value for item in items for value in item.timing.wall_seconds)
        aggregates.append(
            DeterminizationAggregate(
                determinizations,
                len(items),
                sum(item.root_actions for item in items),
                sum(item.routes for item in items),
                sum(item.rollout_steps for item in items),
                sum(item.total_engine_transitions for item in items),
                sum(item.completed_routes for item in items),
                sum(item.terminal_routes for item in items),
                sum(item.safety_ceiling_hits for item in items),
                max(item.maximum_route_transitions for item in items),
                TimingSummary(all_timings),
                RateSummary(tuple(repetition_rates)),
                max(item.timing.p95_seconds for item in items),
                max(item.peak_rss_bytes for item in items),
            )
        )
    return tuple(aggregates)


def evaluate_rollout_feasibility_gates(
    result: RolloutFeasibilityResult,
) -> RolloutGateEvaluation:
    """Evaluate core search separately from full protocol and downstream smoke gates."""

    target = next(
        (
            item
            for item in result.aggregates
            if item.determinization_count == FROZEN_TARGET_DETERMINIZATION_COUNT
        ),
        None,
    )
    all_routes = sum(item.routes for item in result.measurements)
    completed_routes = sum(item.completed_routes for item in result.measurements)
    max_route = max((item.maximum_route_transitions for item in result.measurements), default=0)
    expected_measurements = len(result.config.corpus_names) * len(
        result.config.determinization_counts
    )

    def measured(name: str, passed: bool, actual: object, requirement: str) -> GateCheck:
        return GateCheck(name, GateStatus.PASS if passed else GateStatus.FAIL, actual, requirement)

    core_checks = (
        measured("zero-failures", not result.failures, len(result.failures), "exactly 0"),
        measured(
            "complete-measurement-matrix",
            len(result.measurements) == expected_measurements,
            len(result.measurements),
            f"exactly {expected_measurements}",
        ),
        measured(
            "all-routes-complete",
            all_routes > 0 and completed_routes == all_routes,
            f"{completed_routes}/{all_routes}",
            "100% terminal or completed-turn boundary",
        ),
        measured(
            "zero-safety-ceiling-hits",
            all(item.safety_ceiling_hits == 0 for item in result.measurements),
            sum(item.safety_ceiling_hits for item in result.measurements),
            "exactly 0",
        ),
        measured(
            "maximum-route-length",
            max_route <= FROZEN_MAX_ROUTE_TRANSITIONS,
            max_route,
            (f"<= {FROZEN_MAX_ROUTE_TRANSITIONS} transitions under the experimental 1024 ceiling"),
        ),
        measured(
            "target-two-determinization-results",
            target is not None and target.roots == len(result.config.corpus_names),
            0 if target is None else target.roots,
            f"{len(result.config.corpus_names)} roots at 2 determinizations",
        ),
        measured(
            "target-median-throughput",
            target is not None
            and target.throughput.median_roots_per_second >= FROZEN_MIN_MEDIAN_ROOTS_PER_SECOND,
            None if target is None else target.throughput.median_roots_per_second,
            f">= {FROZEN_MIN_MEDIAN_ROOTS_PER_SECOND} roots/s",
        ),
        measured(
            "target-per-root-p95",
            target is not None
            and target.max_per_root_p95_seconds <= FROZEN_MAX_PER_ROOT_P95_SECONDS,
            None if target is None else target.max_per_root_p95_seconds,
            f"<= {FROZEN_MAX_PER_ROOT_P95_SECONDS} seconds",
        ),
        measured(
            "deterministic-immediate-rerun",
            bool(result.measurements)
            and all(item.deterministic_rerun_match for item in result.measurements),
            sum(item.deterministic_rerun_match for item in result.measurements),
            f"exactly {len(result.measurements)} matching measurements",
        ),
    )
    protocol_check = measured(
        "complete-reference-configuration",
        result.config.is_reference_protocol,
        result.config.is_reference_protocol,
        "the full predeclared corpus/configuration, not a subset or reduced smoke",
    )
    soak = result.route_length_soak
    if soak is None:
        ceiling_soak = GateCheck(
            "all-card-route-length-soak",
            GateStatus.PENDING,
            None,
            "reference seeded/all-card/high-choice soak with zero failures/hits and max <= 512",
        )
    else:
        soak_passed = (
            soak.config.is_reference_protocol
            and not soak.failures
            and len(soak.measurements) == soak.planned_roots
            and soak.all_routes_complete
            and soak.safety_ceiling_hits == 0
            and soak.maximum_route_transitions <= FROZEN_MAX_ROUTE_TRANSITIONS
        )
        ceiling_soak = measured(
            "all-card-route-length-soak",
            soak_passed,
            {
                "reference_protocol": soak.config.is_reference_protocol,
                "measured_roots": len(soak.measurements),
                "planned_roots": soak.planned_roots,
                "completed_routes": soak.completed_routes,
                "routes": soak.routes,
                "failures": len(soak.failures),
                "safety_ceiling_hits": soak.safety_ceiling_hits,
                "maximum_route_transitions": soak.maximum_route_transitions,
            },
            "reference seeded/all-card/high-choice soak with zero failures/hits and max <= 512",
        )

    def smoke_check(
        name: str,
        smoke: RolloutThroughputSmokeResult | None,
        expected_kind: str,
        minimum: float,
    ) -> GateCheck:
        if smoke is None:
            return GateCheck(
                name,
                GateStatus.PENDING,
                None,
                f"complete reference {expected_kind} smoke >= {minimum} committed actions/s",
            )
        passed = (
            smoke.smoke_kind == expected_kind
            and smoke.is_reference_protocol
            and smoke.complete
            and smoke.actions_per_second >= minimum
            and (expected_kind != "arena" or (smoke.rollout_decisions or 0) > 0)
        )
        return measured(
            name,
            passed,
            {
                "reference_protocol": smoke.is_reference_protocol,
                "completed_games": smoke.completed_games,
                "expected_games": smoke.expected_games,
                "committed_actions": smoke.committed_actions,
                "actions_per_second": smoke.actions_per_second,
                "failures": len(smoke.failures),
                "rollout_decisions": smoke.rollout_decisions,
            },
            f"complete reference {expected_kind} smoke >= {minimum} committed actions/s",
        )

    downstream = (
        ceiling_soak,
        smoke_check(
            "generation-smoke-throughput",
            result.generation_smoke,
            "generation",
            FROZEN_MIN_GENERATION_ACTIONS_PER_SECOND,
        ),
        smoke_check(
            "arena-smoke-throughput",
            result.arena_smoke,
            "arena",
            FROZEN_MIN_ARENA_ACTIONS_PER_SECOND,
        ),
    )
    checks = (*core_checks, protocol_check, *downstream)
    core_search_passed = all(item.passed for item in core_checks)
    if any(item.status is GateStatus.FAIL for item in checks):
        overall = GateStatus.FAIL
    elif any(item.status is GateStatus.PENDING for item in checks):
        overall = GateStatus.PENDING
    else:
        overall = GateStatus.PASS
    return RolloutGateEvaluation(
        FROZEN_TARGET_DETERMINIZATION_COUNT,
        core_search_passed,
        overall,
        checks,
    )


def run_rollout_feasibility(
    config: RolloutFeasibilityConfig | None = None,
    *,
    route_length_soak: RolloutRouteLengthSoakResult | None = None,
    generation_smoke: RolloutThroughputSmokeResult | None = None,
    arena_smoke: RolloutThroughputSmokeResult | None = None,
) -> RolloutFeasibilityResult:
    """Run the warmed R4 corpus sweep and an immediate semantic rerun for every case."""

    config = config or RolloutFeasibilityConfig()

    registry = load_card_registry()
    programs = load_effect_programs()
    all_entries = _rollout_corpus(config.corpus_seed, registry, programs)
    entries = tuple(item for item in all_entries if item.name in config.corpus_names)
    measurements: list[RolloutFeasibilityMeasurement] = []
    failures: list[RolloutFeasibilityFailure] = []

    with validation(config.validation_level):
        for entry in entries:
            for determinizations in config.determinization_counts:
                stage = "warmup"
                repetition: int | None = None
                try:
                    for warmup_index in range(config.warmup_repetitions):
                        repetition = warmup_index
                        _run_once(entry, determinizations, config, registry, programs)
                    stage = "measurement"
                    runs_list: list[_TimedSelection] = []
                    for measurement_index in range(config.measurement_repetitions):
                        repetition = measurement_index
                        runs_list.append(
                            _run_once(entry, determinizations, config, registry, programs)
                        )
                    runs = tuple(runs_list)
                    stage = "immediate-rerun"
                    repetition = None
                    rerun = _run_once(entry, determinizations, config, registry, programs)
                    measurement = _measurement_from_runs(
                        entry, determinizations, config, runs, rerun
                    )
                    if not measurement.deterministic_rerun_match:
                        raise RolloutFeasibilityError(
                            "deterministic counters, action values, paths, or selection changed"
                        )
                    measurements.append(measurement)
                except Exception as error:
                    failure = RolloutFeasibilityFailure(
                        entry.name,
                        entry.category,
                        determinizations,
                        stage,
                        repetition,
                        type(error).__name__,
                        str(error),
                    )
                    if config.strict:
                        raise RolloutFeasibilityError(
                            f"rollout feasibility failed for {entry.name} at "
                            f"determinizations={determinizations} during {stage}: "
                            f"{failure.error_type}: {failure.message}"
                        ) from error
                    failures.append(failure)

    measurement_tuple = tuple(measurements)
    return RolloutFeasibilityResult(
        config,
        _corpus_digest(entries, registry, programs),
        measurement_tuple,
        _agreement_diagnostics(measurement_tuple),
        tuple(failures),
        route_length_soak,
        generation_smoke,
        arena_smoke,
    )


def write_trusted_private_rollout_feasibility_json(
    result: RolloutFeasibilityResult,
    path: str | Path,
) -> Path:
    """Write the canonical trusted-private artifact containing action/sample/path telemetry."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.to_trusted_private_json() + "\n", encoding="utf-8")
    return target


def write_public_redacted_rollout_feasibility_json(
    result: RolloutFeasibilityResult,
    path: str | Path,
) -> Path:
    """Write canonical aggregate-only JSON safe for public publication."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.to_public_redacted_json() + "\n", encoding="utf-8")
    return target


def write_trusted_private_rollout_feasibility_markdown(
    result: RolloutFeasibilityResult,
    path: str | Path,
) -> Path:
    """Write the trusted-private human-readable timing and diagnostic report."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.to_trusted_private_markdown(), encoding="utf-8")
    return target


def write_public_redacted_rollout_feasibility_markdown(
    result: RolloutFeasibilityResult,
    path: str | Path,
) -> Path:
    """Write the public aggregate-only human-readable report."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.to_public_redacted_markdown(), encoding="utf-8")
    return target


RolloutBenchmarkConfig = RolloutFeasibilityConfig
RolloutBenchmarkResult = RolloutFeasibilityResult
run_rollout_benchmark = run_rollout_feasibility


__all__ = [
    "DEFAULT_ROLLOUT_CORPUS_NAMES",
    "EXPERIMENTAL_ROUTE_TRANSITION_CEILING",
    "FROZEN_CONCURRENCY",
    "FROZEN_DETERMINIZATION_COUNTS",
    "FROZEN_MAX_PER_ROOT_P95_SECONDS",
    "FROZEN_MAX_ROUTE_TRANSITIONS",
    "FROZEN_MIN_ARENA_ACTIONS_PER_SECOND",
    "FROZEN_MIN_GENERATION_ACTIONS_PER_SECOND",
    "FROZEN_MIN_MEDIAN_ROOTS_PER_SECOND",
    "FROZEN_TARGET_DETERMINIZATION_COUNT",
    "FROZEN_TIMING_METHOD",
    "FROZEN_TRACING_MODE",
    "ROLLOUT_FEASIBILITY_PUBLIC_REDACTED_FORMAT",
    "ROLLOUT_FEASIBILITY_SCHEMA_VERSION",
    "ROLLOUT_FEASIBILITY_TRUSTED_PRIVATE_FORMAT",
    "ActionValueVariance",
    "CrossCountActionVariance",
    "DeterminizationAggregate",
    "GateCheck",
    "GateStatus",
    "RateSummary",
    "RolloutAgreementDiagnostic",
    "RolloutBenchmarkConfig",
    "RolloutBenchmarkResult",
    "RolloutFeasibilityConfig",
    "RolloutFeasibilityError",
    "RolloutFeasibilityFailure",
    "RolloutFeasibilityMeasurement",
    "RolloutFeasibilityResult",
    "RolloutGateEvaluation",
    "TimingSummary",
    "complete_file_digest",
    "evaluate_rollout_feasibility_gates",
    "run_rollout_benchmark",
    "run_rollout_feasibility",
    "write_public_redacted_rollout_feasibility_json",
    "write_public_redacted_rollout_feasibility_markdown",
    "write_trusted_private_rollout_feasibility_json",
    "write_trusted_private_rollout_feasibility_markdown",
]
