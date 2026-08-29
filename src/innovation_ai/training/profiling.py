"""Stage-7 repeatable profiling contracts and callable scenario runner.

This module deliberately has no CLI dependency and no eager NumPy/PyTorch import.  It gives
orchestration a strict, versioned artifact format while allowing the (still separate) self-play
and arena owners to plug in callables at their natural boundaries.
"""

from __future__ import annotations

import json
import math
import os
import platform
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from time import perf_counter, process_time
from typing import Any, Protocol, cast

from innovation_ai.harness.engine import InnovationEngineAdapter, RunnerEngine
from innovation_ai.harness.seeds import setup_seed
from innovation_ai.innovation.actions import Decision, SemanticAction
from innovation_ai.innovation.zones import ValidationLevel, validation

PROFILE_CONFIG_FORMAT = "innovation-ai-profile-config"
PROFILE_REPORT_FORMAT = "innovation-ai-profile-report"
PROFILE_SCHEMA_VERSION = 1

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ProfileError(ValueError):
    """Base error for an invalid profile request, sample, or artifact."""


class ProfileSchemaError(ProfileError):
    """A JSON profile document is not an exact supported schema."""


class ProfileCategory(StrEnum):
    """Measured work categories used for comparable bottleneck accounting."""

    ENGINE = "engine"
    ENCODING = "encoding"
    INFERENCE = "inference"
    TRAINING = "training"
    DETERMINIZATION = "determinization"
    AFTERSTATE = "afterstate"
    REPLAY_EXTRACTION = "replay-extraction"
    SELF_PLAY = "self-play"
    ARENA = "arena"


class SweepDimension(StrEnum):
    """Configuration axes a scenario deliberately elects to sweep."""

    BATCH_SIZE = "batch_size"
    TORCH_NUM_THREADS = "torch_num_threads"
    GAMES_IN_FLIGHT = "games_in_flight"
    DETERMINIZATIONS = "determinizations"


class ScaleRecommendation(StrEnum):
    """The only next-scale recommendations emitted by Stage 7."""

    ACTOR_PROCESSES = "actor-processes"
    DEDICATED_INFERENCE = "dedicated-inference"
    GPU_BATCHING = "gpu-batching"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProfileError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ProfileError(f"{name} must be a finite non-negative number")
    return result


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ProfileError(f"{name} must be a non-empty trimmed string")
    return value


def _json_scalar(value: object, name: str) -> JsonScalar:
    if value is None or isinstance(value, (str, bool, int)):
        return cast(JsonScalar, value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ProfileError(f"{name} must be a finite JSON scalar")


def _pairs(
    values: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]], name: str
) -> tuple[tuple[str, JsonScalar], ...]:
    raw = values.items() if isinstance(values, Mapping) else values
    result = tuple(
        (_identifier(key, f"{name} key"), _json_scalar(value, name)) for key, value in raw
    )
    if len({key for key, _ in result}) != len(result):
        raise ProfileError(f"{name} keys must be unique")
    return tuple(sorted(result))


def _pairs_payload(values: tuple[tuple[str, JsonScalar], ...]) -> dict[str, JsonScalar]:
    return dict(values)


@dataclass(frozen=True, slots=True)
class IntegrityConfig:
    """Explicitly records every cost-saving integrity choice.

    ``validation_level`` is always set; disabling full validation therefore cannot happen by an
    accidental default.  ``transition_hashes`` and ``state_invariants`` are separate because
    runners can pay either cost independently.
    """

    validation_level: ValidationLevel = ValidationLevel.FULL
    transition_hashes: bool = True
    state_invariants: bool = True
    correctness_spot_checks: bool = False
    strict_spot_checks: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.validation_level, ValidationLevel):
            raise ProfileError("integrity validation_level must be a ValidationLevel")
        for name in (
            "transition_hashes",
            "state_invariants",
            "correctness_spot_checks",
            "strict_spot_checks",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ProfileError(f"integrity {name} must be boolean")
        if self.strict_spot_checks and not self.correctness_spot_checks:
            raise ProfileError("strict_spot_checks requires correctness_spot_checks")

    def payload(self) -> dict[str, JsonScalar]:
        return {
            "validation_level": self.validation_level.value,
            "transition_hashes": self.transition_hashes,
            "state_invariants": self.state_invariants,
            "correctness_spot_checks": self.correctness_spot_checks,
            "strict_spot_checks": self.strict_spot_checks,
        }


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Fully resolved, serializable configuration for one profiling artifact."""

    run_id: str
    command: tuple[str, ...]
    warmup_samples: int = 1
    timed_samples: int = 3
    batch_sizes: tuple[int, ...] = (1,)
    torch_num_threads: tuple[int, ...] = (1,)
    games_in_flight: tuple[int, ...] = (1,)
    determinizations: tuple[int, ...] = (1,)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    environment_overrides: tuple[tuple[str, JsonScalar], ...] = ()
    schema_version: int = PROFILE_SCHEMA_VERSION
    format: str = PROFILE_CONFIG_FORMAT

    def __post_init__(self) -> None:
        _identifier(self.run_id, "run_id")
        if self.format != PROFILE_CONFIG_FORMAT or self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ProfileError("unsupported profile config format or schema version")
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ProfileError("command must contain non-empty exact argv strings")
        _positive_int(self.warmup_samples, "warmup_samples")
        _positive_int(self.timed_samples, "timed_samples")
        for name in ("batch_sizes", "torch_num_threads", "games_in_flight", "determinizations"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ProfileError(f"{name} must be a non-empty tuple")
            if len(set(values)) != len(values):
                raise ProfileError(f"{name} must not contain duplicates")
            for value in values:
                _positive_int(value, name)
        if not isinstance(self.integrity, IntegrityConfig):
            raise ProfileError("integrity must be IntegrityConfig")
        object.__setattr__(
            self, "environment_overrides", _pairs(self.environment_overrides, "environment")
        )

    def payload(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "command": list(self.command),
            "warmup_samples": self.warmup_samples,
            "timed_samples": self.timed_samples,
            "batch_sizes": list(self.batch_sizes),
            "torch_num_threads": list(self.torch_num_threads),
            "games_in_flight": list(self.games_in_flight),
            "determinizations": list(self.determinizations),
            "integrity": cast(JsonValue, self.integrity.payload()),
            "environment_overrides": cast(JsonValue, _pairs_payload(self.environment_overrides)),
        }


@dataclass(frozen=True, slots=True)
class ProfileInvocation:
    """One adapter call, including its resolved sweep point and integrity contract."""

    config: ProfileConfig
    scenario_name: str
    category: ProfileCategory
    batch_size: int
    torch_num_threads: int
    games_in_flight: int
    determinizations: int
    sample_index: int
    warmup: bool
    parameters: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.scenario_name, "scenario name")
        if not isinstance(self.category, ProfileCategory):
            raise ProfileError("scenario category is unsupported")
        for name in ("batch_size", "torch_num_threads", "games_in_flight", "determinizations"):
            _positive_int(getattr(self, name), name)
        if self.sample_index < 0:
            raise ProfileError("sample_index cannot be negative")
        object.__setattr__(self, "parameters", _pairs(self.parameters, "scenario parameters"))

    def parameters_payload(self) -> dict[str, JsonScalar]:
        return {
            "batch_size": self.batch_size,
            "torch_num_threads": self.torch_num_threads,
            "games_in_flight": self.games_in_flight,
            "determinizations": self.determinizations,
            **_pairs_payload(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ScenarioWork:
    """Adapter-supplied work count and optional internal category timing split."""

    work_items: int
    work_unit: str
    category_wall_seconds: tuple[tuple[ProfileCategory, float], ...] = ()
    metrics: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        _positive_int(self.work_items, "work_items")
        _identifier(self.work_unit, "work_unit")
        categories = tuple(self.category_wall_seconds)
        if len({category for category, _ in categories}) != len(categories):
            raise ProfileError("category wall timings must not repeat a category")
        for category, seconds in categories:
            if not isinstance(category, ProfileCategory):
                raise ProfileError("category wall timing uses an unsupported category")
            _finite_nonnegative(seconds, "category wall seconds")
        object.__setattr__(
            self, "category_wall_seconds", tuple(sorted(categories, key=lambda item: item[0].value))
        )
        object.__setattr__(self, "metrics", _pairs(self.metrics, "work metrics"))


@dataclass(frozen=True, slots=True)
class ScenarioSample:
    """One repeated timed sample; no warmups are serialized as measurements."""

    scenario: str
    category: ProfileCategory
    sample_index: int
    parameters: tuple[tuple[str, JsonScalar], ...]
    work_items: int
    work_unit: str
    wall_seconds: float
    cpu_seconds: float
    throughput_per_second: float
    peak_rss_bytes: int
    category_wall_seconds: tuple[tuple[ProfileCategory, float], ...]
    metrics: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.scenario, "sample scenario")
        if not isinstance(self.category, ProfileCategory):
            raise ProfileError("sample category is unsupported")
        if self.sample_index < 0:
            raise ProfileError("sample_index cannot be negative")
        object.__setattr__(self, "parameters", _pairs(self.parameters, "sample parameters"))
        _positive_int(self.work_items, "sample work_items")
        _identifier(self.work_unit, "sample work_unit")
        wall = _finite_nonnegative(self.wall_seconds, "wall_seconds")
        cpu = _finite_nonnegative(self.cpu_seconds, "cpu_seconds")
        rate = _finite_nonnegative(self.throughput_per_second, "throughput_per_second")
        if wall == 0.0 and rate != 0.0:
            raise ProfileError("zero wall time requires zero reported throughput")
        if wall > 0.0 and not math.isclose(
            rate, self.work_items / wall, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ProfileError(
                "throughput_per_second must exactly derive from work_items / wall_seconds"
            )
        if (
            isinstance(self.peak_rss_bytes, bool)
            or not isinstance(self.peak_rss_bytes, int)
            or self.peak_rss_bytes < 0
        ):
            raise ProfileError("peak_rss_bytes must be a non-negative integer")
        splits = tuple(self.category_wall_seconds)
        if not splits:
            splits = ((self.category, wall),)
        if len({category for category, _ in splits}) != len(splits):
            raise ProfileError("sample category wall timings must not repeat a category")
        total = 0.0
        for category, seconds in splits:
            if not isinstance(category, ProfileCategory):
                raise ProfileError("sample category wall timing uses unsupported category")
            total += _finite_nonnegative(seconds, "sample category wall seconds")
        if total > wall + max(1e-9, wall * 1e-6):
            raise ProfileError("category wall timing cannot exceed sample wall time")
        object.__setattr__(
            self, "category_wall_seconds", tuple(sorted(splits, key=lambda item: item[0].value))
        )
        object.__setattr__(self, "metrics", _pairs(self.metrics, "sample metrics"))
        # Assign back normalized finite values for type checkers and JSON consistency.
        object.__setattr__(self, "wall_seconds", wall)
        object.__setattr__(self, "cpu_seconds", cpu)
        object.__setattr__(self, "throughput_per_second", rate)

    def payload(self) -> dict[str, JsonValue]:
        return {
            "scenario": self.scenario,
            "category": self.category.value,
            "sample_index": self.sample_index,
            "parameters": cast(JsonValue, _pairs_payload(self.parameters)),
            "work_items": self.work_items,
            "work_unit": self.work_unit,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "throughput_per_second": self.throughput_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
            "category_wall_seconds": {
                category.value: seconds for category, seconds in self.category_wall_seconds
            },
            "metrics": cast(JsonValue, _pairs_payload(self.metrics)),
        }


@dataclass(frozen=True, slots=True)
class CorrectnessSpotCheck:
    """An explicit cheap-mode/full-mode equivalence result supplied by a scenario owner."""

    scenario: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        _identifier(self.scenario, "spot-check scenario")
        if not isinstance(self.passed, bool):
            raise ProfileError("spot-check passed must be boolean")
        _identifier(self.detail, "spot-check detail")

    def payload(self) -> dict[str, JsonScalar]:
        return {"scenario": self.scenario, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class BottleneckEntry:
    """One category's aggregate wall-time contribution."""

    category: ProfileCategory
    wall_seconds: float
    wall_share: float

    def __post_init__(self) -> None:
        if not isinstance(self.category, ProfileCategory):
            raise ProfileError("bottleneck category is unsupported")
        _finite_nonnegative(self.wall_seconds, "bottleneck wall_seconds")
        share = _finite_nonnegative(self.wall_share, "bottleneck wall_share")
        if share > 1.0 + 1e-9:
            raise ProfileError("bottleneck wall_share cannot exceed one")

    def payload(self) -> dict[str, JsonValue]:
        return {
            "category": self.category.value,
            "wall_seconds": self.wall_seconds,
            "wall_share": self.wall_share,
        }


@dataclass(frozen=True, slots=True)
class BottleneckAnalysis:
    """Measured top-three analysis and a bounded, explainable scale recommendation."""

    largest: BottleneckEntry
    top_three: tuple[BottleneckEntry, ...]
    recommendation: ScaleRecommendation
    rationale: str

    def __post_init__(self) -> None:
        if not self.top_three or self.top_three[0] != self.largest:
            raise ProfileError("bottleneck top_three must begin with largest")
        if len({entry.category for entry in self.top_three}) != len(self.top_three):
            raise ProfileError("bottleneck categories must be unique")
        if (
            tuple(
                sorted(
                    self.top_three, key=lambda entry: (-entry.wall_seconds, entry.category.value)
                )
            )
            != self.top_three
        ):
            raise ProfileError("bottlenecks must be sorted by descending measured wall time")
        if not isinstance(self.recommendation, ScaleRecommendation):
            raise ProfileError("unsupported scale recommendation")
        _identifier(self.rationale, "bottleneck rationale")

    def payload(self) -> dict[str, JsonValue]:
        return {
            "largest": self.largest.payload(),
            "top_three": [entry.payload() for entry in self.top_three],
            "recommendation": self.recommendation.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ProfileEnvironment:
    """Machine/runtime metadata captured for the exact measured process."""

    python_version: str
    implementation: str
    platform: str
    machine: str
    cpu_count: int | None
    active_python_threads: int
    package_versions: tuple[tuple[str, str | None], ...]
    thread_environment: tuple[tuple[str, str | None], ...]
    overrides: tuple[tuple[str, JsonScalar], ...]

    def __post_init__(self) -> None:
        for name in ("python_version", "implementation", "platform", "machine"):
            _identifier(getattr(self, name), name)
        if self.cpu_count is not None:
            _positive_int(self.cpu_count, "cpu_count")
        _positive_int(self.active_python_threads, "active_python_threads")
        for name, values in (
            ("package_versions", self.package_versions),
            ("thread_environment", self.thread_environment),
        ):
            if len({key for key, _ in values}) != len(values):
                raise ProfileError(f"{name} keys must be unique")
            for key, value in values:
                _identifier(key, name)
                if value is not None and (not isinstance(value, str) or not value):
                    raise ProfileError(f"{name} values must be non-empty strings or null")
        object.__setattr__(self, "package_versions", tuple(sorted(self.package_versions)))
        object.__setattr__(self, "thread_environment", tuple(sorted(self.thread_environment)))
        object.__setattr__(self, "overrides", _pairs(self.overrides, "environment overrides"))

    def payload(self) -> dict[str, JsonValue]:
        return {
            "python_version": self.python_version,
            "implementation": self.implementation,
            "platform": self.platform,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "active_python_threads": self.active_python_threads,
            "package_versions": dict(self.package_versions),
            "thread_environment": dict(self.thread_environment),
            "overrides": cast(JsonValue, _pairs_payload(self.overrides)),
        }


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """Strict, deterministic-order Stage-7 profile artifact."""

    config: ProfileConfig
    environment: ProfileEnvironment
    samples: tuple[ScenarioSample, ...]
    correctness_spot_checks: tuple[CorrectnessSpotCheck, ...]
    bottlenecks: BottleneckAnalysis
    schema_version: int = PROFILE_SCHEMA_VERSION
    format: str = PROFILE_REPORT_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROFILE_REPORT_FORMAT or self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ProfileError("unsupported profile report format or schema version")
        if not self.samples:
            raise ProfileError("profile report requires at least one timed sample")
        if not isinstance(self.config, ProfileConfig) or not isinstance(
            self.environment, ProfileEnvironment
        ):
            raise ProfileError("profile report config and environment are required")
        expected_checks = {sample.scenario for sample in self.samples}
        actual_checks = {check.scenario for check in self.correctness_spot_checks}
        if len(actual_checks) != len(self.correctness_spot_checks):
            raise ProfileError("correctness spot checks cannot repeat a scenario")
        if self.config.integrity.correctness_spot_checks and actual_checks != expected_checks:
            raise ProfileError(
                "requested correctness spot checks must cover every measured scenario"
            )
        if self.config.integrity.strict_spot_checks and any(
            not check.passed for check in self.correctness_spot_checks
        ):
            raise ProfileError("a strict correctness spot check failed")
        ordered = tuple(sorted(self.samples, key=_sample_sort_key))
        if ordered != self.samples:
            raise ProfileError("samples must use deterministic scenario/sweep/sample ordering")
        object.__setattr__(
            self,
            "correctness_spot_checks",
            tuple(sorted(self.correctness_spot_checks, key=lambda item: item.scenario)),
        )

    def payload(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "config": self.config.payload(),
            "environment": self.environment.payload(),
            "samples": [sample.payload() for sample in self.samples],
            "correctness_spot_checks": [
                cast(JsonValue, check.payload()) for check in self.correctness_spot_checks
            ],
            "bottlenecks": self.bottlenecks.payload(),
        }

    def to_json(self) -> str:
        """Return canonical compact JSON with deterministic key and sample ordering."""
        return canonical_json(self.payload())

    def to_markdown(self) -> str:
        """Render a concise, non-lossy human summary of the measured samples."""
        rows = [
            (
                "| scenario | category | parameters | median wall (ms) | median CPU (ms) | "
                "median throughput | peak RSS (MiB) |"
            ),
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for key, group in _sample_groups(self.samples):
            sample = group[len(group) // 2]
            params = ", ".join(f"{name}={value}" for name, value in key[2])
            walls = sorted(item.wall_seconds for item in group)
            cpus = sorted(item.cpu_seconds for item in group)
            rates = sorted(item.throughput_per_second for item in group)
            rss = max(item.peak_rss_bytes for item in group) / (1024 * 1024)
            rows.append(
                f"| {sample.scenario} | {sample.category.value} | {params} | "
                f"{1000 * walls[len(walls) // 2]:.3f} | {1000 * cpus[len(cpus) // 2]:.3f} | "
                f"{rates[len(rates) // 2]:.2f} {sample.work_unit}/s | {rss:.2f} |"
            )
        top = self.bottlenecks.largest
        rows.extend(
            (
                "",
                f"**Largest measured wall-time share:** {top.category.value} "
                f"({top.wall_share:.1%}, {top.wall_seconds:.6f}s).",
                (
                    f"**Next scale step:** `{self.bottlenecks.recommendation.value}` — "
                    f"{self.bottlenecks.rationale}"
                ),
            )
        )
        if self.correctness_spot_checks:
            status = (
                "passed"
                if all(check.passed for check in self.correctness_spot_checks)
                else "failed"
            )
            rows.append(
                "**Correctness spot checks:** "
                f"{status} ({len(self.correctness_spot_checks)} scenarios)."
            )
        return "\n".join(rows) + "\n"


class ScenarioAdapter(Protocol):
    """Work owner API; self-play and arena integrate without importing this module's internals."""

    def __call__(self, invocation: ProfileInvocation, /) -> ScenarioWork:
        """Perform one bounded workload and return its exact completed work count."""


SpotCheckAdapter = Callable[[ProfileInvocation], CorrectnessSpotCheck | bool | str]


@dataclass(frozen=True, slots=True)
class ProfileScenario:
    """One named callable workload and its intentional sweep axes."""

    name: str
    category: ProfileCategory
    adapter: ScenarioAdapter
    sweep_dimensions: tuple[SweepDimension, ...] = ()
    parameters: tuple[tuple[str, JsonScalar], ...] = ()
    correctness_spot_check: SpotCheckAdapter | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, "scenario name")
        if not isinstance(self.category, ProfileCategory):
            raise ProfileError("scenario category is unsupported")
        if not callable(self.adapter):
            raise ProfileError("scenario adapter must be callable")
        if len(set(self.sweep_dimensions)) != len(self.sweep_dimensions):
            raise ProfileError("scenario sweep dimensions cannot repeat")
        if any(not isinstance(value, SweepDimension) for value in self.sweep_dimensions):
            raise ProfileError("scenario sweep dimensions are unsupported")
        object.__setattr__(self, "parameters", _pairs(self.parameters, "scenario parameters"))
        if self.correctness_spot_check is not None and not callable(self.correctness_spot_check):
            raise ProfileError("correctness_spot_check must be callable")


def callable_scenario(
    name: str,
    category: ProfileCategory,
    adapter: ScenarioAdapter,
    *,
    sweep_dimensions: Sequence[SweepDimension] = (),
    parameters: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] = (),
    correctness_spot_check: SpotCheckAdapter | None = None,
) -> ProfileScenario:
    """Build a callable adapter scenario, notably for self-play and arena owners."""
    return ProfileScenario(
        name,
        category,
        adapter,
        tuple(sweep_dimensions),
        _pairs(parameters, "scenario parameters"),
        correctness_spot_check,
    )


def engine_baseline_play_adapter[StateT](
    engine: RunnerEngine[StateT],
    *,
    games: int,
    run_seed: int,
    max_actions_per_game: int = 10_000,
    choose_action: Callable[[Decision], SemanticAction] | None = None,
) -> ScenarioAdapter:
    """Build an actual engine-only baseline-play workload adapter.

    It drives only normal runner-engine methods, uses the configured number of games in flight,
    and deliberately selects a legal semantic action without invoking an encoder or evaluator.
    When hashes are enabled it executes the optional authoritative fingerprint path as part of
    the measured workload.
    """
    _positive_int(games, "games")
    _positive_int(max_actions_per_game, "max_actions_per_game")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise ProfileError("run_seed must be an integer")

    def adapter(invocation: ProfileInvocation) -> ScenarioWork:
        selector = choose_action or (lambda decision: decision.legal_actions[0])
        active: dict[str, StateT] = {}
        action_counts: defaultdict[str, int] = defaultdict(int)
        next_game = 0
        actions = 0
        decisions = 0
        level = (
            invocation.config.integrity.validation_level
            if invocation.config.integrity.state_invariants
            else ValidationLevel.OFF
        )
        context = (
            validation(level) if isinstance(engine, InnovationEngineAdapter) else nullcontext()
        )
        with context:
            while active or next_game < games:
                while next_game < games and len(active) < invocation.games_in_flight:
                    game_id = f"profile-engine-{next_game:06d}"
                    state = engine.initial_state(setup_seed(run_seed, game_id))
                    if invocation.config.integrity.transition_hashes:
                        engine.fingerprint(state)
                    if engine.terminal_result(state) is None:
                        active[game_id] = state
                    next_game += 1
                for game_id in tuple(active):
                    state = active[game_id]
                    pending = engine.pending_decisions(state)
                    if not pending:
                        raise ProfileError(f"engine-only game {game_id} blocked without a decision")
                    for decision in pending:
                        if action_counts[game_id] >= max_actions_per_game:
                            raise ProfileError(
                                f"engine-only game {game_id} exceeded action ceiling "
                                f"{max_actions_per_game}"
                            )
                        action = selector(decision)
                        if action not in decision.legal_actions:
                            raise ProfileError(
                                "engine-only baseline selector returned an illegal action"
                            )
                        state = engine.apply(state, action)
                        action_counts[game_id] += 1
                        actions += 1
                        decisions += 1
                        if invocation.config.integrity.transition_hashes:
                            engine.fingerprint(state)
                        if engine.terminal_result(state) is not None:
                            break
                    if engine.terminal_result(state) is None:
                        active[game_id] = state
                    else:
                        del active[game_id]
        return ScenarioWork(
            actions,
            "actions",
            metrics=(("games", games), ("decisions", decisions)),
        )

    return adapter


def engine_baseline_play_scenario[StateT](
    name: str,
    engine: RunnerEngine[StateT],
    *,
    games: int,
    run_seed: int,
    max_actions_per_game: int = 10_000,
    choose_action: Callable[[Decision], SemanticAction] | None = None,
    correctness_spot_check: SpotCheckAdapter | None = None,
) -> ProfileScenario:
    """Build the Stage-7 engine-only baseline scenario around a real runner engine."""
    return engine_only_baseline_scenario(
        name,
        engine_baseline_play_adapter(
            engine,
            games=games,
            run_seed=run_seed,
            max_actions_per_game=max_actions_per_game,
            choose_action=choose_action,
        ),
        parameters={"games": games, "max_actions_per_game": max_actions_per_game},
        correctness_spot_check=correctness_spot_check,
    )


def engine_only_baseline_scenario(
    name: str,
    adapter: ScenarioAdapter,
    *,
    parameters: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] = (),
    correctness_spot_check: SpotCheckAdapter | None = None,
) -> ProfileScenario:
    """Declare an engine-only baseline-play adapter with a games-in-flight sweep."""
    return callable_scenario(
        name,
        ProfileCategory.ENGINE,
        adapter,
        sweep_dimensions=(SweepDimension.GAMES_IN_FLIGHT,),
        parameters=parameters,
        correctness_spot_check=correctness_spot_check,
    )


def encoding_scenario(name: str, adapter: ScenarioAdapter, **kwargs: Any) -> ProfileScenario:
    """Declare an encoding adapter with a batch-size sweep."""
    return callable_scenario(
        name,
        ProfileCategory.ENCODING,
        adapter,
        sweep_dimensions=(SweepDimension.BATCH_SIZE,),
        **kwargs,
    )


def inference_scenario(name: str, adapter: ScenarioAdapter, **kwargs: Any) -> ProfileScenario:
    """Declare model-inference work with batch-size and thread sweeps."""
    return callable_scenario(
        name,
        ProfileCategory.INFERENCE,
        adapter,
        sweep_dimensions=(SweepDimension.BATCH_SIZE, SweepDimension.TORCH_NUM_THREADS),
        **kwargs,
    )


def training_scenario(name: str, adapter: ScenarioAdapter, **kwargs: Any) -> ProfileScenario:
    """Declare training work with batch-size and intra-op-thread sweeps."""
    return callable_scenario(
        name,
        ProfileCategory.TRAINING,
        adapter,
        sweep_dimensions=(SweepDimension.BATCH_SIZE, SweepDimension.TORCH_NUM_THREADS),
        **kwargs,
    )


def determinization_scenario(name: str, adapter: ScenarioAdapter, **kwargs: Any) -> ProfileScenario:
    """Declare information-set sampling work with determinization-count sweep."""
    return callable_scenario(
        name,
        ProfileCategory.DETERMINIZATION,
        adapter,
        sweep_dimensions=(SweepDimension.DETERMINIZATIONS,),
        **kwargs,
    )


def afterstate_expansion_scenario(
    name: str, adapter: ScenarioAdapter, **kwargs: Any
) -> ProfileScenario:
    """Declare candidate expansion work with determinization and batch sweeps."""
    return callable_scenario(
        name,
        ProfileCategory.AFTERSTATE,
        adapter,
        sweep_dimensions=(SweepDimension.DETERMINIZATIONS, SweepDimension.BATCH_SIZE),
        **kwargs,
    )


def replay_extraction_scenario(
    name: str, adapter: ScenarioAdapter, **kwargs: Any
) -> ProfileScenario:
    """Declare verified compact-replay extraction work."""
    return callable_scenario(name, ProfileCategory.REPLAY_EXTRACTION, adapter, **kwargs)


def self_play_scenario(name: str, adapter: ScenarioAdapter, **kwargs: Any) -> ProfileScenario:
    """Declare a self-play callable; no self-play package import is required here."""
    return callable_scenario(
        name,
        ProfileCategory.SELF_PLAY,
        adapter,
        sweep_dimensions=(
            SweepDimension.GAMES_IN_FLIGHT,
            SweepDimension.BATCH_SIZE,
            SweepDimension.DETERMINIZATIONS,
        ),
        **kwargs,
    )


def arena_scenario(name: str, adapter: ScenarioAdapter, **kwargs: Any) -> ProfileScenario:
    """Declare an arena callable with actor/batch/determinization sweep points."""
    return callable_scenario(
        name,
        ProfileCategory.ARENA,
        adapter,
        sweep_dimensions=(
            SweepDimension.GAMES_IN_FLIGHT,
            SweepDimension.BATCH_SIZE,
            SweepDimension.DETERMINIZATIONS,
        ),
        **kwargs,
    )


def run_profile(config: ProfileConfig, scenarios: Sequence[ProfileScenario]) -> ProfileReport:
    """Warm up and repeatedly time all explicit scenario sweep points.

    The runner only changes PyTorch's intra-op setting when it is already importable.  No optional
    ML dependency is required for engine/replay-only profiling.
    """
    if not scenarios:
        raise ProfileError("at least one profiling scenario is required")
    scenario_batch = tuple(scenarios)
    if len({scenario.name for scenario in scenario_batch}) != len(scenario_batch):
        raise ProfileError("profile scenario names must be unique")
    samples: list[ScenarioSample] = []
    checks: list[CorrectnessSpotCheck] = []
    for scenario in sorted(scenario_batch, key=lambda item: item.name):
        spot_check_invocation: ProfileInvocation | None = None
        for point in _sweep_points(config, scenario.sweep_dimensions):
            warmup = ProfileInvocation(
                config, scenario.name, scenario.category, *point, 0, True, scenario.parameters
            )
            if spot_check_invocation is None:
                spot_check_invocation = warmup
            for warmup_index in range(config.warmup_samples):
                _maybe_set_torch_threads(point[1])
                scenario.adapter(_replace_sample_index(warmup, warmup_index))
            for sample_index in range(config.timed_samples):
                invocation = ProfileInvocation(
                    config,
                    scenario.name,
                    scenario.category,
                    *point,
                    sample_index,
                    False,
                    scenario.parameters,
                )
                _maybe_set_torch_threads(point[1])
                before_wall, before_cpu = perf_counter(), process_time()
                work = scenario.adapter(invocation)
                wall, cpu = perf_counter() - before_wall, process_time() - before_cpu
                if not isinstance(work, ScenarioWork):
                    raise ProfileError(
                        f"scenario {scenario.name!r} adapter must return ScenarioWork"
                    )
                split_by_category = dict(work.category_wall_seconds)
                measured_split = sum(split_by_category.values())
                if measured_split > wall + max(1e-9, wall * 1e-6):
                    raise ProfileError(
                        f"scenario {scenario.name!r} reported component timing above its wall time"
                    )
                split_by_category[scenario.category] = split_by_category.get(
                    scenario.category, 0.0
                ) + max(0.0, wall - measured_split)
                split = tuple(split_by_category.items())
                samples.append(
                    ScenarioSample(
                        scenario.name,
                        scenario.category,
                        sample_index,
                        tuple(sorted(invocation.parameters_payload().items())),
                        work.work_items,
                        work.work_unit,
                        wall,
                        cpu,
                        work.work_items / wall if wall else 0.0,
                        peak_rss_bytes(),
                        split,
                        work.metrics,
                    )
                )
        if config.integrity.correctness_spot_checks:
            if scenario.correctness_spot_check is None:
                raise ProfileError(
                    f"scenario {scenario.name!r} lacks required correctness spot check"
                )
            assert spot_check_invocation is not None  # every scenario has at least one sweep point
            checks.append(_run_spot_check(scenario, spot_check_invocation))
    ordered = tuple(sorted(samples, key=_sample_sort_key))
    return ProfileReport(
        config,
        collect_profile_environment(config),
        ordered,
        tuple(checks),
        analyze_bottlenecks(ordered),
    )


def _replace_sample_index(invocation: ProfileInvocation, index: int) -> ProfileInvocation:
    return ProfileInvocation(
        invocation.config,
        invocation.scenario_name,
        invocation.category,
        invocation.batch_size,
        invocation.torch_num_threads,
        invocation.games_in_flight,
        invocation.determinizations,
        index,
        invocation.warmup,
        invocation.parameters,
    )


def _run_spot_check(
    scenario: ProfileScenario, invocation: ProfileInvocation
) -> CorrectnessSpotCheck:
    assert scenario.correctness_spot_check is not None
    result = scenario.correctness_spot_check(invocation)
    if isinstance(result, CorrectnessSpotCheck):
        if result.scenario != scenario.name:
            raise ProfileError("spot check reported a different scenario")
        return result
    if isinstance(result, bool):
        return CorrectnessSpotCheck(scenario.name, result, "adapter returned equivalence result")
    if isinstance(result, str) and result:
        return CorrectnessSpotCheck(scenario.name, True, result)
    raise ProfileError(
        "spot-check adapter must return CorrectnessSpotCheck, bool, or non-empty detail"
    )


def _sweep_points(
    config: ProfileConfig, dimensions: tuple[SweepDimension, ...]
) -> tuple[tuple[int, int, int, int], ...]:
    axes: dict[SweepDimension, tuple[int, ...]] = {
        SweepDimension.BATCH_SIZE: config.batch_sizes,
        SweepDimension.TORCH_NUM_THREADS: config.torch_num_threads,
        SweepDimension.GAMES_IN_FLIGHT: config.games_in_flight,
        SweepDimension.DETERMINIZATIONS: config.determinizations,
    }
    values = [
        axes[dimension] if dimension in dimensions else (axes[dimension][0],)
        for dimension in SweepDimension
    ]
    return cast(tuple[tuple[int, int, int, int], ...], tuple(product(*values)))


def _maybe_set_torch_threads(value: int) -> None:
    try:
        torch = __import__("torch")
    except ImportError:
        return
    torch.set_num_threads(value)


def peak_rss_bytes() -> int:
    """Return process high-water RSS normalized to bytes without a psutil dependency."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if platform.system() == "Darwin" else raw * 1024


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def collect_profile_environment(config: ProfileConfig) -> ProfileEnvironment:
    """Capture exact relevant software and thread controls beside every report."""
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS")
    return ProfileEnvironment(
        platform.python_version(),
        platform.python_implementation(),
        platform.platform(),
        platform.machine(),
        os.cpu_count(),
        threading.active_count(),
        (
            ("numpy", _installed_version("numpy")),
            ("torch", _installed_version("torch")),
            ("innovation-ai", _installed_version("innovation-ai")),
        ),
        tuple((name, os.environ.get(name)) for name in names),
        config.environment_overrides,
    )


def analyze_bottlenecks(samples: Sequence[ScenarioSample]) -> BottleneckAnalysis:
    """Select the largest measured category and derive a bounded evidence-based next step."""
    if not samples:
        raise ProfileError("cannot analyze an empty profile")
    totals: defaultdict[ProfileCategory, float] = defaultdict(float)
    for sample in samples:
        for category, seconds in sample.category_wall_seconds:
            totals[category] += seconds
    total = sum(totals.values())
    if total <= 0.0:
        # Clock resolution can produce an all-zero synthetic test.  Keep an auditable answer.
        totals[samples[0].category] = 0.0
        total = 1.0
    entries = tuple(
        BottleneckEntry(category, seconds, seconds / total)
        for category, seconds in sorted(totals.items(), key=lambda item: (-item[1], item[0].value))
    )
    largest = entries[0]
    recommendation = _recommend(largest.category, samples)
    rationale = _rationale(largest, recommendation, samples)
    return BottleneckAnalysis(largest, entries[:3], recommendation, rationale)


def _recommend(category: ProfileCategory, samples: Sequence[ScenarioSample]) -> ScaleRecommendation:
    if category is ProfileCategory.TRAINING:
        return ScaleRecommendation.GPU_BATCHING
    if category in (ProfileCategory.INFERENCE, ProfileCategory.ENCODING):
        rates: dict[int, list[float]] = defaultdict(list)
        for sample in samples:
            if sample.category is category:
                batch = cast(int, dict(sample.parameters).get("batch_size", 1))
                rates[batch].append(sample.throughput_per_second)
        if len(rates) >= 2:
            first, last = min(rates), max(rates)
            first_rate = sum(rates[first]) / len(rates[first])
            last_rate = sum(rates[last]) / len(rates[last])
            if first_rate > 0.0 and last_rate / first_rate >= 1.2:
                return ScaleRecommendation.GPU_BATCHING
        return ScaleRecommendation.DEDICATED_INFERENCE
    return ScaleRecommendation.ACTOR_PROCESSES


def _rationale(
    largest: BottleneckEntry, recommendation: ScaleRecommendation, samples: Sequence[ScenarioSample]
) -> str:
    if recommendation is ScaleRecommendation.GPU_BATCHING:
        return (
            f"{largest.category.value} is the largest measured category; measured batch/thread "
            "work should move to GPU batching before adding actors."
        )
    if recommendation is ScaleRecommendation.DEDICATED_INFERENCE:
        return (
            f"{largest.category.value} is the largest measured category without a decisive batch "
            "scaling gain; isolate evaluator work in a dedicated inference process."
        )
    return (
        f"{largest.category.value} is the largest measured category; independent "
        "games/speculation dominate, so add bounded actor processes first."
    )


def _sample_sort_key(
    sample: ScenarioSample,
) -> tuple[str, str, tuple[tuple[str, JsonScalar], ...], int]:
    return (sample.scenario, sample.category.value, sample.parameters, sample.sample_index)


def _sample_groups(
    samples: Sequence[ScenarioSample],
) -> tuple[
    tuple[tuple[str, str, tuple[tuple[str, JsonScalar], ...]], tuple[ScenarioSample, ...]], ...
]:
    groups: dict[tuple[str, str, tuple[tuple[str, JsonScalar], ...]], list[ScenarioSample]] = (
        defaultdict(list)
    )
    for sample in samples:
        groups[(sample.scenario, sample.category.value, sample.parameters)].append(sample)
    return tuple(
        (key, tuple(sorted(group, key=lambda item: item.sample_index)))
        for key, group in sorted(groups.items())
    )


def canonical_json(payload: JsonValue) -> str:
    """Encode a finite profile document in the sole deterministic JSON representation."""
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ProfileSchemaError("profile payload is not finite JSON") from error


def dumps_profile_config(config: ProfileConfig) -> str:
    return canonical_json(config.payload())


def dumps_profile_report(report: ProfileReport) -> str:
    return report.to_json()


def loads_profile_config(text: str) -> ProfileConfig:
    payload = _object(_parse(text), "profile config")
    _keys(
        payload,
        {
            "format",
            "schema_version",
            "run_id",
            "command",
            "warmup_samples",
            "timed_samples",
            "batch_sizes",
            "torch_num_threads",
            "games_in_flight",
            "determinizations",
            "integrity",
            "environment_overrides",
        },
        "profile config",
    )
    integrity = _object(payload["integrity"], "integrity")
    _keys(
        integrity,
        {
            "validation_level",
            "transition_hashes",
            "state_invariants",
            "correctness_spot_checks",
            "strict_spot_checks",
        },
        "integrity",
    )
    try:
        return ProfileConfig(
            _string(payload["run_id"], "run_id"),
            tuple(_strings(payload["command"], "command")),
            _int(payload["warmup_samples"], "warmup_samples"),
            _int(payload["timed_samples"], "timed_samples"),
            tuple(_ints(payload["batch_sizes"], "batch_sizes")),
            tuple(_ints(payload["torch_num_threads"], "torch_num_threads")),
            tuple(_ints(payload["games_in_flight"], "games_in_flight")),
            tuple(_ints(payload["determinizations"], "determinizations")),
            IntegrityConfig(
                ValidationLevel(_string(integrity["validation_level"], "validation_level")),
                _bool(integrity["transition_hashes"], "transition_hashes"),
                _bool(integrity["state_invariants"], "state_invariants"),
                _bool(integrity["correctness_spot_checks"], "correctness_spot_checks"),
                _bool(integrity["strict_spot_checks"], "strict_spot_checks"),
            ),
            _object_pairs(payload["environment_overrides"], "environment_overrides"),
            _int(payload["schema_version"], "schema_version"),
            _string(payload["format"], "format"),
        )
    except (KeyError, ValueError, ProfileError) as error:
        raise ProfileSchemaError(str(error)) from error


def loads_profile_report(text: str) -> ProfileReport:
    """Decode an exact report schema, including embedded strict config and samples."""
    payload = _object(_parse(text), "profile report")
    _keys(
        payload,
        {
            "format",
            "schema_version",
            "config",
            "environment",
            "samples",
            "correctness_spot_checks",
            "bottlenecks",
        },
        "profile report",
    )
    config = loads_profile_config(canonical_json(payload["config"]))
    environment = _environment_from_payload(_object(payload["environment"], "environment"))
    samples = tuple(
        _sample_from_payload(_object(item, "sample"))
        for item in _array(payload["samples"], "samples")
    )
    checks = tuple(
        _check_from_payload(_object(item, "correctness_spot_check"))
        for item in _array(payload["correctness_spot_checks"], "correctness_spot_checks")
    )
    bottlenecks = _analysis_from_payload(_object(payload["bottlenecks"], "bottlenecks"))
    try:
        return ProfileReport(
            config,
            environment,
            samples,
            checks,
            bottlenecks,
            _int(payload["schema_version"], "schema_version"),
            _string(payload["format"], "format"),
        )
    except (KeyError, ValueError, ProfileError) as error:
        raise ProfileSchemaError(str(error)) from error


def _parse(text: str) -> JsonValue:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProfileSchemaError("profile JSON is invalid") from error
    return cast(JsonValue, value)


def _object(value: object, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ProfileSchemaError(f"{name} must be a JSON object")
    return cast(dict[str, JsonValue], value)


def _array(value: object, name: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ProfileSchemaError(f"{name} must be a JSON array")
    return cast(list[JsonValue], value)


def _keys(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ProfileSchemaError(f"{name} keys differ: missing={missing}, extra={extra}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ProfileSchemaError(f"{name} must be a string")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileSchemaError(f"{name} must be boolean")
    return value


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileSchemaError(f"{name} must be integer")
    return value


def _strings(value: object, name: str) -> list[str]:
    return [_string(item, name) for item in _array(value, name)]


def _ints(value: object, name: str) -> list[int]:
    return [_int(item, name) for item in _array(value, name)]


def _object_pairs(value: object, name: str) -> tuple[tuple[str, JsonScalar], ...]:
    payload = _object(value, name)
    return _pairs(cast(Mapping[str, JsonScalar], payload), name)


def _environment_from_payload(payload: dict[str, JsonValue]) -> ProfileEnvironment:
    _keys(
        payload,
        {
            "python_version",
            "implementation",
            "platform",
            "machine",
            "cpu_count",
            "active_python_threads",
            "package_versions",
            "thread_environment",
            "overrides",
        },
        "environment",
    )
    cpu: int | None = None
    if payload["cpu_count"] is not None:
        cpu = _int(payload["cpu_count"], "cpu_count")
    return ProfileEnvironment(
        _string(payload["python_version"], "python_version"),
        _string(payload["implementation"], "implementation"),
        _string(payload["platform"], "platform"),
        _string(payload["machine"], "machine"),
        cpu,
        _int(payload["active_python_threads"], "active_python_threads"),
        tuple(
            (key, value if value is None else _string(value, "package_versions value"))
            for key, value in _object(payload["package_versions"], "package_versions").items()
        ),
        tuple(
            (key, value if value is None else _string(value, "thread_environment value"))
            for key, value in _object(payload["thread_environment"], "thread_environment").items()
        ),
        _object_pairs(payload["overrides"], "overrides"),
    )


def _sample_from_payload(payload: dict[str, JsonValue]) -> ScenarioSample:
    _keys(
        payload,
        {
            "scenario",
            "category",
            "sample_index",
            "parameters",
            "work_items",
            "work_unit",
            "wall_seconds",
            "cpu_seconds",
            "throughput_per_second",
            "peak_rss_bytes",
            "category_wall_seconds",
            "metrics",
        },
        "sample",
    )
    return ScenarioSample(
        _string(payload["scenario"], "scenario"),
        ProfileCategory(_string(payload["category"], "category")),
        _int(payload["sample_index"], "sample_index"),
        _object_pairs(payload["parameters"], "parameters"),
        _int(payload["work_items"], "work_items"),
        _string(payload["work_unit"], "work_unit"),
        _number(payload["wall_seconds"], "wall_seconds"),
        _number(payload["cpu_seconds"], "cpu_seconds"),
        _number(payload["throughput_per_second"], "throughput_per_second"),
        _int(payload["peak_rss_bytes"], "peak_rss_bytes"),
        tuple(
            (ProfileCategory(key), _number(value, "category_wall_seconds"))
            for key, value in _object(
                payload["category_wall_seconds"], "category_wall_seconds"
            ).items()
        ),
        _object_pairs(payload["metrics"], "metrics"),
    )


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileSchemaError(f"{name} must be number")
    return float(value)


def _check_from_payload(payload: dict[str, JsonValue]) -> CorrectnessSpotCheck:
    _keys(payload, {"scenario", "passed", "detail"}, "correctness_spot_check")
    return CorrectnessSpotCheck(
        _string(payload["scenario"], "scenario"),
        _bool(payload["passed"], "passed"),
        _string(payload["detail"], "detail"),
    )


def _entry_from_payload(payload: dict[str, JsonValue]) -> BottleneckEntry:
    _keys(payload, {"category", "wall_seconds", "wall_share"}, "bottleneck entry")
    return BottleneckEntry(
        ProfileCategory(_string(payload["category"], "category")),
        _number(payload["wall_seconds"], "wall_seconds"),
        _number(payload["wall_share"], "wall_share"),
    )


def _analysis_from_payload(payload: dict[str, JsonValue]) -> BottleneckAnalysis:
    _keys(payload, {"largest", "top_three", "recommendation", "rationale"}, "bottlenecks")
    return BottleneckAnalysis(
        _entry_from_payload(_object(payload["largest"], "largest")),
        tuple(
            _entry_from_payload(_object(item, "top_three"))
            for item in _array(payload["top_three"], "top_three")
        ),
        ScaleRecommendation(_string(payload["recommendation"], "recommendation")),
        _string(payload["rationale"], "rationale"),
    )
