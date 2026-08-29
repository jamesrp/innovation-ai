"""Repeatable Stage-0 baseline scenarios and machine-readable performance reports."""

from __future__ import annotations

import json
import os
import platform
import threading
from collections import Counter
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any, cast

from innovation_ai.agents.base import Agent
from innovation_ai.agents.descriptors import (
    RANDOM_AGENT_DESCRIPTOR,
    SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    AgentDescriptor,
)
from innovation_ai.agents.heuristic import SimpleHeuristicAgent
from innovation_ai.agents.random import RandomAgent
from innovation_ai.harness.actor_pool import BoundedActorPool
from innovation_ai.harness.engine import InnovationEngineAdapter, RunnerEngine
from innovation_ai.harness.metrics import MetricSink, NoOpMetricSink, NoOpTimerSink, TimerSink
from innovation_ai.harness.records import (
    GameResult,
    RunnerRecording,
    SemanticActionEvent,
    SemanticActionSink,
)
from innovation_ai.harness.runner import GameBlockedError, GameSpec, Submission
from innovation_ai.harness.seeds import agent_seed, setup_seed
from innovation_ai.innovation.actions import action_payload
from innovation_ai.innovation.serialization import terminal_payload
from innovation_ai.innovation.types import PlayerId
from innovation_ai.innovation.zones import ValidationLevel, validation

BASELINE_REPORT_SCHEMA_VERSION = 1


class _ScenarioEventDigest(SemanticActionSink):
    """Stream semantic actions into per-game digests without retaining action records."""

    def __init__(self) -> None:
        self._digests: dict[str, Any] = {}
        self.action_counts: Counter[str] = Counter()

    def record_action(self, event: SemanticActionEvent, /) -> None:
        digest = self._digests.setdefault(event.game_id, sha256())
        # The action-event recording path must stay independent of optional state fingerprints.
        payload = {
            "game_id": event.game_id,
            "setup_seed": event.setup_seed,
            "decision_id": event.decision_id,
            "chooser": event.chooser.value,
            "action": action_payload(event.action),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        digest.update(encoded + b"\n")
        self.action_counts[event.game_id] += 1

    def finish(self, result: GameResult[object]) -> str:
        record = result.record
        digest = self._digests.setdefault(record.game_id, sha256())
        payload = {
            "initial_state": record.initial_state,
            "terminal": terminal_payload(record.terminal),
            "final_state": record.final_state,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        digest.update(encoded)
        return str(digest.hexdigest())


@dataclass(frozen=True, slots=True)
class BaselineScenario:
    """One fixed two-seat baseline matchup."""

    name: str
    player_1: AgentDescriptor
    player_2: AgentDescriptor


@dataclass(frozen=True, slots=True)
class BaselineBenchmarkConfig:
    """Resolved deterministic inputs for a small baseline benchmark run."""

    run_seed: int
    games_per_scenario: int = 32
    games_in_flight: int = 1
    validation_levels: tuple[ValidationLevel, ...] = (
        ValidationLevel.FULL,
        ValidationLevel.CHEAP,
    )
    max_actions_per_game: int = 10_000

    def __post_init__(self) -> None:
        if self.games_per_scenario < 1:
            raise ValueError("games per scenario must be positive")
        if self.games_in_flight < 1:
            raise ValueError("games in flight must be positive")
        if self.max_actions_per_game < 1:
            raise ValueError("max actions per game must be positive")
        if not self.validation_levels:
            raise ValueError("at least one validation level is required")
        if len(set(self.validation_levels)) != len(self.validation_levels):
            raise ValueError("validation levels must be unique")


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Stable descriptive statistics for an integer game-level distribution."""

    mean: float
    p50: int
    p95: int
    maximum: int

    @classmethod
    def from_values(cls, values: Iterable[int]) -> DistributionSummary:
        ordered = sorted(values)
        if not ordered:
            raise ValueError("distribution requires at least one value")
        count = len(ordered)
        return cls(
            sum(ordered) / count,
            ordered[(count - 1) // 2],
            ordered[(95 * count + 99) // 100 - 1],
            ordered[-1],
        )

    def payload(self) -> dict[str, float | int]:
        """Return a JSON-compatible summary."""

        return {
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "max": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class BaselineScenarioReport:
    """One scenario/validation result with deterministic game outcomes and live timings."""

    scenario: BaselineScenario
    validation_level: ValidationLevel
    games: int
    actions: int
    decisions: int
    action_length: DistributionSummary
    decision_length: DistributionSummary
    terminal_reasons: tuple[tuple[str, int], ...]
    semantic_digest: str
    elapsed_seconds: float
    games_per_second: float
    actions_per_second: float
    decisions_per_second: float
    peak_rss_bytes: int

    def payload(self) -> dict[str, object]:
        """Return the machine-readable report fragment."""

        return {
            "scenario": self.scenario.name,
            "agents": {
                "player_1": self.scenario.player_1.payload(),
                "player_2": self.scenario.player_2.payload(),
            },
            "validation_level": self.validation_level.value,
            "games": self.games,
            "actions": self.actions,
            "decisions": self.decisions,
            "action_length": self.action_length.payload(),
            "decision_length": self.decision_length.payload(),
            "terminal_reasons": dict(self.terminal_reasons),
            "semantic_digest": self.semantic_digest,
            "elapsed_seconds": self.elapsed_seconds,
            "games_per_second": self.games_per_second,
            "actions_per_second": self.actions_per_second,
            "decisions_per_second": self.decisions_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Runtime/software/thread metadata captured alongside a benchmark run."""

    python_version: str
    implementation: str
    platform: str
    machine: str
    cpu_count: int | None
    active_python_threads: int
    thread_environment: tuple[tuple[str, str], ...]
    numpy_version: str | None
    torch_version: str | None
    peak_rss_bytes: int

    def payload(self) -> dict[str, object]:
        """Return the machine-readable environment fragment."""

        return {
            "python_version": self.python_version,
            "implementation": self.implementation,
            "platform": self.platform,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "active_python_threads": self.active_python_threads,
            "thread_environment": dict(self.thread_environment),
            "numpy_version": self.numpy_version,
            "torch_version": self.torch_version,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class BaselineBenchmarkReport:
    """Complete report that keeps deterministic semantics separate from live throughput."""

    config: BaselineBenchmarkConfig
    environment: EnvironmentReport
    scenarios: tuple[BaselineScenarioReport, ...]
    schema_version: int = BASELINE_REPORT_SCHEMA_VERSION

    def payload(self) -> dict[str, object]:
        """Return canonical report data suitable for a JSON artifact."""

        return {
            "schema_version": self.schema_version,
            "config": {
                "run_seed": self.config.run_seed,
                "games_per_scenario": self.config.games_per_scenario,
                "games_in_flight": self.config.games_in_flight,
                "validation_levels": [level.value for level in self.config.validation_levels],
                "max_actions_per_game": self.config.max_actions_per_game,
            },
            "environment": self.environment.payload(),
            "scenarios": [scenario.payload() for scenario in self.scenarios],
        }

    def to_json(self) -> str:
        """Encode this report in canonical JSON formatting."""

        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))

    def to_markdown(self) -> str:
        """Render a concise human-readable throughput table."""

        rows = [
            "| scenario | validation | games/s | actions/s | decisions | peak RSS (MiB) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for scenario in self.scenarios:
            rss_mib = scenario.peak_rss_bytes / (1024 * 1024)
            rows.append(
                f"| {scenario.scenario.name} | {scenario.validation_level.value} | "
                f"{scenario.games_per_second:.3f} | {scenario.actions_per_second:.1f} | "
                f"{scenario.decisions} | {rss_mib:.1f} |"
            )
        return "\n".join(rows) + "\n"


def baseline_scenarios() -> tuple[BaselineScenario, ...]:
    """Return the fixed Stage-0 random and heuristic matchup suite."""

    return (
        BaselineScenario("random-v-random", RANDOM_AGENT_DESCRIPTOR, RANDOM_AGENT_DESCRIPTOR),
        BaselineScenario(
            "heuristic-v-random",
            SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
            RANDOM_AGENT_DESCRIPTOR,
        ),
        BaselineScenario(
            "heuristic-self-play",
            SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
            SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
        ),
    )


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _peak_rss_bytes() -> int:
    # ru_maxrss is KiB on Linux and bytes on macOS/BSD. This project benchmarks Linux first, but
    # normalizing here keeps reports portable without adding psutil as a core dependency.
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows lacks resource
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def collect_environment() -> EnvironmentReport:
    """Capture dependency versions, visible thread controls, and peak process RSS."""

    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS")
    return EnvironmentReport(
        python_version=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        active_python_threads=threading.active_count(),
        thread_environment=tuple((name, os.environ[name]) for name in names if name in os.environ),
        numpy_version=_installed_version("numpy"),
        torch_version=_installed_version("torch"),
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _make_agent(
    descriptor: AgentDescriptor, run_seed: int, game_id: str, player: PlayerId
) -> Agent:
    if descriptor == RANDOM_AGENT_DESCRIPTOR:
        return RandomAgent(agent_seed(run_seed, game_id, player.value, descriptor.descriptor_id))
    if descriptor == SIMPLE_HEURISTIC_AGENT_DESCRIPTOR:
        return SimpleHeuristicAgent()
    raise ValueError(f"unsupported Stage-0 baseline descriptor: {descriptor.descriptor_id}")


def _semantic_digest(game_digests: dict[str, str]) -> str:
    digest = sha256()
    for game_id in sorted(game_digests):
        digest.update(game_id.encode("utf-8") + b"\0" + game_digests[game_id].encode("ascii"))
    return digest.hexdigest()


def run_baseline_scenario[StateT](
    engine: RunnerEngine[StateT],
    scenario: BaselineScenario,
    config: BaselineBenchmarkConfig,
    validation_level: ValidationLevel,
    *,
    metric_sink: MetricSink | None = None,
    timer_sink: TimerSink | None = None,
) -> BaselineScenarioReport:
    """Run one seeded matchup with compact action events and bounded actor retention."""

    metrics = metric_sink or NoOpMetricSink()
    timers = timer_sink or NoOpTimerSink()
    events = _ScenarioEventDigest()
    summaries: dict[str, tuple[int, str, str]] = {}
    agents: dict[str, dict[PlayerId, Agent]] = {}

    def complete(result: GameResult[StateT]) -> None:
        game_id = result.record.game_id
        summaries[game_id] = (
            events.action_counts[game_id],
            events.finish(result),
            result.record.terminal.reason.value,
        )
        agents.pop(game_id, None)

    specs = tuple(
        GameSpec(
            f"{scenario.name}-{validation_level.value}-{index:06d}",
            setup_seed(config.run_seed, f"{scenario.name}-{validation_level.value}-{index:06d}"),
        )
        for index in range(config.games_per_scenario)
    )
    recording = RunnerRecording(
        retain_actions=False,
        transition_fingerprints=False,
        action_sink=events,
    )
    pool = BoundedActorPool(
        engine,
        specs,
        max_games_in_flight=config.games_in_flight,
        on_complete=complete,
        recording=recording,
    )
    started = perf_counter()
    context: AbstractContextManager[None] = (
        validation(validation_level)
        if isinstance(engine, InnovationEngineAdapter)
        else nullcontext()
    )
    with (
        context,
        timers.timer(
            "baseline.scenario.elapsed_seconds",
            scenario=scenario.name,
            validation_level=validation_level.value,
        ),
    ):
        while not pool.is_finished:
            pending = pool.pending()
            if not pending:
                blocked = pool.blocked_game_ids()
                raise GameBlockedError(f"baseline actor pool has blocked games: {blocked}")
            submissions: list[Submission] = []
            for request in pending:
                game_agents = agents.setdefault(
                    request.game_id,
                    {
                        PlayerId.PLAYER_1: _make_agent(
                            scenario.player_1, config.run_seed, request.game_id, PlayerId.PLAYER_1
                        ),
                        PlayerId.PLAYER_2: _make_agent(
                            scenario.player_2, config.run_seed, request.game_id, PlayerId.PLAYER_2
                        ),
                    },
                )
                if events.action_counts[request.game_id] >= config.max_actions_per_game:
                    raise RuntimeError(
                        f"baseline game {request.game_id} exceeded action ceiling "
                        f"{config.max_actions_per_game}"
                    )
                submissions.append(
                    Submission(
                        request.game_id,
                        game_agents[request.decision.chooser].choose_action(request.decision),
                    )
                )
            pool.submit(submissions)
    elapsed = perf_counter() - started
    if len(summaries) != config.games_per_scenario:  # pragma: no cover - defensive
        raise RuntimeError("baseline actor pool did not produce every requested terminal summary")
    action_lengths = tuple(count for count, _, _ in summaries.values())
    reason_counts: Counter[str] = Counter(reason for _, _, reason in summaries.values())
    terminal_reasons = tuple(sorted(reason_counts.items()))
    actions = sum(action_lengths)
    report = BaselineScenarioReport(
        scenario=scenario,
        validation_level=validation_level,
        games=config.games_per_scenario,
        actions=actions,
        decisions=actions,
        action_length=DistributionSummary.from_values(action_lengths),
        decision_length=DistributionSummary.from_values(action_lengths),
        terminal_reasons=terminal_reasons,
        semantic_digest=_semantic_digest(
            {game_id: digest for game_id, (_, digest, _) in summaries.items()}
        ),
        elapsed_seconds=elapsed,
        games_per_second=config.games_per_scenario / elapsed if elapsed else 0.0,
        actions_per_second=actions / elapsed if elapsed else 0.0,
        decisions_per_second=actions / elapsed if elapsed else 0.0,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    metrics.record(
        "baseline.games",
        report.games,
        scenario=scenario.name,
        validation_level=validation_level.value,
    )
    metrics.record(
        "baseline.actions",
        report.actions,
        scenario=scenario.name,
        validation_level=validation_level.value,
    )
    metrics.record(
        "baseline.games_per_second",
        report.games_per_second,
        scenario=scenario.name,
        validation_level=validation_level.value,
    )
    return report


def run_baseline_benchmark(
    config: BaselineBenchmarkConfig,
    *,
    engine: RunnerEngine[object] | None = None,
    metric_sink: MetricSink | None = None,
    timer_sink: TimerSink | None = None,
) -> BaselineBenchmarkReport:
    """Run the complete random/heuristic suite under every requested validation level."""

    benchmark_engine = cast(RunnerEngine[object], engine or InnovationEngineAdapter())
    reports = tuple(
        run_baseline_scenario(
            benchmark_engine,
            scenario,
            config,
            level,
            metric_sink=metric_sink,
            timer_sink=timer_sink,
        )
        for level in config.validation_levels
        for scenario in baseline_scenarios()
    )
    return BaselineBenchmarkReport(config, collect_environment(), reports)
