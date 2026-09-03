"""Committed representative corpus and benchmark for Milestone 4 search feasibility.

The authoritative states in this module are corpus-construction inputs only.  Every measured case
immediately crosses the trusted information-set boundary, deterministically samples synthetic
states, and invokes the production sampled minimax implementation.  Live states are never included
in report payloads.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import cast

from innovation_ai.innovation.actions import Decision, DogmaAction, SemanticAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GameState,
    build_explicit_state,
    build_setup_state,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId
from innovation_ai.search.contracts import SearchDescriptor
from innovation_ai.search.information_sets import InformationSetSampler, InformationSetSpecBuilder
from innovation_ai.search.minimax import DeterministicSampledMinimax

SEARCH_FEASIBILITY_SCHEMA_VERSION = 1
FROZEN_ROOT_TURN_HORIZON = 1
FROZEN_OPPONENT_TURN_HORIZON = 1
FROZEN_STARTING_MELD_HORIZON = 1

_DEFAULT_CORPUS_NAMES = (
    "both-pending-starting-meld",
    "one-latent-starting-meld",
    "early-first-paid-turn",
    "explicit-second-paid-action",
    "own-turn-effect-choice",
    "opponent-demand-effect-choice",
    "opponent-shared-effect-choice",
    "late-high-branching-position",
)


class SearchFeasibilityError(RuntimeError):
    """A named corpus/configuration case failed without using a search fallback."""


@dataclass(frozen=True, slots=True)
class SearchFeasibilityConfig:
    """Deterministic benchmark inputs; the three search horizons are intentionally frozen."""

    route_transition_budgets: tuple[int, ...] = (400, 800, 1600)
    determinization_counts: tuple[int, ...] = (1, 2, 4)
    corpus_names: tuple[str, ...] = _DEFAULT_CORPUS_NAMES
    corpus_seed: int = 4404
    sampler_seed: int = 4405
    sampler_retry_limit: int = 32
    raise_on_failure: bool = True

    def __post_init__(self) -> None:
        for values, label in (
            (self.route_transition_budgets, "route transition budgets"),
            (self.determinization_counts, "determinization counts"),
        ):
            if not values or any(isinstance(value, bool) or value < 1 for value in values):
                raise ValueError(f"{label} must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")
        if not self.corpus_names or any(not name for name in self.corpus_names):
            raise ValueError("corpus names must be non-empty")
        if len(set(self.corpus_names)) != len(self.corpus_names):
            raise ValueError("corpus names must be unique")
        unknown = set(self.corpus_names) - set(_DEFAULT_CORPUS_NAMES)
        if unknown:
            raise ValueError(f"unknown search corpus entries: {sorted(unknown)}")
        if isinstance(self.sampler_retry_limit, bool) or self.sampler_retry_limit < 1:
            raise ValueError("sampler retry limit must be positive")

    def payload(self) -> dict[str, object]:
        """Return the complete JSON-safe benchmark configuration."""

        return {
            "route_transition_budgets": list(self.route_transition_budgets),
            "determinization_counts": list(self.determinization_counts),
            "corpus_names": list(self.corpus_names),
            "corpus_seed": self.corpus_seed,
            "sampler_seed": self.sampler_seed,
            "sampler_retry_limit": self.sampler_retry_limit,
            "raise_on_failure": self.raise_on_failure,
            "horizons": {
                "root_turn": FROZEN_ROOT_TURN_HORIZON,
                "opponent_turn": FROZEN_OPPONENT_TURN_HORIZON,
                "starting_meld": FROZEN_STARTING_MELD_HORIZON,
            },
        }


@dataclass(frozen=True, slots=True)
class _CorpusEntry:
    """Private authoritative fixture; only its name/category leave the benchmark runner."""

    name: str
    category: str
    state: GameState = field(repr=False)
    decision: Decision = field(repr=False)


@dataclass(frozen=True, slots=True)
class CompletedDepthCount:
    """Count of independently budgeted routes retaining one completed-turn depth."""

    depth: int
    routes: int

    def payload(self) -> dict[str, int]:
        return {"depth": self.depth, "routes": self.routes}


@dataclass(frozen=True, slots=True)
class SearchFeasibilityMeasurement:
    """One corpus root searched with one route budget and determinization count."""

    corpus_name: str
    category: str
    decision_kind: str
    chooser: str
    route_transition_budget: int
    determinization_count: int
    search_descriptor_id: str
    information_set_digest: str
    selected_action_key: str
    root_actions: int
    root_decisions: int
    routes: int
    nodes: int
    recursive_engine_transitions: int
    root_engine_transitions: int
    setup_engine_transitions: int
    total_engine_transitions: int
    transposition_hits: int
    cycle_cutoffs: int
    budget_cutoff_routes: int
    immediate_fallback_routes: int
    completed_depth_distribution: tuple[CompletedDepthCount, ...]
    wall_seconds: float
    transitions_per_second: float
    decisions_per_second: float
    peak_rss_bytes: int

    def deterministic_payload(self) -> dict[str, object]:
        """Return semantic/counter content, deliberately excluding machine-dependent values."""

        return {
            "corpus_name": self.corpus_name,
            "category": self.category,
            "decision_kind": self.decision_kind,
            "chooser": self.chooser,
            "route_transition_budget": self.route_transition_budget,
            "determinization_count": self.determinization_count,
            "search_descriptor_id": self.search_descriptor_id,
            "information_set_digest": self.information_set_digest,
            "selected_action_key": self.selected_action_key,
            "root_actions": self.root_actions,
            "root_decisions": self.root_decisions,
            "routes": self.routes,
            "nodes": self.nodes,
            "recursive_engine_transitions": self.recursive_engine_transitions,
            "root_engine_transitions": self.root_engine_transitions,
            "setup_engine_transitions": self.setup_engine_transitions,
            "total_engine_transitions": self.total_engine_transitions,
            "transposition_hits": self.transposition_hits,
            "cycle_cutoffs": self.cycle_cutoffs,
            "budget_cutoff_routes": self.budget_cutoff_routes,
            "immediate_fallback_routes": self.immediate_fallback_routes,
            "completed_depth_distribution": [
                item.payload() for item in self.completed_depth_distribution
            ],
        }

    def payload(self) -> dict[str, object]:
        """Return deterministic counters plus live timing and memory observations."""

        return {
            **self.deterministic_payload(),
            "wall_seconds": self.wall_seconds,
            "transitions_per_second": self.transitions_per_second,
            "decisions_per_second": self.decisions_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class SearchFeasibilityFailure:
    """Explicit failed case emitted only when fail-fast behavior is disabled."""

    corpus_name: str
    category: str
    route_transition_budget: int
    determinization_count: int
    error_type: str
    message: str

    def payload(self) -> dict[str, object]:
        return {
            "corpus_name": self.corpus_name,
            "category": self.category,
            "route_transition_budget": self.route_transition_budget,
            "determinization_count": self.determinization_count,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class SearchFeasibilityResult:
    """Complete machine-readable feasibility report with deterministic content identity."""

    config: SearchFeasibilityConfig
    measurements: tuple[SearchFeasibilityMeasurement, ...]
    failures: tuple[SearchFeasibilityFailure, ...] = ()
    schema_version: int = SEARCH_FEASIBILITY_SCHEMA_VERSION
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_FEASIBILITY_SCHEMA_VERSION:
            raise ValueError("unsupported search feasibility schema version")
        encoded = json.dumps(
            self.deterministic_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        object.__setattr__(self, "content_digest", f"sha256:{hashlib.sha256(encoded).hexdigest()}")

    def deterministic_payload(self) -> dict[str, object]:
        """Return identity-bearing content with all timing, rates, and RSS omitted."""

        return {
            "schema_version": self.schema_version,
            "config": self.config.payload(),
            "measurements": [item.deterministic_payload() for item in self.measurements],
            "failures": [item.payload() for item in self.failures],
        }

    def payload(self) -> dict[str, object]:
        """Return the complete JSON artifact payload."""

        return {
            "schema_version": self.schema_version,
            "content_digest": self.content_digest,
            "config": self.config.payload(),
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "measurements": [item.payload() for item in self.measurements],
            "failures": [item.payload() for item in self.failures],
        }

    def to_json(self) -> str:
        """Encode the report as stable, compact JSON."""

        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def to_markdown(self) -> str:
        """Render the feasibility sweep as a compact review table."""

        lines = [
            "# Milestone 4 search feasibility",
            "",
            f"Content digest: `{self.content_digest}`",
            "",
            "Frozen horizon: one completed player turn from every supported root.",
            "",
            (
                "| corpus | category | budget | dets | actions | routes | nodes | transitions | "
                "TT hits | cycles | budget cutoffs | fallback | depth distribution | wall (s) | "
                "trans/s |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
        for item in self.measurements:
            depths = ", ".join(
                f"{depth.depth}:{depth.routes}" for depth in item.completed_depth_distribution
            )
            lines.append(
                f"| {item.corpus_name} | {item.category} | {item.route_transition_budget} | "
                f"{item.determinization_count} | {item.root_actions} | {item.routes} | "
                f"{item.nodes} | {item.total_engine_transitions} | {item.transposition_hits} | "
                f"{item.cycle_cutoffs} | {item.budget_cutoff_routes} | "
                f"{item.immediate_fallback_routes} | {depths} | {item.wall_seconds:.6f} | "
                f"{item.transitions_per_second:.1f} |"
            )
        if self.failures:
            lines.extend(("", "## Failures", ""))
            lines.extend(
                f"- **{item.corpus_name}** (budget {item.route_transition_budget}, "
                f"{item.determinization_count} determinizations): "
                f"`{item.error_type}: {item.message}`"
                for item in self.failures
            )
        return "\n".join(lines) + "\n"


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
    state: GameState,
    card: str,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> _CorpusEntry:
    root = current_decisions(state, registry, programs)[0]
    transition = apply_action(state, _dogma_action(root, card), registry, programs)
    if transition.decision is None:
        raise SearchFeasibilityError(f"corpus fixture {name} did not pause at an effect choice")
    return _CorpusEntry(name, category, transition.state, transition.decision)


def _representative_corpus(
    seed: int,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> tuple[_CorpusEntry, ...]:
    """Construct the committed roots with validated production state/transition helpers."""

    setup = build_setup_state(seed, registry)
    setup_decisions = current_decisions(setup, registry, programs)
    both_pending = _CorpusEntry("both-pending-starting-meld", "setup", setup, setup_decisions[0])

    one_committed = apply_action(
        setup, setup_decisions[0].legal_actions[0], registry, programs
    ).state
    latent_decision = current_decisions(one_committed, registry, programs)[0]
    one_latent = _CorpusEntry(
        "one-latent-starting-meld", "setup-latent", one_committed, latent_decision
    )

    play = apply_action(one_committed, latent_decision.legal_actions[0], registry, programs).state
    first_paid_decision = current_decisions(play, registry, programs)[0]
    early = _CorpusEntry("early-first-paid-turn", "early", play, first_paid_decision)

    opponent_first_paid = apply_action(
        play, first_paid_decision.legal_actions[0], registry, programs
    ).state
    opponent_first_decision = current_decisions(opponent_first_paid, registry, programs)[0]
    second_paid_state = apply_action(
        opponent_first_paid, opponent_first_decision.legal_actions[0], registry, programs
    ).state
    second_paid = _CorpusEntry(
        "explicit-second-paid-action",
        "early-second-action",
        second_paid_state,
        current_decisions(second_paid_state, registry, programs)[0],
    )

    own_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(CardId("tools"),),
                    score_pile=(CardId("alchemy"),),
                    board=((Color.YELLOW, (CardId("canal-building"),)),),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.RED, (CardId("archery"),)),)),
            ),
        ),
        turn_number=5,
    )
    own_effect = _effect_entry(
        "own-turn-effect-choice", "effect-own-turn", own_state, "canal-building", registry, programs
    )

    demand_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(board=((Color.RED, (CardId("archery"),)),)),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(
                    hand=(CardId("canal-building"), CardId("construction")),
                    board=((Color.BLUE, (CardId("pottery"),)),),
                ),
            ),
        ),
        supply_tops=((1, (CardId("writing"),)),),
        turn_number=9,
    )
    opponent_demand = _effect_entry(
        "opponent-demand-effect-choice",
        "effect-opponent-demand",
        demand_state,
        "archery",
        registry,
        programs,
    )

    shared_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(CardId("tools"),),
                    board=((Color.YELLOW, (CardId("canal-building"),)),),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(
                    hand=(CardId("writing"),),
                    board=((Color.GREEN, (CardId("mapmaking"),)),),
                ),
            ),
        ),
        turn_number=11,
    )
    opponent_shared = _effect_entry(
        "opponent-shared-effect-choice",
        "effect-opponent-shared",
        shared_state,
        "canal-building",
        registry,
        programs,
    )

    late_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(
                        CardId("pottery"),
                        CardId("writing"),
                        CardId("agriculture"),
                        CardId("clothing"),
                        CardId("construction"),
                        CardId("currency"),
                        CardId("alchemy"),
                        CardId("compass"),
                    ),
                    board=(
                        (Color.RED, (CardId("archery"),)),
                        (Color.BLUE, (CardId("tools"),)),
                        (Color.GREEN, (CardId("mapmaking"),)),
                        (Color.YELLOW, (CardId("canal-building"),)),
                        (Color.PURPLE, (CardId("philosophy"),)),
                    ),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.PURPLE, (CardId("mysticism"),)),)),
            ),
        ),
        turn_number=31,
    )
    late = _CorpusEntry(
        "late-high-branching-position",
        "late-high-branching",
        late_state,
        current_decisions(late_state, registry, programs)[0],
    )
    return (
        both_pending,
        one_latent,
        early,
        second_paid,
        own_effect,
        opponent_demand,
        opponent_shared,
        late,
    )


def _peak_rss_bytes() -> int:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def _measurement(
    entry: _CorpusEntry,
    config: SearchFeasibilityConfig,
    budget: int,
    determinizations: int,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> SearchFeasibilityMeasurement:
    started = perf_counter()
    spec = InformationSetSpecBuilder(registry, programs).build(entry.state, entry.decision)
    sampled = InformationSetSampler(
        registry,
        programs,
        retry_limit=config.sampler_retry_limit,
        strict=True,
    ).sample_many(
        spec,
        determinizations,
        f"search-feasibility:{config.sampler_seed}:{entry.name}",
    )
    if any(state is None for state in sampled):  # pragma: no cover - strict sampler contract
        raise SearchFeasibilityError("strict information-set sampling returned no state")
    descriptor = SearchDescriptor(
        root_turn_horizon=FROZEN_ROOT_TURN_HORIZON,
        opponent_turn_horizon=FROZEN_OPPONENT_TURN_HORIZON,
        starting_meld_horizon=FROZEN_STARTING_MELD_HORIZON,
        determinization_count=determinizations,
        route_transition_budget=budget,
    )
    samples = cast(tuple[GameState, ...], sampled)
    result = DeterministicSampledMinimax(descriptor, registry=registry, programs=programs).select(
        spec, samples
    )
    elapsed = perf_counter() - started
    stats = result.statistics
    depths = Counter(route.completed_turn_depth for route in result.telemetry.routes)
    return SearchFeasibilityMeasurement(
        corpus_name=entry.name,
        category=entry.category,
        decision_kind=entry.decision.kind.value,
        chooser=entry.decision.chooser.value,
        route_transition_budget=budget,
        determinization_count=determinizations,
        search_descriptor_id=descriptor.descriptor_id,
        information_set_digest=spec.spec_digest,
        selected_action_key=result.telemetry.selected_action_key,
        root_actions=len(spec.legal_actions),
        root_decisions=1,
        routes=stats.routes,
        nodes=stats.nodes,
        recursive_engine_transitions=stats.recursive_engine_transitions,
        root_engine_transitions=stats.root_transitions,
        setup_engine_transitions=stats.mandatory_setup_transitions,
        total_engine_transitions=stats.total_engine_transitions,
        transposition_hits=stats.transposition_hits,
        cycle_cutoffs=stats.repeated_position_cutoffs,
        budget_cutoff_routes=stats.budget_cutoff_routes,
        immediate_fallback_routes=stats.immediate_leaf_fallback_routes,
        completed_depth_distribution=tuple(
            CompletedDepthCount(depth, count) for depth, count in sorted(depths.items())
        ),
        wall_seconds=elapsed,
        transitions_per_second=stats.total_engine_transitions / elapsed if elapsed else 0.0,
        decisions_per_second=1.0 / elapsed if elapsed else 0.0,
        peak_rss_bytes=_peak_rss_bytes(),
    )


def run_search_feasibility(
    config: SearchFeasibilityConfig,
) -> SearchFeasibilityResult:
    """Run the requested deterministic corpus sweep, raising contextual failures by default."""

    registry = load_card_registry()
    programs = load_effect_programs()
    entries = tuple(
        entry
        for entry in _representative_corpus(config.corpus_seed, registry, programs)
        if entry.name in config.corpus_names
    )
    measurements: list[SearchFeasibilityMeasurement] = []
    failures: list[SearchFeasibilityFailure] = []
    for entry in entries:
        for budget in config.route_transition_budgets:
            for determinizations in config.determinization_counts:
                try:
                    measurements.append(
                        _measurement(
                            entry,
                            config,
                            budget,
                            determinizations,
                            registry,
                            programs,
                        )
                    )
                except Exception as error:
                    failure = SearchFeasibilityFailure(
                        entry.name,
                        entry.category,
                        budget,
                        determinizations,
                        type(error).__name__,
                        str(error),
                    )
                    if config.raise_on_failure:
                        raise SearchFeasibilityError(
                            f"search feasibility failed for {entry.name} at budget={budget}, "
                            f"determinizations={determinizations}: "
                            f"{failure.error_type}: {failure.message}"
                        ) from error
                    failures.append(failure)
    return SearchFeasibilityResult(config, tuple(measurements), tuple(failures))


def write_search_feasibility_json(result: SearchFeasibilityResult, path: str | Path) -> Path:
    """Write one complete JSON report and return its path."""

    target = Path(path)
    target.write_text(result.to_json() + "\n", encoding="utf-8")
    return target


def write_search_feasibility_markdown(result: SearchFeasibilityResult, path: str | Path) -> Path:
    """Write one human-readable Markdown report and return its path."""

    target = Path(path)
    target.write_text(result.to_markdown(), encoding="utf-8")
    return target


# Short aliases for callers that already import this module as ``search.benchmark``.
BenchmarkConfig = SearchFeasibilityConfig
BenchmarkResult = SearchFeasibilityResult
write_json = write_search_feasibility_json
write_markdown = write_search_feasibility_markdown


__all__ = [
    "FROZEN_OPPONENT_TURN_HORIZON",
    "FROZEN_ROOT_TURN_HORIZON",
    "FROZEN_STARTING_MELD_HORIZON",
    "SEARCH_FEASIBILITY_SCHEMA_VERSION",
    "BenchmarkConfig",
    "BenchmarkResult",
    "CompletedDepthCount",
    "SearchFeasibilityConfig",
    "SearchFeasibilityError",
    "SearchFeasibilityFailure",
    "SearchFeasibilityMeasurement",
    "SearchFeasibilityResult",
    "run_search_feasibility",
    "write_json",
    "write_markdown",
    "write_search_feasibility_json",
    "write_search_feasibility_markdown",
]
