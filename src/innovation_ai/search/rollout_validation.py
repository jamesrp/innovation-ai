"""Deterministic R4 route-length and end-to-end rollout smoke measurements.

The reference soak deliberately extends beyond the small feasibility corpus: it evaluates fixed
snapshots from seeded games, one paid-action root for every base-game card, and focused high-choice
roots.  The generation and arena smokes route the rollout baseline through the existing immutable
self-play and paired-arena executors; they measure only completion and committed-action throughput,
never playing strength.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from innovation_ai.agents.descriptors import (
    RANDOM_AGENT_DESCRIPTOR,
    SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR,
    AgentDescriptor,
)
from innovation_ai.agents.heuristic import SimpleHeuristicAgent
from innovation_ai.innovation.actions import Decision, DogmaAction
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
from innovation_ai.innovation.zones import ValidationLevel, validation
from innovation_ai.search.contracts import (
    EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING,
    RolloutSearchDescriptor,
)
from innovation_ai.search.information_sets import InformationSetSampler, InformationSetSpecBuilder
from innovation_ai.search.rollout import DeterministicSampledRollout, RolloutSafetyCeilingError
from innovation_ai.search.seeds import seed_digest

ROLLOUT_ROUTE_LENGTH_SOAK_FORMAT = "innovation-ai-rollout-route-length-soak"
ROLLOUT_THROUGHPUT_SMOKE_FORMAT = "innovation-ai-rollout-throughput-smoke"
ROLLOUT_VALIDATION_SCHEMA_VERSION = 1
DEFAULT_ROUTE_SOAK_GAME_SEEDS = tuple(range(46_000, 46_016))
DEFAULT_ROUTE_SOAK_DECISION_INDICES = (0, 4, 8, 12, 16, 20, 24, 28, 32)
DEFAULT_ROUTE_SOAK_HIGH_CHOICE_ROOTS = 2
DEFAULT_ROLLOUT_DETERMINIZATION_COUNT = 2


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _payload_digest(payload: object) -> str:
    encoded = _canonical_json(payload).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _peak_rss_bytes() -> int:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class RolloutRouteLengthSoakConfig:
    """Fixed route-soak inputs; reduced selections are allowed only for fast diagnostics."""

    seeded_game_seeds: tuple[int, ...] = DEFAULT_ROUTE_SOAK_GAME_SEEDS
    seeded_decision_indices: tuple[int, ...] = DEFAULT_ROUTE_SOAK_DECISION_INDICES
    include_all_cards: bool = True
    selected_card_ids: tuple[str, ...] = ()
    include_high_choice_roots: bool = True
    determinization_count: int = DEFAULT_ROLLOUT_DETERMINIZATION_COUNT
    route_transition_ceiling: int = EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING
    sampler_seed: int = 46_101
    sampler_retry_limit: int = 32
    validation_level: ValidationLevel = ValidationLevel.FULL

    def __post_init__(self) -> None:
        if (
            not self.seeded_game_seeds
            and not self.include_all_cards
            and not self.selected_card_ids
            and not self.include_high_choice_roots
        ):
            raise ValueError("route-length soak must contain at least one root source")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeded_game_seeds
        ):
            raise ValueError("route-length soak game seeds must be integers")
        if len(set(self.seeded_game_seeds)) != len(self.seeded_game_seeds):
            raise ValueError("route-length soak game seeds must be unique")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.seeded_decision_indices
        ):
            raise ValueError("route-length soak decision indices must be non-negative integers")
        if tuple(sorted(set(self.seeded_decision_indices))) != self.seeded_decision_indices:
            raise ValueError("route-length soak decision indices must be unique ascending values")
        if self.seeded_game_seeds and not self.seeded_decision_indices:
            raise ValueError("seeded route-length games require decision indices")
        if self.include_all_cards and self.selected_card_ids:
            raise ValueError("all-card and selected-card route scopes are mutually exclusive")
        if len(set(self.selected_card_ids)) != len(self.selected_card_ids) or any(
            not card_id for card_id in self.selected_card_ids
        ):
            raise ValueError("selected route-length cards must be unique non-empty IDs")
        if (
            isinstance(self.determinization_count, bool)
            or not isinstance(self.determinization_count, int)
            or self.determinization_count < 1
        ):
            raise ValueError("route-length soak determinization count must be positive")
        if self.route_transition_ceiling != EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING:
            raise ValueError("route-length soak requires the experimental 1024-transition ceiling")
        if isinstance(self.sampler_seed, bool) or not isinstance(self.sampler_seed, int):
            raise ValueError("route-length soak sampler seed must be an integer")
        if (
            isinstance(self.sampler_retry_limit, bool)
            or not isinstance(self.sampler_retry_limit, int)
            or self.sampler_retry_limit < 1
        ):
            raise ValueError("route-length soak sampler retry limit must be positive")
        if not isinstance(self.validation_level, ValidationLevel):
            raise ValueError("route-length soak validation level must be a ValidationLevel")

    @property
    def is_reference_protocol(self) -> bool:
        return self == type(self)()

    def payload(self) -> dict[str, object]:
        return {
            "seeded_game_seeds": list(self.seeded_game_seeds),
            "seeded_decision_indices": list(self.seeded_decision_indices),
            "include_all_cards": self.include_all_cards,
            "selected_card_ids": list(self.selected_card_ids),
            "include_high_choice_roots": self.include_high_choice_roots,
            "determinization_count": self.determinization_count,
            "route_transition_ceiling": self.route_transition_ceiling,
            "sampler_seed": self.sampler_seed,
            "sampler_retry_limit": self.sampler_retry_limit,
            "validation_level": self.validation_level.value,
        }

    def public_payload(self) -> dict[str, object]:
        return {
            "config_digest": self.config_digest,
            "seeded_game_count": len(self.seeded_game_seeds),
            "seeded_decisions_per_game": len(self.seeded_decision_indices),
            "include_all_cards": self.include_all_cards,
            "selected_card_count": len(self.selected_card_ids),
            "include_high_choice_roots": self.include_high_choice_roots,
            "determinization_count": self.determinization_count,
            "route_transition_ceiling": self.route_transition_ceiling,
            "sampler_retry_limit": self.sampler_retry_limit,
            "validation_level": self.validation_level.value,
            "reference_protocol": self.is_reference_protocol,
        }

    @property
    def config_digest(self) -> str:
        return _payload_digest(self.payload())


@dataclass(frozen=True, slots=True)
class RolloutRouteLengthRootMeasurement:
    root_id: str
    source_kind: str
    decision_kind: str
    root_actions: int
    routes: int
    completed_routes: int
    terminal_routes: int
    safety_ceiling_hits: int
    maximum_route_transitions: int

    def payload(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "source_kind": self.source_kind,
            "decision_kind": self.decision_kind,
            "root_actions": self.root_actions,
            "routes": self.routes,
            "completed_routes": self.completed_routes,
            "terminal_routes": self.terminal_routes,
            "safety_ceiling_hits": self.safety_ceiling_hits,
            "maximum_route_transitions": self.maximum_route_transitions,
        }


@dataclass(frozen=True, slots=True)
class RolloutValidationFailure:
    root_id: str
    source_kind: str
    stage: str
    error_type: str
    message: str
    observed_route_transitions: int = 0
    safety_ceiling_hit: bool = False

    def payload(self, *, public: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "root_id": self.root_id,
            "source_kind": self.source_kind,
            "stage": self.stage,
            "error_type": self.error_type,
            "observed_route_transitions": self.observed_route_transitions,
            "safety_ceiling_hit": self.safety_ceiling_hit,
        }
        if not public:
            payload["message"] = self.message
        return payload


@dataclass(frozen=True, slots=True)
class RolloutRouteLengthSoakResult:
    config: RolloutRouteLengthSoakConfig
    planned_roots: int
    seeded_roots: int
    card_roots: int
    high_choice_roots: int
    measurements: tuple[RolloutRouteLengthRootMeasurement, ...]
    failures: tuple[RolloutValidationFailure, ...]
    elapsed_seconds: float
    peak_rss_bytes: int
    format: str = ROLLOUT_ROUTE_LENGTH_SOAK_FORMAT
    schema_version: int = ROLLOUT_VALIDATION_SCHEMA_VERSION
    deterministic_content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.format != ROLLOUT_ROUTE_LENGTH_SOAK_FORMAT:
            raise ValueError("unsupported rollout route-length soak format")
        if self.schema_version != ROLLOUT_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported rollout validation schema version")
        for value, name in (
            (self.planned_roots, "planned roots"),
            (self.seeded_roots, "seeded roots"),
            (self.card_roots, "card roots"),
            (self.high_choice_roots, "high-choice roots"),
            (self.peak_rss_bytes, "peak RSS"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"route-length soak {name} must be a non-negative integer")
        if self.planned_roots != self.seeded_roots + self.card_roots + self.high_choice_roots:
            raise ValueError("route-length soak source counts must sum to planned roots")
        if len(self.measurements) + len(self.failures) != self.planned_roots:
            raise ValueError("route-length soak must account for every planned root")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("route-length soak elapsed time must be finite and non-negative")
        object.__setattr__(
            self,
            "deterministic_content_digest",
            _payload_digest(self.deterministic_payload()),
        )

    @property
    def routes(self) -> int:
        return sum(item.routes for item in self.measurements)

    @property
    def completed_routes(self) -> int:
        return sum(item.completed_routes for item in self.measurements)

    @property
    def safety_ceiling_hits(self) -> int:
        return sum(item.safety_ceiling_hits for item in self.measurements) + sum(
            item.safety_ceiling_hit for item in self.failures
        )

    @property
    def maximum_route_transitions(self) -> int:
        return max(
            *(item.maximum_route_transitions for item in self.measurements),
            *(item.observed_route_transitions for item in self.failures),
            0,
        )

    @property
    def all_routes_complete(self) -> bool:
        return bool(self.measurements) and self.completed_routes == self.routes

    def deterministic_payload(self, *, public: bool = False) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "config": self.config.public_payload() if public else self.config.payload(),
            "planned_roots": self.planned_roots,
            "source_root_counts": {
                "seeded": self.seeded_roots,
                "all_card_or_selected_card": self.card_roots,
                "high_choice": self.high_choice_roots,
            },
            "measurements": [item.payload() for item in self.measurements],
            "failures": [item.payload(public=public) for item in self.failures],
        }

    def payload(self, *, public: bool = False) -> dict[str, object]:
        return {
            **self.deterministic_payload(public=public),
            "deterministic_content_digest": self.deterministic_content_digest,
            "summary": {
                "measured_roots": len(self.measurements),
                "routes": self.routes,
                "completed_routes": self.completed_routes,
                "all_routes_complete": self.all_routes_complete,
                "failure_count": len(self.failures),
                "safety_ceiling_hits": self.safety_ceiling_hits,
                "maximum_route_transitions": self.maximum_route_transitions,
                "two_x_headroom_under_1024": self.maximum_route_transitions <= 512,
            },
            "runtime": {
                "elapsed_seconds": self.elapsed_seconds,
                "peak_rss_bytes": self.peak_rss_bytes,
            },
        }

    def to_json(self, *, public: bool = False) -> str:
        return _canonical_json(self.payload(public=public))


@dataclass(frozen=True, slots=True)
class _RouteRoot:
    root_id: str
    source_kind: str
    state: GameState = field(repr=False)
    decision: Decision = field(repr=False)


def _seeded_roots(
    config: RolloutRouteLengthSoakConfig,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> tuple[tuple[_RouteRoot, ...], tuple[RolloutValidationFailure, ...]]:
    roots: list[_RouteRoot] = []
    failures: list[RolloutValidationFailure] = []
    walker = SimpleHeuristicAgent(registry)
    targets = set(config.seeded_decision_indices)
    maximum = max(config.seeded_decision_indices, default=-1)
    for seed_index, seed in enumerate(config.seeded_game_seeds):
        collected: set[int] = set()
        construction_error: Exception | None = None
        try:
            state = build_setup_state(seed, registry)
            for transition_index in range(maximum + 1):
                decisions = current_decisions(state, registry, programs)
                if not decisions:
                    break
                decision = decisions[0]
                if transition_index in targets:
                    roots.append(
                        _RouteRoot(
                            f"seeded-{seed_index:03d}-decision-{transition_index:04d}",
                            "seeded-game-decision",
                            state,
                            decision,
                        )
                    )
                    collected.add(transition_index)
                action = walker.choose_action(decision)
                state = apply_action(state, action, registry, programs).state
        except Exception as error:
            construction_error = error
        for missing in sorted(targets - collected):
            failures.append(
                RolloutValidationFailure(
                    f"seeded-{seed_index:03d}-decision-{missing:04d}",
                    "seeded-game-decision",
                    "root-construction",
                    (
                        "MissingSeededDecision"
                        if construction_error is None
                        else type(construction_error).__name__
                    ),
                    (
                        "seeded game terminated before the requested decision index"
                        if construction_error is None
                        else str(construction_error)
                    ),
                )
            )
    return tuple(roots), tuple(failures)


def _card_roots(
    config: RolloutRouteLengthSoakConfig,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> tuple[tuple[_RouteRoot, ...], tuple[RolloutValidationFailure, ...]]:
    requested = (
        tuple(card.id for card in sorted(registry.cards, key=lambda item: str(item.id)))
        if config.include_all_cards
        else tuple(CardId(value) for value in config.selected_card_ids)
    )
    roots: list[_RouteRoot] = []
    failures: list[RolloutValidationFailure] = []
    for index, card_id in enumerate(requested):
        try:
            card = registry.card(card_id)
            state = build_explicit_state(
                registry,
                positions=(
                    (
                        PlayerId.PLAYER_1,
                        ExplicitPlayerPosition(board=((card.color, (card_id,)),)),
                    ),
                ),
                turn_number=100 + index,
                paid_actions_remaining=2,
            )
            decision = current_decisions(state, registry, programs)[0]
            if not any(
                isinstance(action, DogmaAction) and action.card_id == card_id
                for action in decision.legal_actions
            ):
                raise ValueError(f"all-card route root does not expose dogma for {card_id}")
            roots.append(_RouteRoot(f"card-{card_id.value}", "all-card-dogma", state, decision))
        except Exception as error:
            failures.append(
                RolloutValidationFailure(
                    f"card-{card_id.value}",
                    "all-card-dogma",
                    "root-construction",
                    type(error).__name__,
                    str(error),
                )
            )
    return tuple(roots), tuple(failures)


def _high_choice_hand(
    registry: CardRegistry,
    excluded: set[CardId],
    count: int,
) -> tuple[CardId, ...]:
    reserved: set[CardId] = set()
    for age in range(1, 10):
        available = sorted(
            (card.id for card in registry.cards if card.age == age and card.id not in excluded),
            key=str,
        )
        reserved.add(available[0])
    candidates = sorted(
        (card.id for card in registry.cards if card.id not in excluded and card.id not in reserved),
        key=lambda card_id: (registry.card(card_id).age, str(card_id)),
    )
    if len(candidates) < count:
        raise ValueError("insufficient cards for high-choice soak fixture")
    return tuple(candidates[:count])


def _high_choice_roots(
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> tuple[_RouteRoot, ...]:
    board_cards: list[CardId] = []
    for color in Color:
        board_cards.append(
            min(
                (card.id for card in registry.cards if card.color is color),
                key=lambda card_id: (registry.card(card_id).age, str(card_id)),
            )
        )
    excluded = set(board_cards)
    hand = _high_choice_hand(registry, excluded, 40)
    board = tuple(
        (
            color,
            (next(card_id for card_id in board_cards if registry.card(card_id).color is color),),
        )
        for color in Color
    )
    paid_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(hand=hand, board=board),
            ),
        ),
        turn_number=301,
        paid_actions_remaining=2,
    )
    paid_decision = current_decisions(paid_state, registry, programs)[0]

    empiricism = CardId("empiricism")
    effect_state = build_explicit_state(
        registry,
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(CardId("agriculture"), CardId("construction")),
                    board=((registry.card(empiricism).color, (empiricism,)),),
                ),
            ),
        ),
        supply_tops=((9, (CardId("collaboration"),)),),
        turn_number=302,
        paid_actions_remaining=2,
    )
    paid_root = current_decisions(effect_state, registry, programs)[0]
    dogma = next(
        action
        for action in paid_root.legal_actions
        if isinstance(action, DogmaAction) and action.card_id == empiricism
    )
    transition = apply_action(effect_state, dogma, registry, programs)
    if transition.decision is None or len(transition.decision.legal_actions) < 10:
        raise ValueError("high-choice effect fixture did not expose ten choices")
    return (
        _RouteRoot("high-choice-paid-action", "high-choice-paid-action", paid_state, paid_decision),
        _RouteRoot(
            "high-choice-empiricism-effect",
            "high-choice-effect",
            transition.state,
            transition.decision,
        ),
    )


def _measure_route_root(
    root: _RouteRoot,
    config: RolloutRouteLengthSoakConfig,
    registry: CardRegistry,
    programs: EffectProgramRegistry,
) -> RolloutRouteLengthRootMeasurement:
    sampler_seed = f"rollout-route-soak:{config.sampler_seed}:{root.root_id}".encode()
    spec = InformationSetSpecBuilder(registry, programs).build(root.state, root.decision)
    sampled = InformationSetSampler(
        registry,
        programs,
        retry_limit=config.sampler_retry_limit,
        strict=True,
    ).sample_many(spec, config.determinization_count, sampler_seed)
    if any(state is None for state in sampled):
        raise ValueError("strict route-soak sampler returned no synthetic state")
    descriptor = RolloutSearchDescriptor(
        determinization_count=config.determinization_count,
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
    routes = selection.telemetry.routes
    completed = sum(
        route.completed_turn_count == 1 or route.terminal_reason is not None for route in routes
    )
    return RolloutRouteLengthRootMeasurement(
        root.root_id,
        root.source_kind,
        root.decision.kind.value,
        len(root.decision.legal_actions),
        selection.statistics.routes,
        completed,
        selection.statistics.terminal_routes,
        sum(route.safety_ceiling_hit for route in routes),
        selection.statistics.maximum_route_transitions,
    )


def run_rollout_route_length_soak(
    config: RolloutRouteLengthSoakConfig | None = None,
) -> RolloutRouteLengthSoakResult:
    """Run every configured root independently and retain failures instead of stopping early."""

    config = config or RolloutRouteLengthSoakConfig()
    registry = load_card_registry()
    programs = load_effect_programs()
    started = perf_counter()
    seeded, construction_failures = _seeded_roots(config, registry, programs)
    cards, card_failures = _card_roots(config, registry, programs)
    high_choice_failures: tuple[RolloutValidationFailure, ...] = ()
    if config.include_high_choice_roots:
        try:
            high_choice = _high_choice_roots(registry, programs)
        except Exception as error:
            high_choice = ()
            high_choice_failures = tuple(
                RolloutValidationFailure(
                    root_id,
                    source_kind,
                    "root-construction",
                    type(error).__name__,
                    str(error),
                )
                for root_id, source_kind in (
                    ("high-choice-paid-action", "high-choice-paid-action"),
                    ("high-choice-empiricism-effect", "high-choice-effect"),
                )
            )
    else:
        high_choice = ()
    roots = (*seeded, *cards, *high_choice)
    measurements: list[RolloutRouteLengthRootMeasurement] = []
    failures = [*construction_failures, *card_failures, *high_choice_failures]
    with validation(config.validation_level):
        for root in roots:
            try:
                measurements.append(_measure_route_root(root, config, registry, programs))
            except Exception as error:
                observed = 0
                ceiling_hit = False
                if isinstance(error, RolloutSafetyCeilingError):
                    observed = error.trace.route.total_engine_transitions
                    ceiling_hit = True
                failures.append(
                    RolloutValidationFailure(
                        root.root_id,
                        root.source_kind,
                        "rollout-selection",
                        type(error).__name__,
                        str(error),
                        observed,
                        ceiling_hit,
                    )
                )
    seeded_planned = len(config.seeded_game_seeds) * len(config.seeded_decision_indices)
    card_planned = (
        len(registry.cards) if config.include_all_cards else len(config.selected_card_ids)
    )
    high_choice_planned = (
        DEFAULT_ROUTE_SOAK_HIGH_CHOICE_ROOTS if config.include_high_choice_roots else 0
    )
    return RolloutRouteLengthSoakResult(
        config,
        seeded_planned + card_planned + high_choice_planned,
        seeded_planned,
        card_planned,
        high_choice_planned,
        tuple(measurements),
        tuple(failures),
        perf_counter() - started,
        _peak_rss_bytes(),
    )


@dataclass(frozen=True, slots=True)
class RolloutGenerationSmokeConfig:
    run_seed: int = 46_201
    games: int = 2
    generation: int = 0
    games_in_flight: int = 1
    shard_episode_limit: int = 2
    action_ceiling: int = 10_000
    validation_level: str = "cheap"
    determinization_count: int = DEFAULT_ROLLOUT_DETERMINIZATION_COUNT
    route_transition_ceiling: int = EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING

    def __post_init__(self) -> None:
        for value, name, minimum in (
            (self.run_seed, "run seed", 0),
            (self.games, "games", 1),
            (self.generation, "generation", 0),
            (self.games_in_flight, "games in flight", 1),
            (self.shard_episode_limit, "shard episode limit", 1),
            (self.action_ceiling, "action ceiling", 1),
            (self.determinization_count, "determinization count", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"generation smoke {name} must be an integer >= {minimum}")
        if self.validation_level not in {"full", "cheap", "off"}:
            raise ValueError("generation smoke validation level is invalid")
        if self.route_transition_ceiling != EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING:
            raise ValueError("generation smoke requires the experimental 1024-transition ceiling")

    @property
    def is_reference_protocol(self) -> bool:
        return self == type(self)()

    def payload(self) -> dict[str, object]:
        return {
            "run_seed": self.run_seed,
            "games": self.games,
            "generation": self.generation,
            "games_in_flight": self.games_in_flight,
            "shard_episode_limit": self.shard_episode_limit,
            "action_ceiling": self.action_ceiling,
            "validation_level": self.validation_level,
            "determinization_count": self.determinization_count,
            "route_transition_ceiling": self.route_transition_ceiling,
        }

    @property
    def config_digest(self) -> str:
        return _payload_digest(self.payload())


@dataclass(frozen=True, slots=True)
class RolloutArenaSmokeConfig:
    seed_start: int = 46_301
    seed_pairs: int = 1
    run_seed: int = 46_302
    bootstrap_seed: int = 46_303
    bootstrap_resamples: int = 1_000
    action_ceiling: int = 10_000
    validation_level: ValidationLevel = ValidationLevel.CHEAP
    determinization_count: int = DEFAULT_ROLLOUT_DETERMINIZATION_COUNT
    route_transition_ceiling: int = EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING

    def __post_init__(self) -> None:
        for value, name, minimum in (
            (self.seed_start, "seed start", 0),
            (self.seed_pairs, "seed pairs", 1),
            (self.run_seed, "run seed", 0),
            (self.bootstrap_seed, "bootstrap seed", 0),
            (self.bootstrap_resamples, "bootstrap resamples", 1),
            (self.action_ceiling, "action ceiling", 1),
            (self.determinization_count, "determinization count", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"arena smoke {name} must be an integer >= {minimum}")
        if not isinstance(self.validation_level, ValidationLevel):
            raise ValueError("arena smoke validation level must be a ValidationLevel")
        if self.route_transition_ceiling != EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING:
            raise ValueError("arena smoke requires the experimental 1024-transition ceiling")

    @property
    def games(self) -> int:
        return self.seed_pairs * 2

    @property
    def is_reference_protocol(self) -> bool:
        return self == type(self)()

    def payload(self) -> dict[str, object]:
        return {
            "seed_start": self.seed_start,
            "seed_pairs": self.seed_pairs,
            "run_seed": self.run_seed,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_resamples": self.bootstrap_resamples,
            "action_ceiling": self.action_ceiling,
            "validation_level": self.validation_level.value,
            "determinization_count": self.determinization_count,
            "route_transition_ceiling": self.route_transition_ceiling,
        }

    @property
    def config_digest(self) -> str:
        return _payload_digest(self.payload())


RolloutThroughputSmokeConfig = RolloutGenerationSmokeConfig | RolloutArenaSmokeConfig


@dataclass(frozen=True, slots=True)
class RolloutThroughputSmokeResult:
    smoke_kind: Literal["generation", "arena"]
    config: RolloutThroughputSmokeConfig
    completed_games: int
    committed_actions: int
    elapsed_seconds: float
    failures: tuple[RolloutValidationFailure, ...] = ()
    rollout_decisions: int | None = None
    maximum_route_transitions: int = 0
    peak_rss_bytes: int = 0
    format: str = ROLLOUT_THROUGHPUT_SMOKE_FORMAT
    schema_version: int = ROLLOUT_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != ROLLOUT_THROUGHPUT_SMOKE_FORMAT:
            raise ValueError("unsupported rollout throughput smoke format")
        if self.schema_version != ROLLOUT_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported rollout validation schema version")
        expected_type = (
            RolloutGenerationSmokeConfig
            if self.smoke_kind == "generation"
            else RolloutArenaSmokeConfig
            if self.smoke_kind == "arena"
            else None
        )
        if expected_type is None or not isinstance(self.config, expected_type):
            raise ValueError("throughput smoke kind and configuration type differ")
        for value, name in (
            (self.completed_games, "completed games"),
            (self.committed_actions, "committed actions"),
            (self.maximum_route_transitions, "maximum route transitions"),
            (self.peak_rss_bytes, "peak RSS"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"throughput smoke {name} must be a non-negative integer")
        if self.rollout_decisions is not None and (
            isinstance(self.rollout_decisions, bool)
            or not isinstance(self.rollout_decisions, int)
            or self.rollout_decisions < 0
        ):
            raise ValueError("throughput smoke rollout decisions must be non-negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("throughput smoke elapsed time must be finite and non-negative")

    @property
    def expected_games(self) -> int:
        return self.config.games

    @property
    def actions_per_second(self) -> float:
        return self.committed_actions / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def complete(self) -> bool:
        return (
            not self.failures
            and self.completed_games == self.expected_games
            and self.committed_actions > 0
        )

    @property
    def is_reference_protocol(self) -> bool:
        return self.config.is_reference_protocol

    def deterministic_payload(self, *, public: bool = False) -> dict[str, object]:
        config_payload: dict[str, object]
        if public:
            config_payload = {
                "config_digest": self.config.config_digest,
                "games": self.expected_games,
                "determinization_count": self.config.determinization_count,
                "route_transition_ceiling": self.config.route_transition_ceiling,
                "reference_protocol": self.config.is_reference_protocol,
            }
        else:
            config_payload = self.config.payload()
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "smoke_kind": self.smoke_kind,
            "config": config_payload,
            "completed_games": self.completed_games,
            "expected_games": self.expected_games,
            "committed_actions": self.committed_actions,
            "complete": self.complete,
            "rollout_decisions": self.rollout_decisions,
            "maximum_route_transitions": self.maximum_route_transitions,
            "failures": [failure.payload(public=public) for failure in self.failures],
        }

    def payload(self, *, public: bool = False) -> dict[str, object]:
        return {
            **self.deterministic_payload(public=public),
            "elapsed_seconds": self.elapsed_seconds,
            "actions_per_second": self.actions_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
        }

    def to_json(self, *, public: bool = False) -> str:
        return _canonical_json(self.payload(public=public))


def _rollout_descriptor(
    determinization_count: int,
    route_transition_ceiling: int,
) -> RolloutSearchDescriptor:
    return RolloutSearchDescriptor(
        determinization_count=determinization_count,
        route_transition_ceiling=route_transition_ceiling,
    )


def _rollout_agent_descriptor(descriptor: RolloutSearchDescriptor) -> AgentDescriptor:
    family = SAMPLED_TURN_ROLLOUT_AGENT_DESCRIPTOR
    return AgentDescriptor(
        family.name,
        family.version,
        (("search_descriptor_id", descriptor.descriptor_id),),
    )


def _require_fresh_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"throughput measurement directory must be fresh: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_rollout_generation_smoke(
    directory: str | Path,
    config: RolloutGenerationSmokeConfig | None = None,
) -> RolloutThroughputSmokeResult:
    """Measure fresh rollout-vs-rollout compact-replay generation through ``run_generation``."""

    from innovation_ai.training.compact_replay import (
        CompactReplayShardManifest,
        read_compact_replay_shard,
    )
    from innovation_ai.training.self_play import (
        GenerationConfig,
        SeatPolicy,
        plan_generation,
        run_generation,
    )

    config = config or RolloutGenerationSmokeConfig()
    root = Path(directory)
    _require_fresh_directory(root)
    descriptor = _rollout_descriptor(
        config.determinization_count,
        config.route_transition_ceiling,
    )
    policy_descriptor = _rollout_agent_descriptor(descriptor)
    policy = SeatPolicy(policy_descriptor.descriptor_id, "baseline", descriptor=policy_descriptor)
    manifest = plan_generation(
        GenerationConfig(
            "r4-rollout-generation-smoke",
            config.run_seed,
            config.generation,
            max_games_in_flight=config.games_in_flight,
            shard_episode_limit=config.shard_episode_limit,
            action_ceiling=config.action_ceiling,
            validation_level=config.validation_level,
        ),
        (policy,),
        ((policy.policy_id, policy.policy_id),),
        config.games,
    )
    failures: list[RolloutValidationFailure] = []
    committed_actions = 0
    completed_games = 0
    started = perf_counter()
    try:
        sealed = run_generation(
            root,
            manifest,
            search_descriptors={descriptor.descriptor_id: descriptor},
        )
        elapsed = perf_counter() - started
        if len(sealed) != len(manifest.shards):
            raise RuntimeError("generation smoke did not seal every planned replay shard")
        for shard in manifest.shards:
            episodes = read_compact_replay_shard(
                root / "replays" / f"{shard.shard_id}.jsonl.gz",
                CompactReplayShardManifest(shard.shard_id, shard.episode_ids),
                verify=True,
            )
            completed_games += len(episodes)
            committed_actions += sum(episode.transition_count for episode in episodes)
    except Exception as error:
        elapsed = perf_counter() - started
        diagnostic = getattr(error, "diagnostic", {})
        if isinstance(diagnostic, dict):
            action_count = diagnostic.get("action_count", 0)
            if isinstance(action_count, int) and not isinstance(action_count, bool):
                committed_actions = max(0, action_count)
        failures.append(
            RolloutValidationFailure(
                "generation-smoke",
                "self-play-runner",
                "generation",
                type(error).__name__,
                str(error),
            )
        )
    return RolloutThroughputSmokeResult(
        "generation",
        config,
        completed_games,
        committed_actions,
        elapsed,
        tuple(failures),
        peak_rss_bytes=_peak_rss_bytes(),
    )


def run_rollout_arena_smoke(
    directory: str | Path,
    config: RolloutArenaSmokeConfig | None = None,
) -> RolloutThroughputSmokeResult:
    """Measure a non-strength rollout-vs-random paired smoke through ``ArenaRunner``."""

    from innovation_ai.harness.arena import (
        ArenaManifest,
        BootstrapConfig,
        PolicyPool,
        PoolEntry,
        dumps_arena_manifest,
        plan_match_pair,
    )
    from innovation_ai.harness.arena import PolicyDescriptor as ArenaPolicyDescriptor
    from innovation_ai.harness.arena_runner import ArenaRunner, write_execution_artifacts
    from innovation_ai.harness.engine import InnovationEngineAdapter

    config = config or RolloutArenaSmokeConfig()
    root = Path(directory)
    _require_fresh_directory(root)
    descriptor = _rollout_descriptor(
        config.determinization_count,
        config.route_transition_ceiling,
    )
    rollout_policy = _rollout_agent_descriptor(descriptor)
    random_policy_id = RANDOM_AGENT_DESCRIPTOR.descriptor_id
    manifest = ArenaManifest(
        "r4-rollout-arena-throughput-smoke",
        rollout_policy.descriptor_id,
        PolicyPool("r4-rollout-arena-smoke-random", (PoolEntry(random_policy_id, 1),)),
        tuple(
            plan_match_pair(
                f"smoke-{index:04d}",
                config.seed_start + index,
                rollout_policy.descriptor_id,
                random_policy_id,
            )
            for index in range(config.seed_pairs)
        ),
        BootstrapConfig(seed=config.bootstrap_seed, resamples=config.bootstrap_resamples),
    )
    (root / "arena-manifest.json").write_text(
        dumps_arena_manifest(manifest) + "\n",
        encoding="ascii",
    )
    policies = {
        rollout_policy.descriptor_id: ArenaPolicyDescriptor(
            rollout_policy.descriptor_id, "search-heuristic"
        ),
        random_policy_id: ArenaPolicyDescriptor(random_policy_id, "random"),
    }
    runner = ArenaRunner(
        InnovationEngineAdapter(),
        policies,
        search_policy_descriptors={rollout_policy.descriptor_id: descriptor},
        search_descriptors={descriptor.descriptor_id: descriptor},
        max_actions=config.action_ceiling,
        run_seed=config.run_seed,
    )
    failures: list[RolloutValidationFailure] = []
    committed_actions = 0
    completed_games = 0
    rollout_decisions: int | None = None
    maximum_route_transitions = 0
    started = perf_counter()
    try:
        with validation(config.validation_level):
            execution = runner.execute(manifest)
        elapsed = perf_counter() - started
        write_execution_artifacts(root, execution)
        completed_games = len(execution.result.games)
        committed_actions = sum(game.game_length for game in execution.result.games)
        rollout_decisions = execution.search_telemetry.rollout_decisions
        maximum_route_transitions = execution.search_telemetry.maximum_route_transitions
    except Exception as error:
        elapsed = perf_counter() - started
        diagnostic = getattr(error, "diagnostic", {})
        if isinstance(diagnostic, dict):
            action_count = diagnostic.get("action_count", 0)
            if isinstance(action_count, int) and not isinstance(action_count, bool):
                committed_actions = max(0, action_count)
        failures.append(
            RolloutValidationFailure(
                "arena-smoke",
                "arena-runner",
                "arena",
                type(error).__name__,
                str(error),
            )
        )
    return RolloutThroughputSmokeResult(
        "arena",
        config,
        completed_games,
        committed_actions,
        elapsed,
        tuple(failures),
        rollout_decisions,
        maximum_route_transitions,
        _peak_rss_bytes(),
    )


def write_rollout_validation_json(
    result: RolloutRouteLengthSoakResult | RolloutThroughputSmokeResult,
    path: str | Path,
    *,
    public: bool = False,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.to_json(public=public) + "\n", encoding="utf-8")
    return target


__all__ = [
    "DEFAULT_ROLLOUT_DETERMINIZATION_COUNT",
    "DEFAULT_ROUTE_SOAK_DECISION_INDICES",
    "DEFAULT_ROUTE_SOAK_GAME_SEEDS",
    "DEFAULT_ROUTE_SOAK_HIGH_CHOICE_ROOTS",
    "ROLLOUT_ROUTE_LENGTH_SOAK_FORMAT",
    "ROLLOUT_THROUGHPUT_SMOKE_FORMAT",
    "ROLLOUT_VALIDATION_SCHEMA_VERSION",
    "RolloutArenaSmokeConfig",
    "RolloutGenerationSmokeConfig",
    "RolloutRouteLengthRootMeasurement",
    "RolloutRouteLengthSoakConfig",
    "RolloutRouteLengthSoakResult",
    "RolloutThroughputSmokeResult",
    "RolloutValidationFailure",
    "run_rollout_arena_smoke",
    "run_rollout_generation_smoke",
    "run_rollout_route_length_soak",
    "write_rollout_validation_json",
]
