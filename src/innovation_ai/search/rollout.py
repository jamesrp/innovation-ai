"""Deterministic one-turn policy rollout over player-safe synthetic states.

The selector accepts only an :class:`InformationSetSpec` and states reconstructed from it.  Every
legal root action is evaluated on every common determinization.  Later decisions follow the frozen
simple heuristic through its player-safe :class:`Decision` boundary, never an authoritative state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from innovation_ai import __version__
from innovation_ai.agents.heuristic import SimpleHeuristicAgent
from innovation_ai.innovation.actions import (
    ACTION_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    ChooseStartingMeldAction,
    Decision,
    DecisionKind,
    SemanticAction,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.state import RULES_VERSION, GamePhase, GameState
from innovation_ai.innovation.types import PlayerId
from innovation_ai.search.contracts import (
    DEFAULT_CONTINUATION_AGENT_DESCRIPTOR_ID,
    DEFAULT_CONTINUATION_AGENT_DESCRIPTOR_JSON,
    DEFAULT_HIDDEN_ALLOCATION_ALGORITHM,
    DEFAULT_MOVE_ORDERING,
    DEFAULT_ROLLOUT_ACTION_KEY,
    DEFAULT_ROLLOUT_ALGORITHM,
    DEFAULT_ROLLOUT_CEILING_ACCOUNTING,
    DEFAULT_ROLLOUT_CEILING_FAILURE,
    DEFAULT_ROLLOUT_CEILING_STATUS,
    DEFAULT_ROLLOUT_CONTINUATION_BRANCHING,
    DEFAULT_ROLLOUT_HYPOTHETICAL_LIVE_MISMATCH,
    DEFAULT_ROLLOUT_IMPLEMENTATION,
    DEFAULT_ROLLOUT_INFORMATION_POLICY,
    DEFAULT_ROLLOUT_SAMPLE_AGGREGATION,
    DEFAULT_ROLLOUT_SEARCH_DESCRIPTOR,
    DEFAULT_ROLLOUT_STARTING_MELD,
    DEFAULT_ROLLOUT_TURN_BOUNDARY,
    DEFAULT_SAMPLER_SEED_DERIVATION,
    DEFAULT_SAMPLING_ALGORITHM,
    DEFAULT_TIE_BREAK,
    RolloutFailureTrace,
    RolloutLeafTelemetry,
    RolloutPathStep,
    RolloutRootSelectionTelemetry,
    RolloutRouteTelemetry,
    RolloutSearchDescriptor,
)
from innovation_ai.search.evaluator import (
    EVALUATOR_VERSION,
    HandEngineeredEvaluator,
    evaluate_nonterminal_components,
)
from innovation_ai.search.information_sets import (
    INFORMATION_SET_RNG_VERSION,
    INFORMATION_SET_SAMPLER_VERSION,
    SYNTHETIC_SETUP_RNG_VERSION,
    SYNTHETIC_SETUP_SEED,
    InformationSetSpec,
    verify_sampled_state,
)
from innovation_ai.search.minimax import (
    SearchInputError,
    SearchInvariantError,
    StateEvaluator,
    action_key,
)


class ContinuationPolicy(Protocol):
    """Narrow player-safe boundary used for every non-root rollout choice."""

    def choose_action(self, decision: Decision, /) -> SemanticAction: ...


@dataclass(frozen=True, slots=True)
class RolloutHooks:
    """Identity-bearing engine operations for focused noncanonical tests.

    A custom evaluator or continuation hook must use a descriptor carrying these exact identities.
    This prevents graph fixtures and other test mechanics from emitting telemetry under the
    canonical rollout descriptor.
    """

    decisions: Callable[[GameState], tuple[Decision, ...]]
    transition: Callable[[GameState, SemanticAction], GameState]
    implementation_version: str
    evaluator_version: str
    continuation_agent_descriptor_id: str
    validate_sample: Callable[[InformationSetSpec, GameState], None] | None = None
    choose_continuation: Callable[[Decision], SemanticAction] | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.implementation_version, "rollout hook implementation version"),
            (self.evaluator_version, "rollout hook evaluator version"),
            (
                self.continuation_agent_descriptor_id,
                "rollout hook continuation agent descriptor ID",
            ),
        ):
            if not value:
                raise ValueError(f"{name} cannot be empty")


@dataclass(frozen=True, slots=True)
class RolloutStatistics:
    """Deterministic transition totals across completed rollout routes."""

    routes: int
    root_engine_transitions: int
    setup_response_engine_transitions: int
    continuation_engine_transitions: int
    completed_turn_routes: int
    terminal_routes: int
    maximum_route_transitions: int

    @property
    def total_engine_transitions(self) -> int:
        return (
            self.root_engine_transitions
            + self.setup_response_engine_transitions
            + self.continuation_engine_transitions
        )


@dataclass(frozen=True, slots=True)
class RolloutSelection:
    """Selected root action with deterministic rollout telemetry and counters."""

    action: SemanticAction
    telemetry: RolloutRootSelectionTelemetry
    statistics: RolloutStatistics


SampledRolloutResult = RolloutSelection
DeterministicRolloutResult = RolloutSelection


class RolloutSafetyCeilingError(SearchInvariantError):
    """A route reached its strict all-in transition ceiling before its next transition."""

    def __init__(self, trace: RolloutFailureTrace) -> None:
        self.trace = trace
        super().__init__(
            "rollout route reached its strict transition ceiling before the next transition"
        )


@dataclass(slots=True)
class _RouteProgress:
    root_action_index: int
    determinization_index: int
    root_action_key: str
    root_engine_transitions: int = 0
    setup_response_engine_transitions: int = 0
    continuation_engine_transitions: int = 0
    setup_response_action_key: str | None = None
    rollout_path: tuple[RolloutPathStep, ...] = ()

    @property
    def total_engine_transitions(self) -> int:
        return (
            self.root_engine_transitions
            + self.setup_response_engine_transitions
            + self.continuation_engine_transitions
        )


@dataclass(frozen=True, slots=True)
class _PreselectedStartingResponse:
    action_key: str


class _SimpleHeuristicContinuationAdapter:
    """Dedicated immutable adapter for the frozen printed-card observation policy."""

    descriptor_id = DEFAULT_CONTINUATION_AGENT_DESCRIPTOR_ID
    descriptor_json = DEFAULT_CONTINUATION_AGENT_DESCRIPTOR_JSON

    def __init__(self, registry: CardRegistry) -> None:
        self._agent = SimpleHeuristicAgent(registry)

    def choose_action(self, decision: Decision, /) -> SemanticAction:
        return self._agent.choose_action(decision)


class DeterministicSampledRollout:
    """Versioned production sampled deterministic turn-rollout implementation."""

    def __init__(
        self,
        descriptor: RolloutSearchDescriptor = DEFAULT_ROLLOUT_SEARCH_DESCRIPTOR,
        *,
        registry: CardRegistry | None = None,
        programs: EffectProgramRegistry | None = None,
        evaluator: StateEvaluator | None = None,
        hooks: RolloutHooks | None = None,
    ) -> None:
        self.descriptor = descriptor
        self._registry = registry or load_card_registry()
        self._programs = programs or load_effect_programs()
        self._evaluator: StateEvaluator = evaluator or HandEngineeredEvaluator(self._registry)
        evaluator_version = getattr(self._evaluator, "version", None)
        if not isinstance(evaluator_version, str) or not evaluator_version:
            raise SearchInputError(
                "a custom rollout evaluator must declare a non-empty version identity"
            )
        self._evaluator_version = evaluator_version
        if (
            evaluator is not None
            and evaluator_version == EVALUATOR_VERSION
            and not isinstance(evaluator, HandEngineeredEvaluator)
        ):
            raise SearchInputError(
                "a custom rollout evaluator cannot claim the canonical evaluator identity"
            )
        self._continuation: ContinuationPolicy = _SimpleHeuristicContinuationAdapter(self._registry)
        if hooks is None:
            registry_ref = self._registry
            programs_ref = self._programs
            self._hooks = RolloutHooks(
                decisions=lambda state: current_decisions(state, registry_ref, programs_ref),
                transition=lambda state, action: (
                    apply_action(state, action, registry_ref, programs_ref).state
                ),
                implementation_version=DEFAULT_ROLLOUT_IMPLEMENTATION,
                evaluator_version=self._evaluator_version,
                continuation_agent_descriptor_id=self._continuation.descriptor_id,
                validate_sample=lambda spec, state: verify_sampled_state(
                    spec, state, registry_ref, programs_ref
                ),
            )
        else:
            if hooks.implementation_version == DEFAULT_ROLLOUT_IMPLEMENTATION:
                raise SearchInputError(
                    "custom rollout hooks require an explicit noncanonical implementation identity"
                )
            if (
                hooks.choose_continuation is None
                and hooks.continuation_agent_descriptor_id != self._continuation.descriptor_id
            ):
                raise SearchInputError(
                    "default continuation cannot use a custom continuation descriptor identity"
                )
            if (
                hooks.choose_continuation is not None
                and hooks.continuation_agent_descriptor_id == self._continuation.descriptor_id
            ):
                raise SearchInputError(
                    "custom continuation hooks cannot claim the canonical agent identity"
                )
            self._hooks = hooks
        self._validate_descriptor()

    def select(
        self,
        spec: InformationSetSpec,
        sampled_states: Sequence[GameState],
        *,
        sampler_seed_digest: str | None = None,
    ) -> RolloutSelection:
        """Evaluate every legal root action on every common determinization."""

        samples = tuple(sampled_states)
        self._validate_inputs(spec, samples, sampler_seed_digest)
        root_actions = spec.legal_actions
        both_starting_pending = spec.boundary.decision_kind is DecisionKind.STARTING_MELD and all(
            choice is None for choice in spec.runtime.starting_meld_choices
        )
        preselected = tuple(
            self._preselect_starting_response(spec, sample) if both_starting_pending else None
            for sample in samples
        )

        routes: list[RolloutRouteTelemetry] = []
        route_values: list[list[float]] = [[] for _ in root_actions]
        for action_index, root_action in enumerate(root_actions):
            for determinization_index, sample in enumerate(samples):
                route = self._rollout_route(
                    spec=spec,
                    sample=sample,
                    root_action=root_action,
                    root_action_index=action_index,
                    determinization_index=determinization_index,
                    preselected_response=preselected[determinization_index],
                )
                assert route.value is not None
                route_values[action_index].append(route.value)
                routes.append(route)

        means = tuple(math.fsum(values) / len(samples) for values in route_values)
        best = max(means)
        tied = tuple(index for index, value in enumerate(means) if value == best)
        selected_index = tied[0]
        telemetry = RolloutRootSelectionTelemetry(
            rollout_descriptor_id=self.descriptor.descriptor_id,
            continuation_agent_descriptor_id=self.descriptor.continuation_agent_descriptor_id,
            information_set_digest=spec.spec_digest,
            sampler_seed_digest=sampler_seed_digest,
            determinization_count=len(samples),
            action_keys=tuple(action_key(action) for action in root_actions),
            action_mean_values=means,
            selected_action_index=selected_index,
            tied_action_indices=tied,
            routes=tuple(routes),
        )
        statistics = RolloutStatistics(
            routes=len(routes),
            root_engine_transitions=sum(item.root_engine_transitions for item in routes),
            setup_response_engine_transitions=sum(
                item.setup_response_engine_transitions for item in routes
            ),
            continuation_engine_transitions=sum(
                item.continuation_engine_transitions for item in routes
            ),
            completed_turn_routes=sum(item.completed_turn_count == 1 for item in routes),
            terminal_routes=sum(
                item.leaf is not None and item.leaf.terminal_reason is not None for item in routes
            ),
            maximum_route_transitions=max(item.total_engine_transitions for item in routes),
        )
        return RolloutSelection(root_actions[selected_index], telemetry, statistics)

    def _validate_descriptor(self) -> None:
        exact_strings = (
            (self.descriptor.algorithm, DEFAULT_ROLLOUT_ALGORITHM, "algorithm"),
            (
                self.descriptor.information_policy_version,
                DEFAULT_ROLLOUT_INFORMATION_POLICY,
                "information policy",
            ),
            (
                self.descriptor.information_set_sampler_version,
                INFORMATION_SET_SAMPLER_VERSION,
                "information-set sampler",
            ),
            (
                self.descriptor.information_set_rng_version,
                INFORMATION_SET_RNG_VERSION,
                "information-set RNG",
            ),
            (
                self.descriptor.synthetic_setup_rng_version,
                SYNTHETIC_SETUP_RNG_VERSION,
                "synthetic setup RNG",
            ),
            (
                self.descriptor.sampling_algorithm,
                DEFAULT_SAMPLING_ALGORITHM,
                "sampling algorithm",
            ),
            (
                self.descriptor.hidden_allocation_algorithm,
                DEFAULT_HIDDEN_ALLOCATION_ALGORITHM,
                "hidden allocation",
            ),
            (
                self.descriptor.sampler_seed_derivation,
                DEFAULT_SAMPLER_SEED_DERIVATION,
                "sampler seed derivation",
            ),
            (
                self.descriptor.continuation_branching,
                DEFAULT_ROLLOUT_CONTINUATION_BRANCHING,
                "continuation branching",
            ),
            (
                self.descriptor.hypothetical_live_mismatch,
                DEFAULT_ROLLOUT_HYPOTHETICAL_LIVE_MISMATCH,
                "hypothetical/live mismatch",
            ),
            (
                self.descriptor.turn_boundary,
                DEFAULT_ROLLOUT_TURN_BOUNDARY,
                "turn boundary",
            ),
            (
                self.descriptor.sample_aggregation,
                DEFAULT_ROLLOUT_SAMPLE_AGGREGATION,
                "sample aggregation",
            ),
            (
                self.descriptor.starting_meld_semantics,
                DEFAULT_ROLLOUT_STARTING_MELD,
                "starting meld semantics",
            ),
            (self.descriptor.root_ordering, DEFAULT_MOVE_ORDERING, "root ordering"),
            (self.descriptor.action_key_version, DEFAULT_ROLLOUT_ACTION_KEY, "action key"),
            (self.descriptor.tie_break, DEFAULT_TIE_BREAK, "tie break"),
            (
                self.descriptor.ceiling_accounting,
                DEFAULT_ROLLOUT_CEILING_ACCOUNTING,
                "ceiling accounting",
            ),
            (
                self.descriptor.ceiling_failure,
                DEFAULT_ROLLOUT_CEILING_FAILURE,
                "ceiling failure",
            ),
            (
                self.descriptor.ceiling_status,
                DEFAULT_ROLLOUT_CEILING_STATUS,
                "experimental ceiling status",
            ),
            (self.descriptor.engine_version, __version__, "engine version"),
            (self.descriptor.rules_version, RULES_VERSION, "rules version"),
            (
                self.descriptor.card_data_fingerprint,
                self._registry.data_fingerprint,
                "card-data fingerprint",
            ),
            (
                self.descriptor.effects_fingerprint,
                self._programs.fingerprint(),
                "effects fingerprint",
            ),
        )
        for actual, expected, name in exact_strings:
            if actual != expected:
                raise SearchInputError(f"rollout descriptor {name} is incompatible")
        runtime_identities = (
            (
                self.descriptor.implementation_version,
                self._hooks.implementation_version,
                "implementation version",
            ),
            (
                self.descriptor.evaluator_version,
                self._evaluator_version,
                "evaluator version",
            ),
            (
                self.descriptor.continuation_agent_descriptor_id,
                self._hooks.continuation_agent_descriptor_id,
                "continuation agent descriptor ID",
            ),
        )
        for declared, actual, name in runtime_identities:
            if declared != actual:
                raise SearchInputError(f"runtime {name} differs from rollout descriptor")
        if self._hooks.evaluator_version != self._evaluator_version:
            raise SearchInputError("rollout hook evaluator identity differs from runtime evaluator")
        if self.descriptor.action_schema_version != ACTION_SCHEMA_VERSION:
            raise SearchInputError("rollout descriptor action schema is incompatible")
        if self.descriptor.decision_schema_version != DECISION_SCHEMA_VERSION:
            raise SearchInputError("rollout descriptor decision schema is incompatible")

    def _validate_inputs(
        self,
        spec: InformationSetSpec,
        samples: tuple[GameState, ...],
        sampler_seed_digest: str | None,
    ) -> None:
        if spec.spec_version != self.descriptor.information_set_spec_version:
            raise SearchInputError("information-set and rollout descriptor versions differ")
        if spec.information_policy_version != self.descriptor.information_policy_version:
            raise SearchInputError("information-set and rollout information policies differ")
        if spec.sampler_version != self.descriptor.information_set_sampler_version:
            raise SearchInputError("information-set and rollout sampler versions differ")
        if spec.rng_version != self.descriptor.information_set_rng_version:
            raise SearchInputError("information-set and rollout RNG versions differ")
        if len(samples) != self.descriptor.determinization_count:
            raise SearchInputError(
                "sample count must equal the rollout descriptor determinization count"
            )
        if not samples:
            raise SearchInputError("sampled rollout requires at least one determinization")
        if sampler_seed_digest is not None and not sampler_seed_digest.startswith("sha256:"):
            raise SearchInputError("sampler seed digest must be a tagged SHA-256 digest")
        for sample in samples:
            if sample.setup.seed != SYNTHETIC_SETUP_SEED:
                raise SearchInputError("rollout accepts only synthetic information-set states")
            if self._hooks.validate_sample is not None:
                try:
                    self._hooks.validate_sample(spec, sample)
                except Exception as error:
                    raise SearchInputError(f"invalid synthetic rollout sample: {error}") from error
            self._target_decision(spec, sample)

    def _target_decision(self, spec: InformationSetSpec, state: GameState) -> Decision:
        decision = next(
            (
                candidate
                for candidate in self._hooks.decisions(state)
                if candidate.decision_id == spec.target_decision_id
            ),
            None,
        )
        if decision is None:
            raise SearchInputError("sample does not expose the target root decision")
        if decision.chooser is not spec.chooser or decision.legal_actions != spec.legal_actions:
            raise SearchInputError("sample target decision differs from the information set")
        if decision.observation.viewer is not decision.chooser:
            raise SearchInputError("sample root Decision is not owned by its chooser")
        return decision

    def _preselect_starting_response(
        self, spec: InformationSetSpec, sample: GameState
    ) -> _PreselectedStartingResponse:
        decisions = self._hooks.decisions(sample)
        opponents = tuple(
            decision
            for decision in decisions
            if decision.kind is DecisionKind.STARTING_MELD and decision.chooser is not spec.chooser
        )
        if len(opponents) != 1:
            raise SearchInvariantError(
                "two-pending setup root must expose exactly one opponent starting decision"
            )
        action = self._choose_continuation(opponents[0])
        if not isinstance(action, ChooseStartingMeldAction):
            raise SearchInvariantError("starting response policy returned a non-starting action")
        return _PreselectedStartingResponse(action_key(action))

    def _rollout_route(
        self,
        *,
        spec: InformationSetSpec,
        sample: GameState,
        root_action: SemanticAction,
        root_action_index: int,
        determinization_index: int,
        preselected_response: _PreselectedStartingResponse | None,
    ) -> RolloutRouteTelemetry:
        root_decision = self._target_decision(spec, sample)
        if root_action not in root_decision.legal_actions:
            raise SearchInputError("a sample does not expose an original rollout root action")
        progress = _RouteProgress(
            root_action_index,
            determinization_index,
            action_key(root_action),
        )
        setup_root = spec.continuation.phase is GamePhase.STARTING_MELDS
        target_turn = None if setup_root else sample.turn_number
        state = self._apply_transition(
            spec,
            sample,
            root_action,
            progress,
            role="root",
            target_turn=target_turn,
        )

        if setup_root:
            if preselected_response is not None:
                if state.terminal_result is not None:
                    raise SearchInvariantError(
                        "first simultaneous starting choice unexpectedly terminated the game"
                    )
                response_decision = self._single_continuation_decision(state)
                if (
                    response_decision.kind is not DecisionKind.STARTING_MELD
                    or response_decision.chooser is spec.chooser
                ):
                    raise SearchInvariantError(
                        "root starting choice exposed the wrong setup response"
                    )
                response = self._rebind_action(preselected_response.action_key, response_decision)
                state = self._apply_transition(
                    spec,
                    state,
                    response,
                    progress,
                    role="setup-response",
                    target_turn=None,
                )
                progress.setup_response_action_key = preselected_response.action_key
            if state.terminal_result is None and state.phase is not GamePhase.PLAY:
                raise SearchInvariantError("completed starting choices did not enter play")
            target_turn = state.turn_number

        assert target_turn is not None
        while state.terminal_result is None and state.turn_number == target_turn:
            decision = self._single_continuation_decision(state)
            action = self._choose_continuation(decision)
            step = RolloutPathStep(
                action_key(action),
                decision.chooser.value,
                decision.kind.value,
            )
            state = self._apply_transition(
                spec,
                state,
                action,
                progress,
                role="continuation",
                target_turn=target_turn,
                path_step=step,
            )

        if state.terminal_result is None and state.turn_number != target_turn + 1:
            raise SearchInvariantError("rollout did not stop at the next anchored turn boundary")
        completed_turn_count = int(state.turn_number == target_turn + 1)
        leaf = self._evaluate_leaf(state, spec.chooser)
        return RolloutRouteTelemetry(
            root_action_index=root_action_index,
            determinization_index=determinization_index,
            root_action_key=progress.root_action_key,
            leaf=leaf,
            root_engine_transitions=progress.root_engine_transitions,
            setup_response_engine_transitions=progress.setup_response_engine_transitions,
            continuation_engine_transitions=progress.continuation_engine_transitions,
            completed_turn_count=completed_turn_count,
            rollout_path=progress.rollout_path,
            setup_response_action_key=progress.setup_response_action_key,
        )

    def _single_continuation_decision(self, state: GameState) -> Decision:
        decisions = self._hooks.decisions(state)
        if len(decisions) != 1:
            raise SearchInvariantError(
                "a nonterminal rollout state before the boundary must expose exactly one decision"
            )
        decision = decisions[0]
        if decision.observation.viewer is not decision.chooser:
            raise SearchInvariantError("continuation Decision is not owned by its chooser")
        return decision

    def _choose_continuation(self, decision: Decision) -> SemanticAction:
        if self._hooks.choose_continuation is None:
            action = self._continuation.choose_action(decision)
        else:
            action = self._hooks.choose_continuation(decision)
        if action not in decision.legal_actions:
            raise SearchInvariantError("continuation policy returned an illegal action")
        return action

    @staticmethod
    def _rebind_action(key: str, decision: Decision) -> SemanticAction:
        matches = tuple(action for action in decision.legal_actions if action_key(action) == key)
        if len(matches) != 1:
            raise SearchInvariantError(
                "preselected starting response cannot be rebound to the pending decision"
            )
        return matches[0]

    def _apply_transition(
        self,
        spec: InformationSetSpec,
        before: GameState,
        action: SemanticAction,
        progress: _RouteProgress,
        *,
        role: str,
        target_turn: int | None,
        path_step: RolloutPathStep | None = None,
    ) -> GameState:
        if progress.total_engine_transitions >= self.descriptor.route_transition_ceiling:
            failed_route = RolloutRouteTelemetry(
                root_action_index=progress.root_action_index,
                determinization_index=progress.determinization_index,
                root_action_key=progress.root_action_key,
                leaf=None,
                root_engine_transitions=progress.root_engine_transitions,
                setup_response_engine_transitions=progress.setup_response_engine_transitions,
                continuation_engine_transitions=progress.continuation_engine_transitions,
                completed_turn_count=0,
                rollout_path=progress.rollout_path,
                setup_response_action_key=progress.setup_response_action_key,
                safety_ceiling_hit=True,
            )
            trace = RolloutFailureTrace(
                rollout_descriptor_id=self.descriptor.descriptor_id,
                continuation_agent_descriptor_id=(self.descriptor.continuation_agent_descriptor_id),
                information_set_digest=spec.spec_digest,
                route=failed_route,
                failure="route-transition-ceiling",
            )
            raise RolloutSafetyCeilingError(trace)

        after = self._hooks.transition(before, action)
        if role == "root":
            progress.root_engine_transitions += 1
        elif role == "setup-response":
            progress.setup_response_engine_transitions += 1
        elif role == "continuation":
            progress.continuation_engine_transitions += 1
            if path_step is None:
                raise SearchInvariantError("continuation transition omitted its path step")
            progress.rollout_path = (*progress.rollout_path, path_step)
        else:  # pragma: no cover - internal exhaustive guard
            raise AssertionError(f"unknown rollout transition role {role!r}")

        turn_delta = after.turn_number - before.turn_number
        if turn_delta < 0 or turn_delta > 1:
            raise SearchInvariantError("a rollout transition skipped or reversed turn numbers")
        if target_turn is not None and (
            after.turn_number < target_turn or after.turn_number > target_turn + 1
        ):
            raise SearchInvariantError("a rollout transition crossed the anchored turn boundary")
        return after

    def _evaluate_leaf(self, state: GameState, root_player: PlayerId) -> RolloutLeafTelemetry:
        value = float(self._evaluator(state, root_player))
        if not math.isfinite(value):
            raise SearchInvariantError("rollout evaluator returned a non-finite value")
        terminal_reason = (
            None if state.terminal_result is None else state.terminal_result.reason.value
        )
        if state.terminal_result is None and isinstance(self._evaluator, HandEngineeredEvaluator):
            components = evaluate_nonterminal_components(
                state, root_player, self._evaluator.registry
            )
            return RolloutLeafTelemetry(
                value=value,
                achievement=components.achievement,
                score=components.score,
                icons=components.icons,
                board=components.board,
                hand=components.hand,
            )
        return RolloutLeafTelemetry(value=value, terminal_reason=terminal_reason)


# Concise aliases parallel the existing minimax API without conflating their result types.
SampledRollout = DeterministicSampledRollout


def select_sampled_rollout(
    spec: InformationSetSpec,
    sampled_states: Sequence[GameState],
    descriptor: RolloutSearchDescriptor = DEFAULT_ROLLOUT_SEARCH_DESCRIPTOR,
    *,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
    evaluator: StateEvaluator | None = None,
    hooks: RolloutHooks | None = None,
    sampler_seed_digest: str | None = None,
) -> RolloutSelection:
    """Functional entry point for deterministic sampled rollout selection."""

    return DeterministicSampledRollout(
        descriptor,
        registry=registry,
        programs=programs,
        evaluator=evaluator,
        hooks=hooks,
    ).select(spec, sampled_states, sampler_seed_digest=sampler_seed_digest)


select_rollout_action = select_sampled_rollout
search_rollout = select_sampled_rollout


__all__ = [
    "ContinuationPolicy",
    "DeterministicRolloutResult",
    "DeterministicSampledRollout",
    "RolloutHooks",
    "RolloutSafetyCeilingError",
    "RolloutSelection",
    "RolloutStatistics",
    "SampledRollout",
    "SampledRolloutResult",
    "search_rollout",
    "select_rollout_action",
    "select_sampled_rollout",
]
