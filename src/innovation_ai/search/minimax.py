"""Deterministic root-sampled minimax over player-safe synthetic states.

The public entry points in this module accept an :class:`InformationSetSpec` and an explicit
collection of synthetic states only.  The authoritative live state is intentionally not part of
this API.  Every default transition goes through the Innovation protocol's ``current_decisions``
and ``apply_action`` functions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from innovation_ai.innovation.actions import (
    Decision,
    DecisionKind,
    SemanticAction,
    action_payload,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.state import GamePhase, GameState
from innovation_ai.innovation.strategic import strategic_state_digest
from innovation_ai.innovation.types import PlayerId
from innovation_ai.search.contracts import (
    DEFAULT_SEARCH_DESCRIPTOR,
    RootSelectionTelemetry,
    SearchDescriptor,
    SearchRouteTelemetry,
)
from innovation_ai.search.evaluator import HandEngineeredEvaluator
from innovation_ai.search.information_sets import (
    SYNTHETIC_SETUP_SEED,
    InformationSetSpec,
    verify_sampled_state,
)


class SearchInputError(ValueError):
    """The player-safe root or its supplied synthetic samples are inconsistent."""


class SearchInvariantError(RuntimeError):
    """A search hook exposed a state that violates the decision-boundary contract."""


class StateEvaluator(Protocol):
    """Leaf evaluator fixed to the root player's point of view by the caller."""

    def __call__(self, state: GameState, root_player: PlayerId, /) -> float: ...


@dataclass(frozen=True, slots=True)
class SearchHooks:
    """Injectable pure engine operations used by focused search tests and diagnostics.

    Passing custom hooks does not relax the synthetic-provenance check.  It only replaces tree
    mechanics after the caller has supplied states reconstructed from an information set.
    """

    decisions: Callable[[GameState], tuple[Decision, ...]]
    transition: Callable[[GameState, SemanticAction], GameState]
    digest: Callable[[GameState], str] = strategic_state_digest
    validate_sample: Callable[[InformationSetSpec, GameState], None] | None = None


@dataclass(frozen=True, slots=True)
class SearchStatistics:
    """Totals across all independently budgeted action/determinization routes."""

    routes: int
    nodes: int
    recursive_engine_transitions: int
    root_transitions: int
    mandatory_setup_transitions: int
    transposition_hits: int
    repeated_position_cutoffs: int
    budget_cutoff_routes: int
    immediate_leaf_fallback_routes: int

    @property
    def total_engine_transitions(self) -> int:
        """Include transitions deliberately excluded from the per-route search budget."""

        return (
            self.recursive_engine_transitions
            + self.root_transitions
            + self.mandatory_setup_transitions
        )


@dataclass(frozen=True, slots=True)
class MinimaxSelection:
    """Selected root action together with auditable route and aggregate telemetry."""

    action: SemanticAction
    telemetry: RootSelectionTelemetry
    statistics: SearchStatistics


# Descriptive aliases make the result convenient at policy integration boundaries.
SampledMinimaxResult = MinimaxSelection
SearchSelection = MinimaxSelection


@dataclass(slots=True)
class _Budget:
    limit: int
    transitions: int = 0

    def spend(self) -> bool:
        if self.transitions >= self.limit:
            return False
        self.transitions += 1
        return True


@dataclass(slots=True)
class _Counters:
    nodes: int = 0
    transposition_hits: int = 0
    repeated_position_cutoffs: int = 0


@dataclass(frozen=True, slots=True)
class _NodeResult:
    value: float
    principal_variation: tuple[str, ...]
    complete: bool
    cacheable: bool


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    value: float
    principal_variation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedStartingResponse:
    action_key: str
    state: GameState


def action_key(action: SemanticAction) -> str:
    """Return a canonical semantic key that deliberately ignores ``decision_id``.

    Semantic payload fields (card, player, value, order, and so on) remain in the key.  Canonical
    JSON avoids unstable ``repr`` output and stays collision-free across the action variants.
    """

    payload = action_payload(action)
    payload.pop("decision_id", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _completed_turns(before: GameState, after: GameState) -> int:
    # Committing the second simultaneous starting choice creates turn 1, but setup finalization is
    # not itself a completed player turn.
    if before.phase is GamePhase.STARTING_MELDS:
        return 0
    if before.phase is not GamePhase.PLAY:
        return 0
    return max(0, after.turn_number - before.turn_number)


def _finite_evaluation(evaluator: StateEvaluator, state: GameState, root_player: PlayerId) -> float:
    value = float(evaluator(state, root_player))
    if not math.isfinite(value):
        raise SearchInvariantError("search evaluator returned a non-finite value")
    return value


class DeterministicSampledMinimax:
    """Frozen deterministic sampled minimax implementation.

    A route is one ``(original root action, determinization)`` pair.  Its transition budget and
    exact-only transposition table are independent from every other route, while iterative
    deepening iterations consume that route's budget cumulatively.
    """

    def __init__(
        self,
        descriptor: SearchDescriptor = DEFAULT_SEARCH_DESCRIPTOR,
        *,
        registry: CardRegistry | None = None,
        programs: EffectProgramRegistry | None = None,
        evaluator: StateEvaluator | None = None,
        hooks: SearchHooks | None = None,
    ) -> None:
        self.descriptor = descriptor
        self._registry = registry or load_card_registry()
        self._programs = programs or load_effect_programs()
        self._evaluator: StateEvaluator = evaluator or HandEngineeredEvaluator(self._registry)
        if hooks is None:
            registry_ref = self._registry
            programs_ref = self._programs
            self._hooks = SearchHooks(
                decisions=lambda state: current_decisions(state, registry_ref, programs_ref),
                transition=lambda state, action: (
                    apply_action(state, action, registry_ref, programs_ref).state
                ),
                validate_sample=lambda spec, state: verify_sampled_state(
                    spec, state, registry_ref, programs_ref
                ),
            )
        else:
            self._hooks = hooks

    def select(
        self, spec: InformationSetSpec, sampled_states: Sequence[GameState]
    ) -> MinimaxSelection:
        """Search the common determinizations and select in stable root legal order."""

        samples = tuple(sampled_states)
        self._validate_inputs(spec, samples)
        root_actions = spec.legal_actions
        route_telemetry: list[SearchRouteTelemetry] = []
        route_values: list[list[float]] = [[] for _ in root_actions]
        root_transitions = 0
        mandatory_setup_transitions = 0

        both_starting_pending = spec.boundary.decision_kind is DecisionKind.STARTING_MELD and all(
            choice is None for choice in spec.runtime.starting_meld_choices
        )
        target_depth = self._target_depth(spec)

        for action_index, root_action in enumerate(root_actions):
            for determinization_index, sample in enumerate(samples):
                current_root = self._target_decision(spec, sample)
                if root_action not in current_root.legal_actions:
                    raise SearchInputError("a sample does not expose an original root action")
                after_root = self._hooks.transition(sample, root_action)
                root_transitions += 1
                initial_completed = _completed_turns(sample, after_root)
                prepared: tuple[_PreparedStartingResponse, ...] = ()
                if both_starting_pending:
                    prepared = self._prepare_starting_responses(spec, after_root)
                    mandatory_setup_transitions += len(prepared)

                route = self._search_route(
                    root_player=spec.chooser,
                    root_action_key=action_key(root_action),
                    root_action_index=action_index,
                    determinization_index=determinization_index,
                    after_root=after_root,
                    prepared_starting_responses=prepared,
                    initial_completed_turns=initial_completed,
                    target_depth=target_depth,
                )
                route_values[action_index].append(route.value)
                route_telemetry.append(route)

        means = tuple(math.fsum(values) / len(values) for values in route_values)
        best = max(means)
        tied = tuple(index for index, value in enumerate(means) if value == best)
        selected_index = tied[0]
        telemetry = RootSelectionTelemetry(
            search_descriptor_id=self.descriptor.descriptor_id,
            action_keys=tuple(action_key(action) for action in root_actions),
            action_mean_values=means,
            selected_action_index=selected_index,
            routes=tuple(route_telemetry),
            tied_action_indices=tied,
        )
        statistics = SearchStatistics(
            routes=len(route_telemetry),
            nodes=sum(route.nodes for route in route_telemetry),
            recursive_engine_transitions=sum(route.engine_transitions for route in route_telemetry),
            root_transitions=root_transitions,
            mandatory_setup_transitions=mandatory_setup_transitions,
            transposition_hits=sum(route.transposition_hits for route in route_telemetry),
            repeated_position_cutoffs=sum(
                route.repeated_position_cutoffs for route in route_telemetry
            ),
            budget_cutoff_routes=sum(route.budget_cutoff for route in route_telemetry),
            immediate_leaf_fallback_routes=sum(
                route.immediate_leaf_fallback for route in route_telemetry
            ),
        )
        return MinimaxSelection(root_actions[selected_index], telemetry, statistics)

    def _validate_inputs(self, spec: InformationSetSpec, samples: tuple[GameState, ...]) -> None:
        if spec.spec_version != self.descriptor.information_set_spec_version:
            raise SearchInputError("information-set and search descriptor versions differ")
        if self.descriptor.evaluator_version != getattr(
            self._evaluator, "version", self.descriptor.evaluator_version
        ):
            raise SearchInputError("evaluator and search descriptor versions differ")
        if len(samples) != self.descriptor.determinization_count:
            raise SearchInputError(
                "sample count must equal the search descriptor determinization count"
            )
        if not samples:
            raise SearchInputError("sampled minimax requires at least one determinization")
        for sample in samples:
            if sample.setup.seed != SYNTHETIC_SETUP_SEED:
                raise SearchInputError("search accepts only synthetic information-set states")
            if self._hooks.validate_sample is not None:
                try:
                    self._hooks.validate_sample(spec, sample)
                except Exception as error:
                    raise SearchInputError(f"invalid synthetic search sample: {error}") from error
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
        return decision

    def _target_depth(self, spec: InformationSetSpec) -> int:
        if spec.continuation.phase is GamePhase.STARTING_MELDS:
            return self.descriptor.starting_meld_horizon
        if spec.continuation.active_player is spec.chooser:
            return self.descriptor.root_turn_horizon
        return self.descriptor.opponent_turn_horizon

    def _prepare_starting_responses(
        self, spec: InformationSetSpec, after_root: GameState
    ) -> tuple[_PreparedStartingResponse, ...]:
        decisions = self._hooks.decisions(after_root)
        opponent = next(
            (
                decision
                for decision in decisions
                if decision.kind is DecisionKind.STARTING_MELD
                and decision.chooser is not spec.chooser
            ),
            None,
        )
        if opponent is None:
            raise SearchInvariantError("root starting choice exposed no opponent response")
        return tuple(
            _PreparedStartingResponse(
                action_key(response), self._hooks.transition(after_root, response)
            )
            for response in opponent.legal_actions
        )

    def _search_route(
        self,
        *,
        root_player: PlayerId,
        root_action_key: str,
        root_action_index: int,
        determinization_index: int,
        after_root: GameState,
        prepared_starting_responses: tuple[_PreparedStartingResponse, ...],
        initial_completed_turns: int,
        target_depth: int,
    ) -> SearchRouteTelemetry:
        budget = _Budget(self.descriptor.route_transition_budget)
        counters = _Counters()
        cache: dict[tuple[str, int], _CacheEntry] = {}

        if prepared_starting_responses:
            fallback_value, fallback_suffix = self._starting_immediate_leaf(
                prepared_starting_responses, root_player, counters
            )
        else:
            counters.nodes += 1
            fallback_value = _finite_evaluation(self._evaluator, after_root, root_player)
            fallback_suffix = ()

        retained_value = fallback_value
        retained_pv = (root_action_key, *fallback_suffix)
        completed_depth = 0
        budget_cutoff = False

        # A terminal afterstate has no deeper choices; every requested completed-turn depth is
        # vacuously complete and should not be mislabeled as an immediate-leaf budget fallback.
        terminal_states = (
            tuple(item.state for item in prepared_starting_responses)
            if prepared_starting_responses
            else (after_root,)
        )
        if all(state.terminal_result is not None for state in terminal_states):
            completed_depth = target_depth
        else:
            for depth in range(1, target_depth + 1):
                remaining = max(0, depth - initial_completed_turns)
                if prepared_starting_responses:
                    result = self._search_starting_minimum(
                        prepared_starting_responses,
                        root_player,
                        remaining,
                        budget,
                        counters,
                        cache,
                    )
                else:
                    result = self._minimax(
                        after_root,
                        root_player,
                        remaining,
                        -math.inf,
                        math.inf,
                        budget,
                        counters,
                        cache,
                        frozenset(),
                    )
                if not result.complete:
                    budget_cutoff = True
                    break
                completed_depth = depth
                retained_value = result.value
                retained_pv = (root_action_key, *result.principal_variation)

        return SearchRouteTelemetry(
            root_action_index=root_action_index,
            determinization_index=determinization_index,
            value=retained_value,
            completed_turn_depth=completed_depth,
            nodes=counters.nodes,
            engine_transitions=budget.transitions,
            transposition_hits=counters.transposition_hits,
            repeated_position_cutoffs=counters.repeated_position_cutoffs,
            budget_cutoff=budget_cutoff,
            immediate_leaf_fallback=completed_depth == 0,
            principal_variation=retained_pv,
        )

    def _starting_immediate_leaf(
        self,
        responses: tuple[_PreparedStartingResponse, ...],
        root_player: PlayerId,
        counters: _Counters,
    ) -> tuple[float, tuple[str, ...]]:
        best_value = math.inf
        best_key = responses[0].action_key
        for response in responses:
            counters.nodes += 1
            value = _finite_evaluation(self._evaluator, response.state, root_player)
            if value < best_value:
                best_value = value
                best_key = response.action_key
        return best_value, (best_key,)

    def _search_starting_minimum(
        self,
        responses: tuple[_PreparedStartingResponse, ...],
        root_player: PlayerId,
        remaining_turns: int,
        budget: _Budget,
        counters: _Counters,
        cache: dict[tuple[str, int], _CacheEntry],
    ) -> _NodeResult:
        best_value = math.inf
        best_pv: tuple[str, ...] = ()
        cacheable = True
        for response in responses:
            result = self._minimax(
                response.state,
                root_player,
                remaining_turns,
                -math.inf,
                best_value,
                budget,
                counters,
                cache,
                frozenset(),
            )
            if not result.complete:
                return _NodeResult(best_value, best_pv, False, False)
            cacheable = cacheable and result.cacheable
            if result.value < best_value:
                best_value = result.value
                best_pv = (response.action_key, *result.principal_variation)
        return _NodeResult(best_value, best_pv, True, cacheable)

    def _minimax(
        self,
        state: GameState,
        root_player: PlayerId,
        remaining_turns: int,
        alpha: float,
        beta: float,
        budget: _Budget,
        counters: _Counters,
        cache: dict[tuple[str, int], _CacheEntry],
        path: frozenset[str],
    ) -> _NodeResult:
        counters.nodes += 1
        if state.terminal_result is not None:
            return _NodeResult(
                _finite_evaluation(self._evaluator, state, root_player), (), True, True
            )

        digest = self._hooks.digest(state)
        if digest in path:
            counters.repeated_position_cutoffs += 1
            return _NodeResult(
                _finite_evaluation(self._evaluator, state, root_player), (), True, False
            )
        if remaining_turns <= 0:
            return _NodeResult(
                _finite_evaluation(self._evaluator, state, root_player), (), True, True
            )

        cache_key = (digest, remaining_turns)
        cached = cache.get(cache_key)
        if cached is not None:
            counters.transposition_hits += 1
            return _NodeResult(cached.value, cached.principal_variation, True, True)

        decisions = self._hooks.decisions(state)
        if len(decisions) != 1:
            raise SearchInvariantError(
                "a recursive nonterminal search state must expose exactly one decision"
            )
        decision = decisions[0]
        maximize = decision.chooser is root_player
        best_value = -math.inf if maximize else math.inf
        best_pv: tuple[str, ...] = ()
        child_cacheable = True
        pruned = False
        next_path = path | {digest}

        for action in decision.legal_actions:
            if not budget.spend():
                return _NodeResult(best_value, best_pv, False, False)
            child = self._hooks.transition(state, action)
            child_result = self._minimax(
                child,
                root_player,
                max(0, remaining_turns - _completed_turns(state, child)),
                alpha,
                beta,
                budget,
                counters,
                cache,
                next_path,
            )
            if not child_result.complete:
                return _NodeResult(best_value, best_pv, False, False)
            child_cacheable = child_cacheable and child_result.cacheable
            candidate_pv = (action_key(action), *child_result.principal_variation)
            if maximize:
                if child_result.value > best_value:
                    best_value = child_result.value
                    best_pv = candidate_pv
                alpha = max(alpha, best_value)
            else:
                if child_result.value < best_value:
                    best_value = child_result.value
                    best_pv = candidate_pv
                beta = min(beta, best_value)
            if alpha >= beta:
                pruned = True
                break

        result = _NodeResult(best_value, best_pv, True, child_cacheable and not pruned)
        if result.cacheable:
            cache[cache_key] = _CacheEntry(result.value, result.principal_variation)
        return result


# Short class alias used by policy code.
SampledMinimax = DeterministicSampledMinimax


def select_sampled_minimax(
    spec: InformationSetSpec,
    sampled_states: Sequence[GameState],
    descriptor: SearchDescriptor = DEFAULT_SEARCH_DESCRIPTOR,
    *,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
    evaluator: StateEvaluator | None = None,
    hooks: SearchHooks | None = None,
) -> MinimaxSelection:
    """Functional entry point for deterministic sampled minimax selection."""

    return DeterministicSampledMinimax(
        descriptor,
        registry=registry,
        programs=programs,
        evaluator=evaluator,
        hooks=hooks,
    ).select(spec, sampled_states)


# Intent-revealing functional aliases for call sites that already carry the information-set spec.
select_minimax_action = select_sampled_minimax
search_minimax = select_sampled_minimax


__all__ = [
    "DeterministicSampledMinimax",
    "MinimaxSelection",
    "SampledMinimax",
    "SampledMinimaxResult",
    "SearchHooks",
    "SearchInputError",
    "SearchInvariantError",
    "SearchSelection",
    "SearchStatistics",
    "StateEvaluator",
    "action_key",
    "search_minimax",
    "select_minimax_action",
    "select_sampled_minimax",
]
