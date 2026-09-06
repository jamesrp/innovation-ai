"""Pull-runner scheduler for learned afterstates and player-safe sampled search.

The scheduler is deliberately the only orchestration layer that sees both a
live runner state (solely to build an audited information-set specification)
and policy evaluators. Candidate actions are never scored on the authoritative
state: learned expansion and sampled search operate only on reconstructed
synthetic states.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from innovation_ai.agents.base import Agent
from innovation_ai.agents.heuristic import SimpleHeuristicAgent
from innovation_ai.harness.afterstates import (
    CandidateExpansion,
    CandidateExpansionError,
    TrustedCandidateExpander,
)
from innovation_ai.harness.policy import BatchValueEvaluator, PolicySelection
from innovation_ai.harness.runner import PendingGameDecision, PullGameRunner, Submission
from innovation_ai.innovation.actions import DecisionKind, SemanticAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import GameState
from innovation_ai.innovation.types import PlayerId
from innovation_ai.search.contracts import (
    DEFAULT_ROLLOUT_ALGORITHM,
    DEFAULT_SAMPLER_SEED_DERIVATION,
    DEFAULT_SEARCH_VERSION,
    MINIMAX_SEARCH_DESCRIPTOR_KIND,
    PRODUCTION_ROLLOUT_SEARCH_DESCRIPTOR,
    PRODUCTION_SEARCH_DESCRIPTOR,
    ROLLOUT_SEARCH_DESCRIPTOR_KIND,
    AnySearchDescriptor,
    RolloutSearchDescriptor,
    SearchDescriptor,
)
from innovation_ai.search.information_sets import (
    InformationSetError as SearchInformationSetError,
)
from innovation_ai.search.information_sets import (
    InformationSetSampler as SearchInformationSetSampler,
)
from innovation_ai.search.information_sets import (
    InformationSetSpecBuilder as SearchInformationSetSpecBuilder,
)
from innovation_ai.search.minimax import (
    DeterministicSampledMinimax,
    MinimaxSelection,
    SearchInvariantError,
)
from innovation_ai.search.rollout import DeterministicSampledRollout, RolloutSelection
from innovation_ai.search.seeds import SearchRngFactory, seed_digest
from innovation_ai.training.checkpoint import (
    DEFAULT_SAMPLER_RNG_VERSION,
    DEFAULT_SEARCH_CONTINUATION_POLICY,
    DEFAULT_SELECTOR_RNG_VERSION,
    DEFAULT_SELECTOR_VERSION,
    LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    POLICY_DESCRIPTOR_SCHEMA_VERSION,
    PolicyDescriptor,
)
from innovation_ai.training.determinizations import (
    INFORMATION_SET_SAMPLER_VERSION,
    InformationSetError,
    InformationSetSampler,
    InformationSetSpecBuilder,
)
from innovation_ai.training.selection import (
    REPETITION_AWARE_SELECTOR_VERSION,
    REPETITION_HISTORY_WINDOW,
    SELECTION_RNG_VERSION,
    SUPPORTED_SELECTOR_VERSIONS,
    PolicyRngFactory,
    SelectionDomain,
    SelectionError,
    select_expansion_action,
)


class SamplerFailureMode(StrEnum):
    """The explicit response when safe determinization cannot be constructed."""

    STRICT = "strict"
    HEURISTIC = "heuristic"


class PolicySchedulerError(RuntimeError):
    """Base class for scheduler contract violations."""


class SamplerFailure(PolicySchedulerError):
    """A turn action could not be safely sampled; no real-state score was used."""

    def __init__(self, game_id: str, decision_id: int, cause: BaseException) -> None:
        self.game_id = game_id
        self.decision_id = decision_id
        self.cause = cause
        super().__init__(f"sampler failed for {game_id!r} decision {decision_id}: {cause}")


class SearchFailure(PolicySchedulerError):
    """A player-safe sampled search failed; no production fallback was attempted."""

    def __init__(self, game_id: str, decision_id: int, cause: BaseException) -> None:
        self.game_id = game_id
        self.decision_id = decision_id
        self.cause = cause
        super().__init__(f"search failed for {game_id!r} decision {decision_id}: {cause}")


class EvaluatorFailure(PolicySchedulerError):
    """One evaluator key could not score its routed sampled candidate positions."""

    def __init__(self, evaluator_key: str, cause: BaseException) -> None:
        self.evaluator_key = evaluator_key
        self.cause = cause
        super().__init__(f"evaluator {evaluator_key!r} failed: {cause}")


class FallbackActionError(PolicySchedulerError):
    """The configured heuristic returned an action outside the current legal set."""


@dataclass(frozen=True, slots=True)
class LearnedPolicyAssignment:
    """One game's immutable learned policy and opaque evaluator routing key."""

    descriptor: PolicyDescriptor
    evaluator_key: str

    def __post_init__(self) -> None:
        if not self.evaluator_key:
            raise ValueError("evaluator key cannot be empty")
        if self.descriptor.information_set_sampler_version != INFORMATION_SET_SAMPLER_VERSION:
            raise ValueError("policy uses an unsupported information-set sampler version")
        if self.descriptor.sampler_rng_version != DEFAULT_SAMPLER_RNG_VERSION:
            raise ValueError("policy uses an unsupported sampler derivation RNG version")
        if self.descriptor.selector_version not in SUPPORTED_SELECTOR_VERSIONS | {
            DEFAULT_SELECTOR_VERSION
        }:
            raise ValueError("policy uses an unsupported selection version")
        if self.descriptor.selector_rng_version not in {
            SELECTION_RNG_VERSION,
            DEFAULT_SELECTOR_RNG_VERSION,
        }:
            raise ValueError("policy uses an unsupported selector RNG version")


type SearchAuditResult = MinimaxSelection | RolloutSelection


class SearchAuditSelection(Protocol):
    """Common audit surface while downstream minimax-only reports remain unchanged."""

    @property
    def action(self) -> SemanticAction: ...

    @property
    def telemetry(self) -> Any: ...

    @property
    def statistics(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class SearchPolicyAssignment:
    """One route's complete sampled-search identity and frozen descriptor."""

    policy_id: str
    descriptor: AnySearchDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("search policy ID cannot be empty")
        _validate_search_descriptor(self.descriptor)
        if self.descriptor.sampler_seed_derivation != DEFAULT_SAMPLER_SEED_DERIVATION:
            raise ValueError("search policy uses an unsupported sampler seed derivation")


@dataclass(frozen=True, slots=True)
class PolicyDecisionAudit:
    """One submission plus auditable learned/fallback provenance."""

    submission: Submission
    chooser: PlayerId
    selection: PolicySelection | None
    handling: str
    failure: PolicySchedulerError | None = None
    search_selection: SearchAuditSelection | None = None

    def __post_init__(self) -> None:
        if self.selection is not None:
            if self.selection.game_id != self.submission.game_id:
                raise ValueError("selection and submission game IDs differ")
            if self.selection.action != self.submission.action:
                raise ValueError("selection and submission actions differ")
        if (
            self.search_selection is not None
            and self.search_selection.action != self.submission.action
        ):
            raise ValueError("search selection and submission actions differ")
        if self.handling not in {
            "learned",
            "heuristic",
            "baseline",
            "sampler-fallback",
            "evaluator-fallback",
            "search",
            "learned-search-fallback",
        }:
            raise ValueError("unknown policy decision handling")
        if self.handling in {"search", "learned-search-fallback"}:
            if (
                self.search_selection is None
                or self.selection is not None
                or self.failure is not None
            ):
                raise ValueError("search audit requires a search selection and no failure")
        elif self.search_selection is not None:
            raise ValueError("non-search audit cannot claim a search selection")
        if self.handling == "learned" and (self.selection is None or self.failure is not None):
            raise ValueError("learned audit requires a selection and no failure")
        if self.handling not in {"learned", "search", "learned-search-fallback"} and (
            self.selection is not None
        ):
            raise ValueError("fallback audit cannot claim a learned selection")


@dataclass(frozen=True, slots=True)
class PolicySchedule:
    """A pull snapshot's legal submissions and their selection audit trail."""

    submissions: tuple[Submission, ...]
    selections: tuple[PolicySelection, ...]
    audits: tuple[PolicyDecisionAudit, ...]
    search_selections: tuple[SearchAuditSelection, ...] = ()

    def __post_init__(self) -> None:
        if tuple(audit.submission for audit in self.audits) != self.submissions:
            raise ValueError("schedule submissions must exactly match audit order")
        learned_selections = tuple(
            audit.selection for audit in self.audits if audit.selection is not None
        )
        if learned_selections != self.selections:
            raise ValueError("schedule selections must exactly match learned audits")
        search_selections = tuple(
            audit.search_selection for audit in self.audits if audit.search_selection is not None
        )
        if search_selections != self.search_selections:
            raise ValueError("schedule search selections must exactly match search audits")
        keys = tuple((item.game_id, item.action.decision_id) for item in self.submissions)
        if len(set(keys)) != len(keys):
            raise ValueError("schedule cannot submit one decision more than once")


SamplerFactory = Callable[[bytes], InformationSetSampler]
SearchSamplerFactory = Callable[[bytes], SearchInformationSetSampler]


@dataclass(slots=True)
class _LearnedPending:
    request: PendingGameDecision
    assignment: LearnedPolicyAssignment
    expansion: CandidateExpansion
    sampler_seed_digest: str


PolicyAssignmentKey = str | tuple[str, PlayerId]
SearchAssignmentKey = PolicyAssignmentKey
FallbackAgentKey = tuple[str, PlayerId]


def _validate_search_descriptor(descriptor: AnySearchDescriptor) -> None:
    """Require exact descriptor-kind, algorithm, and implementation-family agreement."""

    if isinstance(descriptor, SearchDescriptor):
        if descriptor.descriptor_kind != MINIMAX_SEARCH_DESCRIPTOR_KIND:
            raise ValueError("minimax search descriptor has an unsupported descriptor kind")
        if descriptor.algorithm != DEFAULT_SEARCH_VERSION:
            raise ValueError(f"unsupported minimax search algorithm {descriptor.algorithm!r}")
        return
    if isinstance(descriptor, RolloutSearchDescriptor):
        if descriptor.descriptor_kind != ROLLOUT_SEARCH_DESCRIPTOR_KIND:
            raise ValueError("rollout search descriptor has an unsupported descriptor kind")
        if descriptor.algorithm != DEFAULT_ROLLOUT_ALGORITHM:
            raise ValueError(f"unsupported rollout search algorithm {descriptor.algorithm!r}")
        return
    raise TypeError("search descriptor must be a minimax or rollout descriptor")


class PolicyScheduler:
    """Route learned paid actions, strict sampled search, and baseline agents.

    Schema-v1 learned policies retain their historical setup/effect heuristic behavior. Schema-v2
    policies retain minimax continuation, while schema-v3 policies resolve and verify an explicitly
    typed minimax or rollout fallback. Search failure never falls back to authoritative-state
    expansion or a simple agent.
    """

    def __init__(
        self,
        assignments: Mapping[PolicyAssignmentKey, LearnedPolicyAssignment],
        evaluators: Mapping[str, BatchValueEvaluator],
        *,
        run_seed: int | str | bytes,
        generation: int,
        search_assignments: Mapping[SearchAssignmentKey, SearchPolicyAssignment] | None = None,
        search_descriptors: Mapping[str, AnySearchDescriptor] | None = None,
        sampler_failure_mode: SamplerFailureMode = SamplerFailureMode.STRICT,
        heuristic: Agent | None = None,
        fallback_agents: Mapping[FallbackAgentKey, Agent] | None = None,
        registry: CardRegistry | None = None,
        sampler_factory: SamplerFactory | None = None,
        search_sampler_factory: SearchSamplerFactory | None = None,
    ) -> None:
        self._assignments = dict(assignments)
        self._search_assignments = dict(search_assignments or {})
        self._evaluators = dict(evaluators)
        self._failure_mode = SamplerFailureMode(sampler_failure_mode)
        self._registry = registry or load_card_registry()
        self._heuristic = heuristic or SimpleHeuristicAgent(self._registry)
        self._fallback_agents = dict(fallback_agents or {})
        self._spec_builder = InformationSetSpecBuilder(self._registry)
        self._expander = TrustedCandidateExpander(self._registry)
        self._search_spec_builder = SearchInformationSetSpecBuilder(self._registry)
        self._rng_factory = PolicyRngFactory(run_seed, generation)
        self._search_rng_factory = SearchRngFactory(run_seed, generation)
        self._sampler_factory = sampler_factory or self._default_sampler
        self._search_sampler_factory = search_sampler_factory or self._default_search_sampler
        descriptors: dict[str, AnySearchDescriptor] = {
            PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id: PRODUCTION_SEARCH_DESCRIPTOR,
            PRODUCTION_ROLLOUT_SEARCH_DESCRIPTOR.descriptor_id: (
                PRODUCTION_ROLLOUT_SEARCH_DESCRIPTOR
            ),
        }
        descriptors.update(search_descriptors or {})
        for descriptor_id, descriptor in descriptors.items():
            _validate_search_descriptor(descriptor)
            if descriptor_id != descriptor.descriptor_id:
                raise ValueError("search descriptor mapping key differs from descriptor identity")
            if descriptor.sampler_seed_derivation != DEFAULT_SAMPLER_SEED_DERIVATION:
                raise ValueError("search descriptor uses an unsupported sampler seed derivation")
        self._search_descriptors = descriptors
        self._searchers: dict[str, DeterministicSampledMinimax | DeterministicSampledRollout] = {}
        self._recent_paid_actions: dict[tuple[str, PlayerId, str], deque[SemanticAction]] = {}
        self._recorded_learned_decisions: set[tuple[str, int]] = set()

    def _default_sampler(self, seed: bytes) -> InformationSetSampler:
        # Strict sampler errors are caught below so scheduler policy, rather than
        # sampler implementation details, controls fallback versus loud failure.
        return InformationSetSampler(self._registry, seed=seed, strict=True)

    def _default_search_sampler(self, seed: bytes) -> SearchInformationSetSampler:
        return SearchInformationSetSampler(self._registry, seed=seed, strict=True)

    def schedule(self, runner: PullGameRunner[GameState]) -> PolicySchedule:
        """Build submissions for one immutable ``runner.pending()`` snapshot.

        The method is side-effect free with respect to the runner.  Callers can
        inspect the returned audit before passing ``schedule.submissions`` to
        ``PullGameRunner.submit()``, which also protects simultaneous setup and
        terminal-mid-batch transition atomicity.
        """

        pending = runner.pending()
        routed_audits: dict[tuple[str, int], PolicyDecisionAudit] = {}
        learned: list[_LearnedPending] = []
        for request in pending:
            decision = request.decision
            key = (request.game_id, decision.decision_id)
            explicit_search = self._search_assignment_for(request.game_id, decision.chooser)
            if explicit_search is not None:
                routed_audits[key] = self._search_audit(runner, request, explicit_search, "search")
                continue

            assignment = self._assignment_for(request.game_id, decision.chooser)
            if assignment is None:
                routed_audits[key] = self._fallback_audit(request, "baseline")
                continue
            if decision.kind is not DecisionKind.TURN_ACTION:
                if assignment.descriptor.schema_version == LEGACY_POLICY_DESCRIPTOR_SCHEMA_VERSION:
                    # Historical schema-v1 behavior is deliberately frozen: only
                    # paid actions are learned and every continuation uses the
                    # original simple heuristic, not a seat baseline override.
                    routed_audits[key] = self._fallback_audit(
                        request, "heuristic", force_heuristic=True
                    )
                else:
                    search_assignment = self._learned_search_assignment(request, assignment)
                    routed_audits[key] = self._search_audit(
                        runner,
                        request,
                        search_assignment,
                        "learned-search-fallback",
                    )
                continue
            try:
                learned.append(self._expand_pending(runner, request, assignment))
            except (InformationSetError, CandidateExpansionError) as error:
                failure = SamplerFailure(request.game_id, decision.decision_id, error)
                routed_audits[key] = self._sampler_failure_audit(request, failure)
            except PolicySchedulerError:
                raise

        values_by_pending = self._evaluate_pending(learned)
        learned_audits: dict[tuple[str, int], PolicyDecisionAudit] = {}
        for item in learned:
            key = (item.request.game_id, item.request.decision.decision_id)
            evaluated = values_by_pending.get(key)
            if isinstance(evaluated, EvaluatorFailure):
                if self._failure_mode is SamplerFailureMode.STRICT:
                    raise evaluated
                learned_audits[key] = self._fallback_audit(
                    item.request, "evaluator-fallback", evaluated, force_heuristic=True
                )
                continue
            if evaluated is None:  # terminal-only expansion has an empty value tuple
                evaluated = ()
            try:
                selector_rng = (
                    None
                    if item.assignment.descriptor.temperature == 0.0
                    else self._rng_factory.for_decision(
                        game_id=item.request.game_id,
                        chooser=item.request.decision.chooser,
                        decision_id=item.request.decision.decision_id,
                        domain=SelectionDomain.TEMPERATURE,
                        policy_id=item.assignment.descriptor.policy_id,
                    )
                )
                selection = select_expansion_action(
                    policy_id=item.assignment.descriptor.policy_id,
                    game_id=item.request.game_id,
                    decision_id=item.request.decision.decision_id,
                    legal_actions=item.request.decision.legal_actions,
                    expansion=item.expansion,
                    evaluated_values=evaluated,
                    temperature=item.assignment.descriptor.temperature,
                    selector_version=item.assignment.descriptor.selector_version,
                    recent_actions=self._recent_actions_for(item),
                    rng=selector_rng,
                )
                selection = replace(
                    selection,
                    sampler_seed_digest=item.sampler_seed_digest,
                    selector_seed_digest=(
                        None if selector_rng is None else seed_digest(selector_rng.seed)
                    ),
                )
            except SelectionError as error:
                raise PolicySchedulerError(
                    f"invalid learned selection for {key[0]!r} decision {key[1]}: {error}"
                ) from error
            submission = Submission(item.request.game_id, selection.action)
            learned_audits[key] = PolicyDecisionAudit(
                submission,
                item.request.decision.chooser,
                selection,
                "learned",
            )

        audits = tuple(
            learned_audits.get((request.game_id, request.decision.decision_id))
            or routed_audits[(request.game_id, request.decision.decision_id)]
            for request in pending
        )
        return PolicySchedule(
            tuple(audit.submission for audit in audits),
            tuple(audit.selection for audit in audits if audit.selection is not None),
            audits,
            tuple(audit.search_selection for audit in audits if audit.search_selection is not None),
        )

    def submit(self, runner: PullGameRunner[GameState]) -> PolicySchedule:
        """Schedule one snapshot then submit its exact semantic actions to the runner."""

        schedule = self.schedule(runner)
        runner.submit(schedule.submissions)
        self.record_committed(schedule)
        return schedule

    def record_committed(self, schedule: PolicySchedule) -> None:
        """Record a successfully committed schedule for stateful selector history."""

        for audit in schedule.audits:
            if audit.selection is None:
                continue
            assignment = self._assignment_for(audit.submission.game_id, audit.chooser)
            if assignment is None:
                raise PolicySchedulerError("committed learned audit has no policy assignment")
            if assignment.descriptor.selector_version != REPETITION_AWARE_SELECTOR_VERSION:
                continue
            decision_key = (
                audit.submission.game_id,
                audit.submission.action.decision_id,
            )
            if decision_key in self._recorded_learned_decisions:
                continue
            self._recorded_learned_decisions.add(decision_key)
            key = (audit.submission.game_id, audit.chooser, assignment.descriptor.policy_id)
            history = self._recent_paid_actions.setdefault(
                key, deque(maxlen=REPETITION_HISTORY_WINDOW)
            )
            history.append(audit.submission.action)

    def _recent_actions_for(self, item: _LearnedPending) -> tuple[SemanticAction, ...]:
        descriptor = item.assignment.descriptor
        if descriptor.selector_version != REPETITION_AWARE_SELECTOR_VERSION:
            return ()
        key = (
            item.request.game_id,
            item.request.decision.chooser,
            descriptor.policy_id,
        )
        return tuple(self._recent_paid_actions.get(key, ()))

    def _assignment_for(self, game_id: str, chooser: PlayerId) -> LearnedPolicyAssignment | None:
        seat_assignment = self._assignments.get((game_id, chooser))
        if seat_assignment is not None:
            return seat_assignment
        return self._assignments.get(game_id)

    def _search_assignment_for(
        self, game_id: str, chooser: PlayerId
    ) -> SearchPolicyAssignment | None:
        seat_assignment = self._search_assignments.get((game_id, chooser))
        if seat_assignment is not None:
            return seat_assignment
        return self._search_assignments.get(game_id)

    def _learned_search_assignment(
        self,
        request: PendingGameDecision,
        assignment: LearnedPolicyAssignment,
    ) -> SearchPolicyAssignment:
        descriptor = assignment.descriptor
        descriptor_id: str
        expected_kind: str
        expected_hypothetical_policy: str | None
        if descriptor.schema_version == MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION:
            raw_descriptor_id = descriptor.search_descriptor_id
            assert raw_descriptor_id is not None  # PolicyDescriptor validates schema-v2 fields.
            descriptor_id = raw_descriptor_id
            expected_kind = MINIMAX_SEARCH_DESCRIPTOR_KIND
            expected_hypothetical_policy = None
        elif descriptor.schema_version == POLICY_DESCRIPTOR_SCHEMA_VERSION:
            raw_descriptor_id = descriptor.fallback_algorithm_descriptor_id
            raw_expected_kind = descriptor.fallback_algorithm_kind
            raw_hypothetical_policy = descriptor.hypothetical_continuation_policy
            assert raw_descriptor_id is not None  # PolicyDescriptor validates schema-v3 fields.
            assert raw_expected_kind is not None
            assert raw_hypothetical_policy is not None
            descriptor_id = raw_descriptor_id
            expected_kind = raw_expected_kind
            expected_hypothetical_policy = raw_hypothetical_policy
        else:
            raise PolicySchedulerError(
                "only schema-v2 and schema-v3 learned policies have search fallback"
            )
        try:
            search_descriptor = self._search_descriptors[descriptor_id]
        except KeyError as error:
            raise SearchFailure(
                request.game_id,
                request.decision.decision_id,
                ValueError(f"required search descriptor {descriptor_id!r} is unavailable"),
            ) from error
        if descriptor.schema_version == MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION:
            if not isinstance(search_descriptor, SearchDescriptor):
                raise SearchFailure(
                    request.game_id,
                    request.decision.decision_id,
                    ValueError("schema-v2 learned policy requires a minimax search descriptor"),
                )
        elif search_descriptor.descriptor_kind != expected_kind:
            raise SearchFailure(
                request.game_id,
                request.decision.decision_id,
                ValueError(
                    "learned fallback algorithm kind differs from its registered descriptor"
                ),
            )
        if descriptor.schema_version != MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION:
            if isinstance(search_descriptor, SearchDescriptor):
                # Schema v3 deliberately omits the minimax-named v2 field. The frozen minimax
                # continuation semantics retain their historical policy identity.
                expected_descriptor_policy = DEFAULT_SEARCH_CONTINUATION_POLICY
            elif isinstance(search_descriptor, RolloutSearchDescriptor):
                expected_descriptor_policy = search_descriptor.continuation_agent_descriptor_id
            else:  # Descriptor-map validation makes this unreachable.
                raise SearchFailure(
                    request.game_id,
                    request.decision.decision_id,
                    ValueError("unsupported learned fallback descriptor implementation family"),
                )
            if expected_hypothetical_policy != expected_descriptor_policy:
                raise SearchFailure(
                    request.game_id,
                    request.decision.decision_id,
                    ValueError(
                        "learned hypothetical continuation policy differs from descriptor semantics"
                    ),
                )
        return SearchPolicyAssignment(descriptor.policy_id, search_descriptor)

    def _search_audit(
        self,
        runner: PullGameRunner[GameState],
        request: PendingGameDecision,
        assignment: SearchPolicyAssignment,
        handling: str,
    ) -> PolicyDecisionAudit:
        """Build and sample a player-safe root, then run its exact search algorithm."""

        decision = request.decision
        try:
            state = runner.state(request.game_id)
            # This is the sole search bridge that receives authoritative state,
            # and the exact pulled Decision is mandatory at that boundary.
            spec = self._search_spec_builder.build(state, decision)
            if (
                spec.target_decision_id != decision.decision_id
                or spec.chooser is not decision.chooser
                or spec.legal_actions != decision.legal_actions
            ):
                raise SearchInvariantError(
                    "runner state no longer matches its pending search decision"
                )
            sampler_seed = self._search_rng_factory.seed_for_decision(
                game_id=request.game_id,
                chooser=decision.chooser,
                decision_id=decision.decision_id,
                policy_id=assignment.policy_id,
                search_descriptor_id=assignment.descriptor.descriptor_id,
            )
            sampler = self._search_sampler_factory(sampler_seed)
            sampled = sampler.sample_many(spec, assignment.descriptor.determinization_count)
            if len(sampled) != assignment.descriptor.determinization_count:
                raise SearchInformationSetError("sampler returned an incorrect sample count")
            if any(sample is None for sample in sampled):
                raise SearchInformationSetError("non-strict sampler returned no safe sampled state")
            samples = tuple(sample for sample in sampled if sample is not None)
            searcher = self._searchers.get(assignment.descriptor.descriptor_id)
            descriptor = assignment.descriptor
            if isinstance(descriptor, SearchDescriptor):
                if searcher is None:
                    searcher = DeterministicSampledMinimax(descriptor, registry=self._registry)
                    self._searchers[descriptor.descriptor_id] = searcher
                if not isinstance(searcher, DeterministicSampledMinimax):
                    raise SearchInvariantError("minimax descriptor resolved to a rollout searcher")
                minimax_selection = searcher.select(spec, samples)
                # Historical minimax telemetry named the sampler-seed field after the selector.
                # Preserve that schema and value for compatibility with existing audit consumers.
                selection: SearchAuditResult = replace(
                    minimax_selection,
                    telemetry=replace(
                        minimax_selection.telemetry,
                        selector_seed_digest=seed_digest(sampler_seed),
                    ),
                )
            elif isinstance(descriptor, RolloutSearchDescriptor):
                if searcher is None:
                    searcher = DeterministicSampledRollout(descriptor, registry=self._registry)
                    self._searchers[descriptor.descriptor_id] = searcher
                if not isinstance(searcher, DeterministicSampledRollout):
                    raise SearchInvariantError("rollout descriptor resolved to a minimax searcher")
                selection = searcher.select(
                    spec,
                    samples,
                    sampler_seed_digest=seed_digest(sampler_seed),
                )
            else:  # SearchPolicyAssignment and descriptor-map validation make this unreachable.
                raise SearchInvariantError("unsupported search descriptor implementation family")
            if selection.action not in decision.legal_actions:
                raise SearchInvariantError("sampled search selected an illegal root action")
        except SearchFailure:
            raise
        except Exception as error:
            # Search is production-strict.  In particular, do not invoke either
            # SimpleHeuristicAgent or a true-state action scorer after failure.
            raise SearchFailure(request.game_id, decision.decision_id, error) from error
        return PolicyDecisionAudit(
            Submission(request.game_id, selection.action),
            decision.chooser,
            None,
            handling,
            None,
            selection,
        )

    def _expand_pending(
        self,
        runner: PullGameRunner[GameState],
        request: PendingGameDecision,
        assignment: LearnedPolicyAssignment,
    ) -> _LearnedPending:
        """Build a spec then expand only synthetic sampled candidate states."""

        decision = request.decision
        state = runner.state(request.game_id)
        spec = self._spec_builder.build(state)
        # The builder independently proves these fields, but preserve the pull
        # snapshot identity as a guard against a runner changing mid-schedule.
        if spec.chooser is not decision.chooser or spec.legal_actions != decision.legal_actions:
            raise PolicySchedulerError("runner state no longer matches its pending turn decision")
        sampler_seed = self._rng_factory.seed_for_decision(
            game_id=request.game_id,
            chooser=decision.chooser,
            decision_id=decision.decision_id,
            domain=SelectionDomain.DETERMINIZATION,
            policy_id=assignment.descriptor.policy_id,
        )
        sampler = self._sampler_factory(sampler_seed)
        sampled = sampler.sample_many(spec, assignment.descriptor.determinization_count)
        if len(sampled) != assignment.descriptor.determinization_count:
            raise InformationSetError("sampler returned an incorrect sample count")
        if any(state is None for state in sampled):
            raise InformationSetError("non-strict sampler returned no safe sampled state")
        samples = tuple(state for state in sampled if state is not None)
        expansion = self._expander.expand(
            spec,
            samples,
            game_id=request.game_id,
            evaluator_key=assignment.evaluator_key,
        )
        return _LearnedPending(
            request,
            assignment,
            expansion,
            seed_digest(sampler_seed),
        )

    def _sampler_failure_audit(
        self,
        request: PendingGameDecision,
        failure: SamplerFailure,
    ) -> PolicyDecisionAudit:
        if self._failure_mode is SamplerFailureMode.STRICT:
            raise failure
        return self._fallback_audit(
            request,
            "sampler-fallback",
            failure,
            force_heuristic=True,
        )

    def _fallback_audit(
        self,
        request: PendingGameDecision,
        handling: str,
        failure: PolicySchedulerError | None = None,
        *,
        force_heuristic: bool = False,
    ) -> PolicyDecisionAudit:
        agent = (
            self._heuristic
            if force_heuristic
            else self._fallback_agents.get(
                (request.game_id, request.decision.chooser), self._heuristic
            )
        )
        action = agent.choose_action(request.decision)
        if action not in request.decision.legal_actions:
            raise FallbackActionError(
                f"heuristic returned an illegal action for {request.game_id!r} "
                f"decision {request.decision.decision_id}"
            )
        return PolicyDecisionAudit(
            Submission(request.game_id, action),
            request.decision.chooser,
            None,
            handling,
            failure,
        )

    def _evaluate_pending(
        self,
        learned: Sequence[_LearnedPending],
    ) -> dict[tuple[str, int], tuple[float, ...] | EvaluatorFailure]:
        """Evaluate each key once, isolating missing/failing evaluator keys by game."""

        output: dict[tuple[str, int], tuple[float, ...] | EvaluatorFailure] = {}
        grouped: dict[str, list[_LearnedPending]] = defaultdict(list)
        for item in learned:
            # An expansion has one evaluator key by construction; terminal-only
            # decisions need no evaluator and remain independently selectable.
            if item.expansion.routes:
                grouped[item.assignment.evaluator_key].append(item)
            else:
                output[(item.request.game_id, item.request.decision.decision_id)] = ()

        for evaluator_key, items in grouped.items():
            routes = tuple(route for item in items for route in item.expansion.routes)
            positions = tuple(position for item in items for position in item.expansion.positions)
            try:
                evaluator = self._evaluators[evaluator_key]
                values = tuple(float(value) for value in evaluator.evaluate(positions))
                if len(values) != len(routes):
                    raise ValueError("evaluator returned an incorrect result count")
                if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                    raise ValueError("evaluator returned a non-probability value")
            except (KeyError, ValueError, TypeError, RuntimeError) as error:
                failure = EvaluatorFailure(evaluator_key, error)
                for item in items:
                    output[(item.request.game_id, item.request.decision.decision_id)] = failure
                continue

            offset = 0
            for item in items:
                count = len(item.expansion.routes)
                output[(item.request.game_id, item.request.decision.decision_id)] = values[
                    offset : offset + count
                ]
                offset += count
        return output


AfterstatePolicyScheduler = PolicyScheduler

__all__ = [
    "AfterstatePolicyScheduler",
    "EvaluatorFailure",
    "FallbackActionError",
    "LearnedPolicyAssignment",
    "PolicyDecisionAudit",
    "PolicySchedule",
    "PolicyScheduler",
    "PolicySchedulerError",
    "SamplerFailure",
    "SamplerFailureMode",
    "SearchAssignmentKey",
    "SearchAuditResult",
    "SearchAuditSelection",
    "SearchFailure",
    "SearchPolicyAssignment",
    "SearchSamplerFactory",
]
