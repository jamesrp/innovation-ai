"""Pull-runner scheduler for the information-safe learned afterstate policy.

The scheduler is deliberately the only orchestration layer that sees both a
live runner state (to build an audited information-set specification) and value
evaluators.  It never applies candidate actions to that live state: all learned
candidate expansion is performed solely on sampled reconstructed states.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from innovation_ai.agents.base import Agent
from innovation_ai.agents.heuristic import SimpleHeuristicAgent
from innovation_ai.harness.afterstates import (
    CandidateExpansion,
    CandidateExpansionError,
    TrustedCandidateExpander,
)
from innovation_ai.harness.policy import BatchValueEvaluator, PolicySelection
from innovation_ai.harness.runner import PendingGameDecision, PullGameRunner, Submission
from innovation_ai.innovation.actions import DecisionKind
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import GameState
from innovation_ai.innovation.types import PlayerId
from innovation_ai.training.checkpoint import (
    DEFAULT_SAMPLER_RNG_VERSION,
    DEFAULT_SELECTOR_RNG_VERSION,
    DEFAULT_SELECTOR_VERSION,
    PolicyDescriptor,
)
from innovation_ai.training.determinizations import (
    INFORMATION_SET_SAMPLER_VERSION,
    InformationSetError,
    InformationSetSampler,
    InformationSetSpecBuilder,
)
from innovation_ai.training.selection import (
    SELECTION_RNG_VERSION,
    SELECTOR_VERSION,
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
        if self.descriptor.selector_version not in {SELECTOR_VERSION, DEFAULT_SELECTOR_VERSION}:
            raise ValueError("policy uses an unsupported selection version")
        if self.descriptor.selector_rng_version not in {
            SELECTION_RNG_VERSION,
            DEFAULT_SELECTOR_RNG_VERSION,
        }:
            raise ValueError("policy uses an unsupported selector RNG version")


@dataclass(frozen=True, slots=True)
class PolicyDecisionAudit:
    """One submission plus auditable learned/fallback provenance."""

    submission: Submission
    selection: PolicySelection | None
    handling: str
    failure: PolicySchedulerError | None = None

    def __post_init__(self) -> None:
        if self.selection is not None:
            if self.selection.game_id != self.submission.game_id:
                raise ValueError("selection and submission game IDs differ")
            if self.selection.action != self.submission.action:
                raise ValueError("selection and submission actions differ")
        if self.handling not in {
            "learned",
            "heuristic",
            "baseline",
            "sampler-fallback",
            "evaluator-fallback",
        }:
            raise ValueError("unknown policy decision handling")
        if self.handling == "learned" and (self.selection is None or self.failure is not None):
            raise ValueError("learned audit requires a selection and no failure")
        if self.handling != "learned" and self.selection is not None:
            raise ValueError("fallback audit cannot claim a learned selection")


@dataclass(frozen=True, slots=True)
class PolicySchedule:
    """A pull snapshot's legal submissions and their selection audit trail."""

    submissions: tuple[Submission, ...]
    selections: tuple[PolicySelection, ...]
    audits: tuple[PolicyDecisionAudit, ...]

    def __post_init__(self) -> None:
        if tuple(audit.submission for audit in self.audits) != self.submissions:
            raise ValueError("schedule submissions must exactly match audit order")
        learned_selections = tuple(
            audit.selection for audit in self.audits if audit.selection is not None
        )
        if learned_selections != self.selections:
            raise ValueError("schedule selections must exactly match learned audits")
        keys = tuple((item.game_id, item.action.decision_id) for item in self.submissions)
        if len(set(keys)) != len(keys):
            raise ValueError("schedule cannot submit one decision more than once")


SamplerFactory = Callable[[bytes], InformationSetSampler]


@dataclass(slots=True)
class _LearnedPending:
    request: PendingGameDecision
    assignment: LearnedPolicyAssignment
    expansion: CandidateExpansion


PolicyAssignmentKey = str | tuple[str, PlayerId]
FallbackAgentKey = tuple[str, PlayerId]


class PolicyScheduler:
    """Flatten safe afterstates across pending games and route values back exactly.

    Setup and nested effect decisions bypass this learned path and are answered
    by the configured player-safe heuristic.  A safe-sampler error either raises
    :class:`SamplerFailure` in strict mode or records a heuristic fallback; it
    never evaluates any candidate from the real authoritative state.
    """

    def __init__(
        self,
        assignments: Mapping[PolicyAssignmentKey, LearnedPolicyAssignment],
        evaluators: Mapping[str, BatchValueEvaluator],
        *,
        run_seed: int | str | bytes,
        generation: int,
        sampler_failure_mode: SamplerFailureMode = SamplerFailureMode.STRICT,
        heuristic: Agent | None = None,
        fallback_agents: Mapping[FallbackAgentKey, Agent] | None = None,
        registry: CardRegistry | None = None,
        sampler_factory: SamplerFactory | None = None,
    ) -> None:
        self._assignments = dict(assignments)
        self._evaluators = dict(evaluators)
        self._failure_mode = SamplerFailureMode(sampler_failure_mode)
        self._registry = registry or load_card_registry()
        self._heuristic = heuristic or SimpleHeuristicAgent(self._registry)
        self._fallback_agents = dict(fallback_agents or {})
        self._spec_builder = InformationSetSpecBuilder(self._registry)
        self._expander = TrustedCandidateExpander(self._registry)
        self._rng_factory = PolicyRngFactory(run_seed, generation)
        self._sampler_factory = sampler_factory or self._default_sampler

    def _default_sampler(self, seed: bytes) -> InformationSetSampler:
        # Strict sampler errors are caught below so scheduler policy, rather than
        # sampler implementation details, controls fallback versus loud failure.
        return InformationSetSampler(self._registry, seed=seed, strict=True)

    def schedule(self, runner: PullGameRunner[GameState]) -> PolicySchedule:
        """Build submissions for one immutable ``runner.pending()`` snapshot.

        The method is side-effect free with respect to the runner.  Callers can
        inspect the returned audit before passing ``schedule.submissions`` to
        ``PullGameRunner.submit()``, which also protects simultaneous setup and
        terminal-mid-batch transition atomicity.
        """

        pending = runner.pending()
        fallback_audits: dict[tuple[str, int], PolicyDecisionAudit] = {}
        learned: list[_LearnedPending] = []
        for request in pending:
            decision = request.decision
            if decision.kind is not DecisionKind.TURN_ACTION:
                fallback_audits[(request.game_id, decision.decision_id)] = self._fallback_audit(
                    request,
                    "heuristic"
                    if self._assignment_for(request.game_id, decision.chooser) is not None
                    else "baseline",
                )
                continue
            assignment = self._assignment_for(request.game_id, decision.chooser)
            if assignment is None:
                fallback_audits[(request.game_id, decision.decision_id)] = self._fallback_audit(
                    request, "baseline"
                )
                continue
            try:
                learned.append(self._expand_pending(runner, request, assignment))
            except (InformationSetError, CandidateExpansionError) as error:
                failure = SamplerFailure(request.game_id, decision.decision_id, error)
                key = (request.game_id, decision.decision_id)
                fallback_audits[key] = self._sampler_failure_audit(request, failure)
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
                selection = select_expansion_action(
                    policy_id=item.assignment.descriptor.policy_id,
                    game_id=item.request.game_id,
                    decision_id=item.request.decision.decision_id,
                    legal_actions=item.request.decision.legal_actions,
                    expansion=item.expansion,
                    evaluated_values=evaluated,
                    temperature=item.assignment.descriptor.temperature,
                    rng=(
                        None
                        if item.assignment.descriptor.temperature == 0.0
                        else self._rng_factory.for_decision(
                            game_id=item.request.game_id,
                            chooser=item.request.decision.chooser,
                            decision_id=item.request.decision.decision_id,
                            domain=SelectionDomain.TEMPERATURE,
                            policy_id=item.assignment.descriptor.policy_id,
                        )
                    ),
                )
            except SelectionError as error:
                raise PolicySchedulerError(
                    f"invalid learned selection for {key[0]!r} decision {key[1]}: {error}"
                ) from error
            submission = Submission(item.request.game_id, selection.action)
            learned_audits[key] = PolicyDecisionAudit(submission, selection, "learned")

        audits = tuple(
            learned_audits.get((request.game_id, request.decision.decision_id))
            or fallback_audits[(request.game_id, request.decision.decision_id)]
            for request in pending
        )
        return PolicySchedule(
            tuple(audit.submission for audit in audits),
            tuple(audit.selection for audit in audits if audit.selection is not None),
            audits,
        )

    def submit(self, runner: PullGameRunner[GameState]) -> PolicySchedule:
        """Schedule one snapshot then submit its exact semantic actions to the runner."""

        schedule = self.schedule(runner)
        runner.submit(schedule.submissions)
        return schedule

    def _assignment_for(self, game_id: str, chooser: PlayerId) -> LearnedPolicyAssignment | None:
        seat_assignment = self._assignments.get((game_id, chooser))
        if seat_assignment is not None:
            return seat_assignment
        return self._assignments.get(game_id)

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
        return _LearnedPending(request, assignment, expansion)

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
        return PolicyDecisionAudit(Submission(request.game_id, action), None, handling, failure)

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
]
