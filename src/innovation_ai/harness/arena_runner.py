"""Execution and champion-promotion boundary for paired arena manifests.

The lightweight :mod:`harness.arena` policy descriptor is deliberately aliased
below as ``ArenaPolicyDescriptor``.  Learned runtime descriptors are the richer
``training.checkpoint.PolicyDescriptor`` and are never conflated with it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, cast

from innovation_ai.agents.base import Agent
from innovation_ai.agents.heuristic import SimpleHeuristicAgent
from innovation_ai.agents.random import RandomAgent
from innovation_ai.harness.arena import (
    ArenaGameResult,
    ArenaManifest,
    ArenaReport,
    ArenaResult,
    ArenaSchemaError,
    MatchPair,
    PlannedGame,
    arena_game_result_from_record,
    build_arena_report,
    dumps_arena_report,
    dumps_arena_result,
    validate_arena_result,
)
from innovation_ai.harness.arena import (
    PolicyDescriptor as ArenaPolicyDescriptor,
)
from innovation_ai.harness.engine import RunnerEngine
from innovation_ai.harness.records import RunnerRecording
from innovation_ai.harness.runner import GameBlockedError, GameSpec, PullGameRunner, Submission
from innovation_ai.innovation.actions import action_payload
from innovation_ai.innovation.serialization import JsonValue, canonical_json, parse_json
from innovation_ai.innovation.state import GameState, state_hash
from innovation_ai.innovation.types import PlayerId
from innovation_ai.search.contracts import PRODUCTION_SEARCH_DESCRIPTOR, SearchDescriptor

if TYPE_CHECKING:
    from innovation_ai.search.minimax import SearchStatistics
    from innovation_ai.training.checkpoint import PolicyDescriptor as TrainingPolicyDescriptor
    from innovation_ai.training.inference import FrozenEvaluatorCache

PROMOTION_PAIR_COUNT = 200
CHAMPION_MANIFEST_FORMAT = "innovation-ai-champion-manifest"
CHAMPION_POINTER_FORMAT = "innovation-ai-champion-pointer"
CHAMPION_SCHEMA_VERSION = 1


class ArenaExecutionError(RuntimeError):
    """Arena execution cannot satisfy the manifest's fixed contract."""


class ArenaActionLimitError(ArenaExecutionError):
    """A planned game crossed the executor's defensive action ceiling."""

    def __init__(self, diagnostic: Mapping[str, object]) -> None:
        self.diagnostic = dict(diagnostic)
        super().__init__(
            f"game {self.diagnostic['game_id']!r} exceeded action ceiling "
            f"{self.diagnostic['action_ceiling']}"
        )


class PromotionDecision(StrEnum):
    BOOTSTRAPPED = "bootstrapped"
    PROMOTED = "promoted"
    RETAINED = "retained"


@dataclass(frozen=True, slots=True)
class ArenaSearchTelemetry:
    """Aggregate sampled-search work committed across one arena execution."""

    routes: int = 0
    nodes: int = 0
    recursive_engine_transitions: int = 0
    root_transitions: int = 0
    mandatory_setup_transitions: int = 0
    transposition_hits: int = 0
    repeated_position_cutoffs: int = 0
    budget_cutoff_routes: int = 0
    immediate_leaf_fallback_routes: int = 0
    decisions: int = 0

    @property
    def recursive_transitions(self) -> int:
        return self.recursive_engine_transitions

    @property
    def setup_transitions(self) -> int:
        return self.mandatory_setup_transitions

    @property
    def tt_hits(self) -> int:
        return self.transposition_hits

    @property
    def cycle_cutoffs(self) -> int:
        return self.repeated_position_cutoffs

    @property
    def immediate_fallback_routes(self) -> int:
        return self.immediate_leaf_fallback_routes


@dataclass(frozen=True, slots=True)
class ArenaExecution:
    """Fully validated raw results, deterministic report, and execution-only telemetry."""

    result: ArenaResult
    report: ArenaReport
    search_telemetry: ArenaSearchTelemetry = ArenaSearchTelemetry()


@dataclass(frozen=True, slots=True)
class ChampionManifest:
    """Reference-only champion state; it contains no copied checkpoint bytes."""

    policy_id: str
    checkpoint_id: str
    source_arena_id: str | None
    decision: PromotionDecision
    statistical_claim: bool
    schema_version: int = CHAMPION_SCHEMA_VERSION
    format: str = CHAMPION_MANIFEST_FORMAT

    def __post_init__(self) -> None:
        if not self.policy_id or not self.checkpoint_id:
            raise ArenaSchemaError("champion policy and checkpoint IDs cannot be empty")
        if self.source_arena_id is not None and not self.source_arena_id:
            raise ArenaSchemaError("champion source arena ID cannot be empty")
        if self.schema_version != CHAMPION_SCHEMA_VERSION:
            raise ArenaSchemaError("unsupported champion manifest schema version")
        if self.format != CHAMPION_MANIFEST_FORMAT:
            raise ArenaSchemaError("unsupported champion manifest format")
        if self.decision is PromotionDecision.BOOTSTRAPPED and self.statistical_claim:
            raise ArenaSchemaError("bootstrap champion cannot make a statistical claim")
        if self.decision is PromotionDecision.PROMOTED and not self.statistical_claim:
            raise ArenaSchemaError("promotion must make a statistical claim")


@dataclass(frozen=True, slots=True)
class ChampionPointer:
    """Small atomically replaced pointer to an immutable champion manifest."""

    manifest_sha256: str
    policy_id: str
    checkpoint_id: str
    schema_version: int = CHAMPION_SCHEMA_VERSION
    format: str = CHAMPION_POINTER_FORMAT

    def __post_init__(self) -> None:
        if not self.manifest_sha256.startswith("sha256:") or len(self.manifest_sha256) != 71:
            raise ArenaSchemaError("champion pointer manifest digest is invalid")
        if not self.policy_id or not self.checkpoint_id:
            raise ArenaSchemaError("champion pointer IDs cannot be empty")
        if self.schema_version != CHAMPION_SCHEMA_VERSION or self.format != CHAMPION_POINTER_FORMAT:
            raise ArenaSchemaError("unsupported champion pointer schema")


@dataclass(frozen=True, slots=True)
class PromotionOutcome:
    decision: PromotionDecision
    champion: ChampionManifest
    lower_bound: float | None


BaselineFactory = Callable[[int], Agent]


class ArenaRunner:
    """Run only predeclared games, routing the policy selected for each physical seat.

    ``policy_descriptors`` supplies arena identities.  ``learned_policies`` is keyed by
    those arena IDs, while its values are training descriptors used by the evaluator cache.
    This explicit separation resolves the two intentionally different PolicyDescriptor types.
    """

    def __init__(
        self,
        engine: RunnerEngine[GameState],
        policy_descriptors: Mapping[str, ArenaPolicyDescriptor],
        *,
        learned_policies: Mapping[str, TrainingPolicyDescriptor] = {},
        evaluator_cache: FrozenEvaluatorCache | None = None,
        search_policy_descriptors: Mapping[str, SearchDescriptor] = {},
        search_descriptors: Mapping[str, SearchDescriptor] | None = None,
        baseline_factories: Mapping[str, BaselineFactory] = {},
        max_actions: int = 10_000,
        run_seed: int | str | bytes = 0,
        generation: int = 0,
    ) -> None:
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self._engine = engine
        self._policies = dict(policy_descriptors)
        self._learned = dict(learned_policies)
        self._cache = evaluator_cache
        self._search_policy_descriptors = dict(search_policy_descriptors)
        self._search_descriptors = {
            PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id: PRODUCTION_SEARCH_DESCRIPTOR
        }
        self._search_descriptors.update(search_descriptors or {})
        self._factories = dict(baseline_factories)
        self._max_actions = max_actions
        self._run_seed = run_seed
        self._generation = generation
        for descriptor_id, registered_descriptor in self._search_descriptors.items():
            if descriptor_id != registered_descriptor.descriptor_id:
                raise ArenaExecutionError(
                    "search descriptor mapping key must equal descriptor identity"
                )
        for policy_id, descriptor in self._policies.items():
            if policy_id != descriptor.policy_id:
                raise ArenaExecutionError("policy descriptor mapping key must equal policy ID")
            search = policy_id in self._search_policy_descriptors
            if descriptor.policy_kind == "search-heuristic" and not search:
                raise ArenaExecutionError(f"search policy {policy_id!r} has no search descriptor")
            if search and descriptor.policy_kind != "search-heuristic":
                raise ArenaExecutionError(
                    f"non-search policy {policy_id!r} has a search descriptor"
                )
            if search:
                search_descriptor = self._search_policy_descriptors[policy_id]
                registered = self._search_descriptors.get(search_descriptor.descriptor_id)
                if registered != search_descriptor:
                    raise ArenaExecutionError(
                        f"search policy {policy_id!r} descriptor identity is unavailable"
                    )
            learned = policy_id in self._learned
            if descriptor.policy_kind == "learned" and not learned:
                raise ArenaExecutionError(
                    f"learned policy {policy_id!r} has no training descriptor"
                )
            if learned and descriptor.policy_kind != "learned":
                raise ArenaExecutionError(
                    f"non-learned policy {policy_id!r} has a training descriptor"
                )
            if learned and descriptor.checkpoint_id != self._learned[policy_id].checkpoint_id:
                raise ArenaExecutionError(
                    f"learned policy {policy_id!r} checkpoint reference does not match "
                    "its descriptor"
                )
        for policy_id in self._search_policy_descriptors:
            if policy_id not in self._policies:
                raise ArenaExecutionError(
                    f"search descriptor references unknown policy {policy_id!r}"
                )
        if self._learned and self._cache is None:
            raise ArenaExecutionError("learned policies require a FrozenEvaluatorCache")

    def execute(self, manifest: ArenaManifest) -> ArenaExecution:
        """Execute every and only manifest game, then validate and report it."""

        if manifest.temperature != 0.0:
            raise ArenaExecutionError("arena execution requires deterministic temperature zero")
        plan_by_game = {
            planned.game_id: (pair, planned)
            for pair in manifest.match_pairs
            for planned in pair.games
        }
        for policy_id in {manifest.candidate_policy_id} | {
            pair.opponent_policy_id for pair in manifest.match_pairs
        }:
            self._require_policy(policy_id)
        for policy_id, descriptor in self._learned.items():
            if descriptor.temperature != 0.0:
                raise ArenaExecutionError(f"learned policy {policy_id!r} is not temperature zero")

        specs = tuple(
            GameSpec(game_id, pair.setup_seed) for game_id, (pair, _) in plan_by_game.items()
        )
        runner = PullGameRunner(self._engine, specs, recording=RunnerRecording())
        baselines: dict[tuple[str, PlayerId], Agent] = {}
        submitted_actions = {game_id: 0 for game_id in plan_by_game}
        scheduler = None
        routed_policy_ids = {
            self._policy_for(pair, planned, seat)
            for pair, planned in plan_by_game.values()
            for seat in PlayerId
        }
        if any(
            self._is_learned(policy_id) or self._is_search(policy_id)
            for policy_id in routed_policy_ids
        ):
            from innovation_ai.harness.policy_scheduler import (
                LearnedPolicyAssignment,
                PolicyAssignmentKey,
                PolicyScheduler,
                SearchPolicyAssignment,
            )

            assignments: dict[PolicyAssignmentKey, LearnedPolicyAssignment] = {}
            search_assignments: dict[PolicyAssignmentKey, SearchPolicyAssignment] = {}
            evaluators = {}
            fallback_agents: dict[tuple[str, PlayerId], Agent] = {}
            for game_id, (pair, planned) in plan_by_game.items():
                for seat in PlayerId:
                    policy_id = self._policy_for(pair, planned, seat)
                    if self._is_learned(policy_id):
                        descriptor = self._learned[policy_id]
                        key = descriptor.policy_id
                        assignments[(game_id, seat)] = LearnedPolicyAssignment(descriptor, key)
                        assert self._cache is not None
                        evaluators[key] = self._cache.evaluator_for(descriptor)
                    elif self._is_search(policy_id):
                        search_assignments[(game_id, seat)] = SearchPolicyAssignment(
                            policy_id, self._search_policy_descriptors[policy_id]
                        )
                    else:
                        fallback_agents[(game_id, seat)] = self._baseline_agent(
                            policy_id, pair, seat
                        )
            scheduler = PolicyScheduler(
                assignments,
                evaluators,
                search_assignments=search_assignments,
                search_descriptors=self._search_descriptors,
                fallback_agents=fallback_agents,
                run_seed=self._run_seed,
                generation=self._generation,
            )
        else:
            for game_id, (pair, planned) in plan_by_game.items():
                for seat in PlayerId:
                    policy_id = self._policy_for(pair, planned, seat)
                    baselines[(game_id, seat)] = self._baseline_agent(policy_id, pair, seat)

        submitted_actions = {game_id: 0 for game_id in plan_by_game}
        action_tails: dict[str, list[dict[str, object]]] = {game_id: [] for game_id in plan_by_game}
        search_statistics: list[SearchStatistics] = []
        while True:
            pending = runner.pending()
            if not pending:
                blocked = runner.blocked_game_ids()
                if blocked:
                    raise GameBlockedError(f"arena games blocked without a decision: {blocked}")
                break
            if any(submitted_actions[item.game_id] >= self._max_actions for item in pending):
                game_id = next(
                    item.game_id
                    for item in pending
                    if submitted_actions[item.game_id] >= self._max_actions
                )
                pair, planned = plan_by_game[game_id]
                raise ArenaActionLimitError(
                    {
                        "format": "innovation-ai-arena-action-ceiling-failure",
                        "schema_version": 1,
                        "arena_id": manifest.arena_id,
                        "pair_id": pair.pair_id,
                        "game_id": game_id,
                        "setup_seed": pair.setup_seed,
                        "candidate_policy_id": pair.candidate_policy_id,
                        "opponent_policy_id": pair.opponent_policy_id,
                        "candidate_seat": planned.candidate_seat.value,
                        "action_count": submitted_actions[game_id],
                        "action_ceiling": self._max_actions,
                        "current_state_hash": state_hash(runner.state(game_id)),
                        "action_tail": action_tails[game_id],
                    }
                )

            policy_schedule = None
            if scheduler is not None:
                policy_schedule = scheduler.schedule(runner)
                submissions = list(policy_schedule.submissions)
            else:
                submissions = [
                    Submission(
                        item.game_id,
                        baselines[(item.game_id, item.decision.chooser)].choose_action(
                            item.decision
                        ),
                    )
                    for item in pending
                ]
            if len(submissions) != len(pending):
                raise ArenaExecutionError(
                    "arena policy routing did not answer every pending decision"
                )
            runner.submit(submissions)
            if scheduler is not None:
                assert policy_schedule is not None
                scheduler.record_committed(policy_schedule)
                search_statistics.extend(
                    selection.statistics for selection in policy_schedule.search_selections
                )
            for submission in submissions:
                submitted_actions[submission.game_id] += 1
                action_tails[submission.game_id].append(
                    {
                        "sequence": submitted_actions[submission.game_id],
                        "action": action_payload(submission.action),
                        "resulting_state_hash": state_hash(runner.state(submission.game_id)),
                    }
                )
                action_tails[submission.game_id] = action_tails[submission.game_id][-32:]

        games: list[ArenaGameResult] = []
        for pair in manifest.match_pairs:
            for planned in pair.games:
                completed = runner.result(planned.game_id)
                if completed is None:
                    raise ArenaExecutionError(f"planned game {planned.game_id!r} did not terminate")
                games.append(arena_game_result_from_record(pair, completed.record))
        result = ArenaResult.for_manifest(manifest, games)
        validate_arena_result(manifest, result)
        telemetry = ArenaSearchTelemetry(
            routes=sum(item.routes for item in search_statistics),
            nodes=sum(item.nodes for item in search_statistics),
            recursive_engine_transitions=sum(
                item.recursive_engine_transitions for item in search_statistics
            ),
            root_transitions=sum(item.root_transitions for item in search_statistics),
            mandatory_setup_transitions=sum(
                item.mandatory_setup_transitions for item in search_statistics
            ),
            transposition_hits=sum(item.transposition_hits for item in search_statistics),
            repeated_position_cutoffs=sum(
                item.repeated_position_cutoffs for item in search_statistics
            ),
            budget_cutoff_routes=sum(item.budget_cutoff_routes for item in search_statistics),
            immediate_leaf_fallback_routes=sum(
                item.immediate_leaf_fallback_routes for item in search_statistics
            ),
            decisions=len(search_statistics),
        )
        return ArenaExecution(result, build_arena_report(manifest, result), telemetry)

    def _require_policy(self, policy_id: str) -> ArenaPolicyDescriptor:
        try:
            return self._policies[policy_id]
        except KeyError as error:
            raise ArenaExecutionError(
                f"manifest references unknown policy {policy_id!r}"
            ) from error

    def _is_learned(self, policy_id: str) -> bool:
        return policy_id in self._learned

    def _is_search(self, policy_id: str) -> bool:
        return policy_id in self._search_policy_descriptors

    @staticmethod
    def _policy_for(pair: MatchPair, planned: PlannedGame, seat: PlayerId) -> str:
        """Route a physical seat against the candidate seat of this exact planned game."""

        return (
            pair.candidate_policy_id if seat is planned.candidate_seat else pair.opponent_policy_id
        )

    def _baseline_agent(self, policy_id: str, pair: MatchPair, seat: PlayerId) -> Agent:
        descriptor = self._require_policy(policy_id)
        if descriptor.policy_kind == "learned":
            # Setup/effect decisions are intentionally served by the safe deterministic fallback.
            return SimpleHeuristicAgent()
        factory = self._factories.get(descriptor.policy_kind)
        if factory is not None:
            return factory(self._baseline_seed(policy_id, pair, seat))
        if descriptor.policy_kind in {"heuristic", "simple-heuristic"}:
            return SimpleHeuristicAgent()
        if descriptor.policy_kind == "random":
            return RandomAgent(self._baseline_seed(policy_id, pair, seat))
        raise ArenaExecutionError(f"unsupported baseline policy kind {descriptor.policy_kind!r}")

    @staticmethod
    def _baseline_seed(policy_id: str, pair: MatchPair, seat: PlayerId) -> int:
        data = f"arena-baseline-v1\0{policy_id}\0{pair.setup_seed}\0{seat.value}".encode()
        return int.from_bytes(sha256(data).digest()[:8], "big")


def promotion_outcome(
    incumbent: ChampionManifest | None,
    candidate: ArenaPolicyDescriptor,
    manifest: ArenaManifest,
    report: ArenaReport,
) -> PromotionOutcome:
    """Apply the non-adaptive promotion rule without writing or copying artifacts."""

    if candidate.policy_kind != "learned" or candidate.checkpoint_id is None:
        raise ArenaExecutionError("only a learned policy with a checkpoint can become champion")
    if incumbent is None:
        return PromotionOutcome(
            PromotionDecision.BOOTSTRAPPED,
            ChampionManifest(
                candidate.policy_id,
                candidate.checkpoint_id,
                None,
                PromotionDecision.BOOTSTRAPPED,
                False,
            ),
            None,
        )
    if manifest.candidate_policy_id != candidate.policy_id:
        raise ArenaExecutionError("promotion manifest candidate differs from candidate descriptor")
    if len(manifest.match_pairs) != PROMOTION_PAIR_COUNT:
        raise ArenaExecutionError(
            f"promotion requires exactly {PROMOTION_PAIR_COUNT} predeclared pairs"
        )
    if {pair.opponent_policy_id for pair in manifest.match_pairs} != {incumbent.policy_id}:
        raise ArenaExecutionError("promotion arena must compare only with the incumbent champion")
    lower = report.all_pairs.confidence_interval.lower
    if lower > 0.5:
        champion = ChampionManifest(
            candidate.policy_id,
            candidate.checkpoint_id,
            manifest.arena_id,
            PromotionDecision.PROMOTED,
            True,
        )
        return PromotionOutcome(PromotionDecision.PROMOTED, champion, lower)
    return PromotionOutcome(PromotionDecision.RETAINED, incumbent, lower)


def champion_manifest_payload(value: ChampionManifest) -> dict[str, object]:
    return {
        "format": value.format,
        "schema_version": value.schema_version,
        "policy_id": value.policy_id,
        "checkpoint_id": value.checkpoint_id,
        "source_arena_id": value.source_arena_id,
        "decision": value.decision.value,
        "statistical_claim": value.statistical_claim,
    }


def champion_pointer_payload(value: ChampionPointer) -> dict[str, object]:
    return {
        "format": value.format,
        "schema_version": value.schema_version,
        "manifest_sha256": value.manifest_sha256,
        "policy_id": value.policy_id,
        "checkpoint_id": value.checkpoint_id,
    }


def dumps_champion_manifest(value: ChampionManifest) -> str:
    return canonical_json(cast(JsonValue, champion_manifest_payload(value)))


def dumps_champion_pointer(value: ChampionPointer) -> str:
    return canonical_json(cast(JsonValue, champion_pointer_payload(value)))


def loads_champion_manifest(text: str) -> ChampionManifest:
    value = parse_json(text)
    if not isinstance(value, dict) or set(value) != {
        "format",
        "schema_version",
        "policy_id",
        "checkpoint_id",
        "source_arena_id",
        "decision",
        "statistical_claim",
    }:
        raise ArenaSchemaError("champion manifest keys differ from schema")
    if (
        not isinstance(value["policy_id"], str)
        or not isinstance(value["checkpoint_id"], str)
        or not isinstance(value["decision"], str)
        or isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or not isinstance(value["format"], str)
        or not isinstance(value["source_arena_id"], (str, type(None)))
        or not isinstance(value["statistical_claim"], bool)
    ):
        raise ArenaSchemaError("champion manifest has invalid field types")
    return ChampionManifest(
        value["policy_id"],
        value["checkpoint_id"],
        value["source_arena_id"],
        PromotionDecision(value["decision"]),
        value["statistical_claim"],
        value["schema_version"],
        value["format"],
    )


def loads_champion_pointer(text: str) -> ChampionPointer:
    value = parse_json(text)
    if not isinstance(value, dict) or set(value) != {
        "format",
        "schema_version",
        "manifest_sha256",
        "policy_id",
        "checkpoint_id",
    }:
        raise ArenaSchemaError("champion pointer keys differ from schema")
    if (
        not isinstance(value["manifest_sha256"], str)
        or not isinstance(value["policy_id"], str)
        or not isinstance(value["checkpoint_id"], str)
        or isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or not isinstance(value["format"], str)
    ):
        raise ArenaSchemaError("champion pointer has invalid field types")
    return ChampionPointer(
        value["manifest_sha256"],
        value["policy_id"],
        value["checkpoint_id"],
        value["schema_version"],
        value["format"],
    )


def write_execution_artifacts(directory: str | Path, execution: ArenaExecution) -> None:
    """Atomically publish immutable canonical result/report JSON under ``directory``."""
    root = Path(directory)
    _atomic_write_immutable(
        root / "arena-result.json", dumps_arena_result(execution.result).encode("ascii")
    )
    _atomic_write_immutable(
        root / "arena-report.json", dumps_arena_report(execution.report).encode("ascii")
    )


def write_champion(directory: str | Path, champion: ChampionManifest) -> ChampionPointer:
    """Write an immutable content-addressed manifest then atomically replace its pointer."""
    root = Path(directory)
    data = dumps_champion_manifest(champion).encode("ascii")
    digest = "sha256:" + sha256(data).hexdigest()
    _atomic_write(root / "champions" / f"{digest[7:]}.json", data)
    pointer = ChampionPointer(digest, champion.policy_id, champion.checkpoint_id)
    _atomic_write(root / "champion.json", dumps_champion_pointer(pointer).encode("ascii"))
    return pointer


def _atomic_write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ArenaExecutionError(f"existing immutable arena artifact differs: {path}")
        return
    _atomic_write(path, data)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
