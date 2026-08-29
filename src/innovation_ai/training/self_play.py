"""Immutable, resumable compact-replay self-play orchestration.

This module intentionally owns only generation artifacts and actor lifecycle.  Dataset
materialization/training are injected callables so command integration can live elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from innovation_ai.agents.base import Agent
from innovation_ai.agents.descriptors import (
    RANDOM_AGENT_DESCRIPTOR,
    SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    AgentDescriptor,
)
from innovation_ai.agents.heuristic import SimpleHeuristicAgent
from innovation_ai.agents.random import AGENT_RNG_VERSION, RandomAgent
from innovation_ai.harness.actor_pool import BoundedActorPool
from innovation_ai.harness.engine import InnovationEngineAdapter
from innovation_ai.harness.records import (
    GameResult,
    RunnerRecording,
    SemanticActionEvent,
    SemanticActionSink,
)
from innovation_ai.harness.runner import GameSpec, Submission
from innovation_ai.harness.seeds import agent_seed, setup_seed
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.state import GameState
from innovation_ai.innovation.types import PlayerId

if TYPE_CHECKING:
    from innovation_ai.training.checkpoint import PolicyDescriptor
from innovation_ai.training.compact_replay import (
    CompactEpisode,
    CompactReplayProvenance,
    CompactReplayRecorder,
    CompactReplayShardManifest,
    DeterminizationProvenance,
    ExplorationProvenance,
    SeatPolicyProvenance,
    read_compact_replay_shard,
    sha256_digest,
    write_compact_replay_shard,
)

SELF_PLAY_FORMAT = "innovation-ai-self-play-generation"
SELF_PLAY_SCHEMA_VERSION = 1


class SelfPlayError(RuntimeError):
    pass


class SelfPlayResumeError(SelfPlayError):
    pass


class GracefulStop(SelfPlayError):
    pass


class SamplerFailureMode(StrEnum):
    """Serialized failure handling; converted to scheduler mode lazily."""

    STRICT = "strict"
    HEURISTIC = "heuristic"


@dataclass(frozen=True, slots=True)
class SeatPolicy:
    """An immutable seat policy reference: baseline descriptor or frozen learned descriptor."""

    policy_id: str
    kind: Literal["baseline", "learned"]
    descriptor: AgentDescriptor | None = None
    learned: PolicyDescriptor | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"baseline", "learned"}:
            raise ValueError("unknown seat policy kind")
        if (
            not self.policy_id
            or (self.kind == "baseline") != (self.descriptor is not None)
            or (self.kind == "learned") != (self.learned is not None)
        ):
            raise ValueError("invalid immutable seat policy")
        if self.descriptor is not None and self.policy_id != self.descriptor.descriptor_id:
            raise ValueError("baseline policy ID differs from descriptor")
        if self.learned is not None and self.policy_id != self.learned.policy_id:
            raise ValueError("learned policy ID differs from descriptor")


@dataclass(frozen=True, slots=True)
class EpisodeAssignment:
    episode_id: str
    setup_seed: int
    seat_policies: tuple[SeatPolicy, SeatPolicy]

    def __post_init__(self) -> None:
        if (
            not self.episode_id
            or isinstance(self.setup_seed, bool)
            or not isinstance(self.setup_seed, int)
            or self.setup_seed < 0
            or len(self.seat_policies) != 2
        ):
            raise ValueError("invalid episode assignment")


@dataclass(frozen=True, slots=True)
class ShardPlan:
    shard_id: str
    episode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.shard_id
            or not self.episode_ids
            or tuple(sorted(self.episode_ids)) != self.episode_ids
            or len(set(self.episode_ids)) != len(self.episode_ids)
        ):
            raise ValueError("invalid shard plan")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    run_id: str
    run_seed: int
    generation: int
    max_games_in_flight: int = 32
    shard_episode_limit: int = 256
    action_ceiling: int = 10_000
    validation_level: str = "cheap"
    sampler_failure_mode: SamplerFailureMode = SamplerFailureMode.STRICT

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or isinstance(self.run_seed, bool)
            or not isinstance(self.run_seed, int)
            or self.generation < 0
            or self.max_games_in_flight < 1
            or self.shard_episode_limit < 1
            or self.action_ceiling < 1
            or not self.validation_level
        ):
            raise ValueError("invalid generation configuration")


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    config: GenerationConfig
    policy_pool: tuple[SeatPolicy, ...]
    assignments: tuple[EpisodeAssignment, ...]
    shards: tuple[ShardPlan, ...]
    format: str = SELF_PLAY_FORMAT
    schema_version: int = SELF_PLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != SELF_PLAY_FORMAT or self.schema_version != SELF_PLAY_SCHEMA_VERSION:
            raise ValueError("unsupported generation manifest")
        ids = tuple(a.episode_id for a in self.assignments)
        if not ids:
            raise ValueError("generation manifest must have assignments")
        if len(set(ids)) != len(ids) or tuple(sorted(ids)) != ids:
            raise ValueError("assignments must have unique canonical episode IDs")
        if len({p.policy_id for p in self.policy_pool}) != len(self.policy_pool):
            raise ValueError("policy pool repeats policy IDs")
        planned = tuple(e for s in self.shards for e in s.episode_ids)
        if tuple(sorted(planned)) != ids:
            raise ValueError("shards must exactly cover assignments")
        if len({s.shard_id for s in self.shards}) != len(self.shards):
            raise ValueError("shard IDs repeat")
        pool = {p.policy_id for p in self.policy_pool}
        if any(p.policy_id not in pool for a in self.assignments for p in a.seat_policies):
            raise ValueError("assignment policy is absent from pool")

    def digest(self) -> str:
        return sha256(_json(self.payload()).encode()).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "config": _config_payload(self.config),
            "policy_pool": [_policy_payload(p) for p in self.policy_pool],
            "assignments": [
                {
                    "episode_id": a.episode_id,
                    "setup_seed": a.setup_seed,
                    "seat_policies": [_policy_payload(p) for p in a.seat_policies],
                }
                for a in self.assignments
            ],
            "shards": [
                {"shard_id": s.shard_id, "episode_ids": list(s.episode_ids)} for s in self.shards
            ],
        }


def _load_policy_descriptor(payload: object) -> PolicyDescriptor:
    from innovation_ai.training.checkpoint import PolicyDescriptor

    return PolicyDescriptor.from_payload(payload)


def _json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _policy_payload(p: SeatPolicy) -> dict[str, object]:
    return {
        "policy_id": p.policy_id,
        "kind": p.kind,
        "descriptor": None if p.descriptor is None else p.descriptor.payload(),
        "learned": None if p.learned is None else p.learned.payload(),
    }


def _config_payload(c: GenerationConfig) -> dict[str, object]:
    return {
        "run_id": c.run_id,
        "run_seed": c.run_seed,
        "generation": c.generation,
        "max_games_in_flight": c.max_games_in_flight,
        "shard_episode_limit": c.shard_episode_limit,
        "action_ceiling": c.action_ceiling,
        "validation_level": c.validation_level,
        "sampler_failure_mode": c.sampler_failure_mode.value,
    }


def save_manifest(path: Path, manifest: GenerationManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic(path, _json(manifest.payload()) + "\n")


def load_manifest(path: Path) -> GenerationManifest:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SelfPlayResumeError(f"invalid manifest: {e}") from e
    try:
        c = raw["config"]
        config = GenerationConfig(
            c["run_id"],
            c["run_seed"],
            c["generation"],
            c["max_games_in_flight"],
            c["shard_episode_limit"],
            c["action_ceiling"],
            c["validation_level"],
            SamplerFailureMode(c["sampler_failure_mode"]),
        )

        def policy(x: object) -> SeatPolicy:
            assert isinstance(x, dict)
            return SeatPolicy(
                x["policy_id"],
                x["kind"],
                None
                if x["descriptor"] is None
                else AgentDescriptor(
                    x["descriptor"]["name"],
                    x["descriptor"]["version"],
                    tuple(sorted(x["descriptor"]["parameters"].items())),
                ),
                None if x["learned"] is None else _load_policy_descriptor(x["learned"]),
            )

        result = GenerationManifest(
            config,
            tuple(policy(x) for x in raw["policy_pool"]),
            tuple(
                EpisodeAssignment(
                    x["episode_id"],
                    x["setup_seed"],
                    (policy(x["seat_policies"][0]), policy(x["seat_policies"][1])),
                )
                for x in raw["assignments"]
            ),
            tuple(ShardPlan(x["shard_id"], tuple(x["episode_ids"])) for x in raw["shards"]),
            raw["format"],
            raw["schema_version"],
        )
    except (KeyError, TypeError, ValueError, AssertionError) as e:
        raise SelfPlayResumeError(f"invalid manifest schema: {e}") from e
    if _json(result.payload()) != _json(raw):
        raise SelfPlayResumeError("manifest is not a strict canonical schema")
    return result


def plan_generation(
    config: GenerationConfig,
    policies: Iterable[SeatPolicy],
    seat_pairs: Iterable[tuple[str, str]],
    episode_count: int,
) -> GenerationManifest:
    pool = tuple(sorted(policies, key=lambda p: p.policy_id))
    by_id = {p.policy_id: p for p in pool}
    pairs = tuple(seat_pairs)
    if episode_count < 1 or not pairs:
        raise ValueError("episode count and seat assignments must be non-empty")
    assignments = tuple(
        EpisodeAssignment(
            f"episode-{i:06d}",
            setup_seed(config.run_seed, f"episode-{i:06d}"),
            (by_id[pairs[i % len(pairs)][0]], by_id[pairs[i % len(pairs)][1]]),
        )
        for i in range(episode_count)
    )
    shards = tuple(
        ShardPlan(
            f"shard-{i // config.shard_episode_limit:05d}",
            tuple(a.episode_id for a in assignments[i : i + config.shard_episode_limit]),
        )
        for i in range(0, episode_count, config.shard_episode_limit)
    )
    return GenerationManifest(config, pool, assignments, shards)


class _RecorderSink(SemanticActionSink):
    def __init__(self, make: Callable[[str], CompactReplayRecorder], action_ceiling: int):
        self.make = make
        self.action_ceiling = action_ceiling
        self.recorders: dict[str, CompactReplayRecorder] = {}
        self.action_counts: dict[str, int] = {}

    def recorder(self, episode_id: str) -> CompactReplayRecorder:
        return self.recorders.setdefault(episode_id, self.make(episode_id))

    def record_action(self, event: SemanticActionEvent) -> None:
        count = self.action_counts.get(event.game_id, 0) + 1
        if count > self.action_ceiling:
            raise SelfPlayError(f"episode {event.game_id!r} exceeded action ceiling")
        self.action_counts[event.game_id] = count
        self.recorder(event.game_id).submit(event.action)


def _atomic(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def _provenance(
    manifest: GenerationManifest, assignment: EpisodeAssignment
) -> CompactReplayProvenance:
    seats = (
        SeatPolicyProvenance(
            PlayerId.PLAYER_1,
            assignment.seat_policies[0].policy_id,
            None
            if assignment.seat_policies[0].learned is None
            else assignment.seat_policies[0].learned.checkpoint_id,
            AGENT_RNG_VERSION,
        ),
        SeatPolicyProvenance(
            PlayerId.PLAYER_2,
            assignment.seat_policies[1].policy_id,
            None
            if assignment.seat_policies[1].learned is None
            else assignment.seat_policies[1].learned.checkpoint_id,
            AGENT_RNG_VERSION,
        ),
    )
    learned = next((p.learned for p in assignment.seat_policies if p.learned), None)
    return CompactReplayProvenance(
        manifest.config.run_id,
        sha256_digest(_json(manifest.payload())),
        manifest.config.generation,
        seats,
        ExplorationProvenance(
            "temperature-softmax-v1",
            0.0 if learned is None else learned.temperature,
            "sha256-domain-separated-v1",
        ),
        DeterminizationProvenance(
            "information-set-sampler-v1",
            "sha256-domain-separated-v1",
            0 if learned is None else learned.determinization_count,
            "simple-heuristic",
            manifest.config.sampler_failure_mode is SamplerFailureMode.STRICT,
        ),
    )


def run_generation(
    run_dir: str | Path,
    manifest: GenerationManifest,
    *,
    checkpoint_root: str | Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    registry: CardRegistry | None = None,
) -> tuple[str, ...]:
    """Run only missing whole shards. A stop is observed between shards, never mid-shard."""
    root = Path(run_dir)
    replay = root / "replays"
    replay.mkdir(parents=True, exist_ok=True)
    existing = root / "run-manifest.json"
    if existing.exists():
        prior = load_manifest(existing)
        if prior != manifest:
            raise SelfPlayResumeError("run directory has an incompatible manifest")
    else:
        save_manifest(existing, manifest)
    registry = registry or load_card_registry()
    assignments = {a.episode_id: a for a in manifest.assignments}
    sealed: list[str] = []
    for shard in manifest.shards:
        path = replay / (shard.shard_id + ".jsonl.gz")
        cm = CompactReplayShardManifest(shard.shard_id, shard.episode_ids)
        if path.exists():
            try:
                episodes = read_compact_replay_shard(path, cm, verify=True, registry=registry)
            except Exception as e:
                raise SelfPlayResumeError(
                    f"incomplete or invalid shard {shard.shard_id}: {e}"
                ) from e
            if any(
                e.provenance.producer_run_id != manifest.config.run_id
                or e.provenance.resolved_config_digest != sha256_digest(_json(manifest.payload()))
                for e in episodes
            ):
                raise SelfPlayResumeError("sealed shard provenance is incompatible")
            sealed.append(shard.shard_id)
            continue
        if stop_requested is not None and stop_requested():
            break
        _run_shard(path, cm, manifest, assignments, registry, checkpoint_root)
        sealed.append(shard.shard_id)
    return tuple(sealed)


def _run_shard(
    path: Path,
    shard: CompactReplayShardManifest,
    manifest: GenerationManifest,
    assignments: Mapping[str, EpisodeAssignment],
    registry: CardRegistry,
    checkpoint_root: str | Path | None,
) -> None:
    engine = InnovationEngineAdapter(registry)

    def make(eid: str) -> CompactReplayRecorder:
        return CompactReplayRecorder(
            engine.initial_state(assignments[eid].setup_seed),
            eid,
            _provenance(manifest, assignments[eid]),
            registry,
        )

    sink = _RecorderSink(make, manifest.config.action_ceiling)
    episodes: list[CompactEpisode] = []

    def done(result: GameResult[GameState]) -> None:
        rec = sink.recorder(result.record.game_id)
        episodes.append(rec.episode())

    pool = BoundedActorPool(
        engine,
        (GameSpec(e, assignments[e].setup_seed) for e in shard.episode_ids),
        max_games_in_flight=manifest.config.max_games_in_flight,
        on_complete=done,
        recording=RunnerRecording(False, False, sink),
    )
    # Current scheduler API is game-wide; reject mixed learned seats until its
    # per-seat API lands.
    learned = {
        e: next((p.learned for p in assignments[e].seat_policies if p.learned), None)
        for e in shard.episode_ids
    }
    if any(
        (a.seat_policies[0].learned is None) != (a.seat_policies[1].learned is None)
        or (
            a.seat_policies[0].learned is not None
            and a.seat_policies[1].learned is not None
            and a.seat_policies[0].learned.policy_id != a.seat_policies[1].learned.policy_id
        )
        for a in (assignments[e] for e in shard.episode_ids)
    ):
        raise SelfPlayError("current scheduler requires the same learned policy in both seats")
    if any(learned.values()):
        if checkpoint_root is None:
            raise SelfPlayError("learned generation requires checkpoint_root")
        from innovation_ai.harness.policy_scheduler import LearnedPolicyAssignment, PolicyScheduler
        from innovation_ai.harness.policy_scheduler import (
            SamplerFailureMode as SchedulerFailureMode,
        )
        from innovation_ai.training.inference import FrozenEvaluatorCache

        cache = FrozenEvaluatorCache(checkpoint_root)
        desc: dict[str, PolicyDescriptor] = {
            p.policy_id: p for p in learned.values() if p is not None
        }
        if any(value is None for value in learned.values()):
            raise SelfPlayError("cannot mix learned and baseline assignments in one shard")
        learned_descriptors = {episode_id: value for episode_id, value in learned.items() if value}
        sched = PolicyScheduler(
            {
                e: LearnedPolicyAssignment(
                    desc[learned_descriptors[e].policy_id], learned_descriptors[e].policy_id
                )
                for e in shard.episode_ids
            },
            {key: cache.evaluator_for(value) for key, value in desc.items()},
            run_seed=manifest.config.run_seed,
            generation=manifest.config.generation,
            sampler_failure_mode=SchedulerFailureMode(manifest.config.sampler_failure_mode),
            registry=registry,
        )
        while not pool.is_finished:
            schedule = sched.schedule(pool._runner)
            if any(
                sink.action_counts.get(submission.game_id, 0) >= manifest.config.action_ceiling
                for submission in schedule.submissions
            ):
                raise SelfPlayError("action ceiling reached before a further submission")
            pool.submit(schedule.submissions)  # actor pool owns retirement/refill
    else:
        agents: dict[tuple[str, PlayerId], Agent] = {}
        while not pool.is_finished:
            submits = []
            for request in pool.pending():
                seat = assignments[request.game_id].seat_policies[
                    0 if request.decision.chooser is PlayerId.PLAYER_1 else 1
                ]
                key = (request.game_id, request.decision.chooser)
                agent = agents.get(key)
                if agent is None:
                    if seat.descriptor == SIMPLE_HEURISTIC_AGENT_DESCRIPTOR:
                        agent = SimpleHeuristicAgent(registry)
                    elif seat.descriptor == RANDOM_AGENT_DESCRIPTOR:
                        agent = RandomAgent(
                            agent_seed(
                                manifest.config.run_seed,
                                request.game_id,
                                request.decision.chooser.value,
                                seat.policy_id,
                            )
                        )
                    else:
                        raise SelfPlayError(f"unsupported baseline policy {seat.policy_id!r}")
                    agents[key] = agent
                submits.append(Submission(request.game_id, agent.choose_action(request.decision)))
            if any(
                sink.action_counts.get(submission.game_id, 0) >= manifest.config.action_ceiling
                for submission in submits
            ):
                raise SelfPlayError("action ceiling reached before a further submission")
            pool.submit(tuple(submits))
    write_compact_replay_shard(path, shard, episodes)


ITERATION_STATE_FORMAT = "innovation-ai-self-play-iteration-state"
ITERATION_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class IterationState:
    """Immutable iteration workflow declaration with callable execution boundaries.

    The static manifests make every episode schedule inspectable before acting. Factories let a
    just-frozen policy descriptor become the learned/candidate schedule without coupling this
    module to any train command implementation.
    """

    bootstrap_manifest: GenerationManifest
    learned_manifest: GenerationManifest
    candidate_manifest: GenerationManifest
    format: str = ITERATION_STATE_FORMAT
    schema_version: int = ITERATION_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.format != ITERATION_STATE_FORMAT
            or self.schema_version != ITERATION_STATE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported iteration state schema")

    def payload(self) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "bootstrap_manifest": self.bootstrap_manifest.payload(),
            "learned_manifest": self.learned_manifest.payload(),
            "candidate_manifest": self.candidate_manifest.payload(),
        }


GenerationFactory = Callable[["PolicyDescriptor"], GenerationManifest]


def orchestrate_iteration(
    state: IterationState,
    run_dir: str | Path,
    *,
    materialize: Callable[[Path], object],
    train: Callable[[object], PolicyDescriptor],
    checkpoint_root: str | Path,
    learned_manifest_for: GenerationFactory | None = None,
    candidate_manifest_for: GenerationFactory | None = None,
) -> PolicyDescriptor:
    """Execute bootstrap -> materialize -> train -> learned -> candidate train.

    Callables are deliberately narrow artifact boundaries. They allow an outer CLI to choose
    training and dataset implementations while this helper preserves frozen-manifest ordering.
    """

    root = Path(run_dir)
    run_generation(root / "bootstrap", state.bootstrap_manifest)
    bootstrap_policy = train(materialize(root / "bootstrap" / "replays"))
    (root / "policies").mkdir(parents=True, exist_ok=True)
    bootstrap_policy.save(root / "policies" / f"{bootstrap_policy.policy_id}.json")
    learned_manifest = (
        state.learned_manifest
        if learned_manifest_for is None
        else learned_manifest_for(bootstrap_policy)
    )
    run_generation(root / "learned", learned_manifest, checkpoint_root=checkpoint_root)
    candidate = train(materialize(root / "learned" / "replays"))
    candidate.save(root / "policies" / f"{candidate.policy_id}.json")
    candidate_manifest = (
        state.candidate_manifest
        if candidate_manifest_for is None
        else candidate_manifest_for(candidate)
    )
    run_generation(root / "candidate", candidate_manifest, checkpoint_root=checkpoint_root)
    return candidate
