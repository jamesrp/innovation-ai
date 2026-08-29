"""Torch-free immutable paired-arena schemas, statistics, and reports.

This module deliberately contains no policy execution.  It freezes the artifact boundary around
an arena so later training/inference code can run planned games and return :class:`ArenaResult`
without changing its statistical meaning.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import ceil, floor, isfinite
from typing import cast

from innovation_ai.harness.records import GameRecord
from innovation_ai.innovation.serialization import (
    JsonValue,
    canonical_json,
    parse_json,
    terminal_from_payload,
    terminal_payload,
)
from innovation_ai.innovation.state import TerminalReason, TerminalResult
from innovation_ai.innovation.types import PlayerId

POLICY_DESCRIPTOR_SCHEMA_VERSION = 1
CHECKPOINT_DESCRIPTOR_SCHEMA_VERSION = 1
POLICY_POOL_SCHEMA_VERSION = 1
CHECKPOINT_POOL_SCHEMA_VERSION = 1
MATCH_PAIR_SCHEMA_VERSION = 1
ARENA_MANIFEST_SCHEMA_VERSION = 1
ARENA_GAME_RESULT_SCHEMA_VERSION = 1
ARENA_RESULT_SCHEMA_VERSION = 1
ARENA_REPORT_SCHEMA_VERSION = 1

ARENA_MANIFEST_FORMAT = "innovation-ai-arena-manifest"
ARENA_RESULT_FORMAT = "innovation-ai-arena-result"
ARENA_REPORT_FORMAT = "innovation-ai-arena-report"
BOOTSTRAP_RNG_VERSION = "sha256-counter-v1"
BOOTSTRAP_PERCENTILE_VERSION = "inclusive-order-statistic-v1"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


class ArenaSchemaError(ValueError):
    """An arena artifact is malformed, incompatible, or internally inconsistent."""


class ArenaValidationError(ArenaSchemaError):
    """A result does not exactly satisfy a pre-game arena manifest."""


class CandidateOutcome(StrEnum):
    """A completed game's outcome from the candidate policy's perspective."""

    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"


@dataclass(frozen=True, slots=True)
class CheckpointDescriptor:
    """An immutable checkpoint identity; artifacts are referenced, never copied."""

    checkpoint_id: str
    artifact_sha256: str
    schema_version: int = CHECKPOINT_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.checkpoint_id, "checkpoint ID")
        _digest(self.artifact_sha256, "checkpoint artifact digest")
        _version(self.schema_version, CHECKPOINT_DESCRIPTOR_SCHEMA_VERSION, "checkpoint descriptor")


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    """A stable policy identity and optional immutable checkpoint reference."""

    policy_id: str
    policy_kind: str
    checkpoint_id: str | None = None
    schema_version: int = POLICY_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "policy ID")
        _identifier(self.policy_kind, "policy kind")
        if self.checkpoint_id is not None:
            _identifier(self.checkpoint_id, "policy checkpoint ID")
        _version(self.schema_version, POLICY_DESCRIPTOR_SCHEMA_VERSION, "policy descriptor")


@dataclass(frozen=True, slots=True)
class PoolEntry:
    """One policy-ID reference and its fixed positive integer sampling weight."""

    policy_id: str
    weight: int

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "pool policy ID")
        if self.weight < 1:
            raise ArenaSchemaError("pool entry weight must be positive")


@dataclass(frozen=True, slots=True)
class PolicyPool:
    """An immutable opponent pool containing references by policy ID only."""

    pool_id: str
    entries: tuple[PoolEntry, ...]
    schema_version: int = POLICY_POOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.pool_id, "pool ID")
        if not self.entries:
            raise ArenaSchemaError("policy pool cannot be empty")
        if len({entry.policy_id for entry in self.entries}) != len(self.entries):
            raise ArenaSchemaError("policy pool has duplicate policy IDs")
        _version(self.schema_version, POLICY_POOL_SCHEMA_VERSION, "policy pool")

    def weight_for(self, policy_id: str) -> int:
        """Return the predeclared weight for one referenced policy."""

        for entry in self.entries:
            if entry.policy_id == policy_id:
                return entry.weight
        raise ArenaValidationError(f"policy {policy_id!r} is not in pool {self.pool_id!r}")


@dataclass(frozen=True, slots=True)
class CheckpointPool:
    """An immutable checkpoint-ID pool, separate from policies that consume it."""

    pool_id: str
    checkpoint_ids: tuple[str, ...]
    schema_version: int = CHECKPOINT_POOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.pool_id, "checkpoint pool ID")
        if not self.checkpoint_ids:
            raise ArenaSchemaError("checkpoint pool cannot be empty")
        if len(set(self.checkpoint_ids)) != len(self.checkpoint_ids):
            raise ArenaSchemaError("checkpoint pool has duplicate checkpoint IDs")
        for checkpoint_id in self.checkpoint_ids:
            _identifier(checkpoint_id, "checkpoint pool ID")
        _version(self.schema_version, CHECKPOINT_POOL_SCHEMA_VERSION, "checkpoint pool")


@dataclass(frozen=True, slots=True)
class PlannedGame:
    """One member of a pair, identified by game ID and candidate seat."""

    game_id: str
    candidate_seat: PlayerId

    def __post_init__(self) -> None:
        _identifier(self.game_id, "game ID")


@dataclass(frozen=True, slots=True)
class MatchPair:
    """Exactly two seat-swapped games sharing one base setup seed."""

    pair_id: str
    setup_seed: int
    candidate_policy_id: str
    opponent_policy_id: str
    games: tuple[PlannedGame, PlannedGame]
    schema_version: int = MATCH_PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.pair_id, "pair ID")
        _integer(self.setup_seed, "setup seed")
        _identifier(self.candidate_policy_id, "candidate policy ID")
        _identifier(self.opponent_policy_id, "opponent policy ID")
        if self.candidate_policy_id == self.opponent_policy_id:
            raise ArenaSchemaError(
                "a match pair must name distinct candidate and opponent policies"
            )
        if self.games[0].game_id == self.games[1].game_id:
            raise ArenaSchemaError("match-pair game IDs must be distinct")
        seats = tuple(game.candidate_seat for game in self.games)
        if seats != tuple(PlayerId):
            raise ArenaSchemaError(
                "match-pair games must be ordered candidate-player-1 then candidate-player-2"
            )
        _version(self.schema_version, MATCH_PAIR_SCHEMA_VERSION, "match pair")

    @property
    def game_ids(self) -> tuple[str, str]:
        """Return the exact planned game IDs in canonical candidate-seat order."""

        return (self.games[0].game_id, self.games[1].game_id)


def plan_match_pair(
    pair_id: str,
    setup_seed: int,
    candidate_policy_id: str,
    opponent_policy_id: str,
) -> MatchPair:
    """Create the canonical, exact two-seat plan for one base setup seed."""

    return MatchPair(
        pair_id,
        setup_seed,
        candidate_policy_id,
        opponent_policy_id,
        (
            PlannedGame(f"{pair_id}:candidate-player-1", PlayerId.PLAYER_1),
            PlannedGame(f"{pair_id}:candidate-player-2", PlayerId.PLAYER_2),
        ),
    )


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Versioned deterministic percentile-bootstrap configuration."""

    seed: int = 0
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    rng_version: str = BOOTSTRAP_RNG_VERSION
    percentile_version: str = BOOTSTRAP_PERCENTILE_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ArenaSchemaError("bootstrap seed must be an integer")
        if self.resamples < 1:
            raise ArenaSchemaError("bootstrap resamples must be positive")
        if self.rng_version != BOOTSTRAP_RNG_VERSION:
            raise ArenaSchemaError(f"unsupported bootstrap RNG version {self.rng_version!r}")
        if self.percentile_version != BOOTSTRAP_PERCENTILE_VERSION:
            raise ArenaSchemaError(
                f"unsupported bootstrap percentile version {self.percentile_version!r}"
            )


DEFAULT_BOOTSTRAP_CONFIG = BootstrapConfig()


@dataclass(frozen=True, slots=True)
class ArenaManifest:
    """The complete immutable pre-game schedule and statistical contract for an arena."""

    arena_id: str
    candidate_policy_id: str
    opponent_pool: PolicyPool
    match_pairs: tuple[MatchPair, ...]
    bootstrap: BootstrapConfig = BootstrapConfig()
    temperature: float = 0.0
    format: str = ARENA_MANIFEST_FORMAT
    schema_version: int = ARENA_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.arena_id, "arena ID")
        _identifier(self.candidate_policy_id, "candidate policy ID")
        if self.format != ARENA_MANIFEST_FORMAT:
            raise ArenaSchemaError(f"unsupported arena manifest format {self.format!r}")
        _version(self.schema_version, ARENA_MANIFEST_SCHEMA_VERSION, "arena manifest")
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ArenaSchemaError("arena temperature must be finite and non-negative")
        if not self.match_pairs:
            raise ArenaSchemaError("arena manifest cannot be empty")
        pair_ids = tuple(pair.pair_id for pair in self.match_pairs)
        if len(set(pair_ids)) != len(pair_ids):
            raise ArenaSchemaError("arena manifest has duplicate pair IDs")
        game_ids = tuple(game_id for pair in self.match_pairs for game_id in pair.game_ids)
        if len(set(game_ids)) != len(game_ids):
            raise ArenaSchemaError("arena manifest has duplicate game IDs")
        opponents = {entry.policy_id for entry in self.opponent_pool.entries}
        scheduled = {pair.opponent_policy_id for pair in self.match_pairs}
        if scheduled != opponents:
            raise ArenaSchemaError("arena schedule must include every pool policy exactly by ID")
        for pair in self.match_pairs:
            if pair.candidate_policy_id != self.candidate_policy_id:
                raise ArenaSchemaError("match-pair candidate does not match arena manifest")


@dataclass(frozen=True, slots=True)
class ArenaGameResult:
    """One terminal planned game's candidate perspective and terminal metadata."""

    pair_id: str
    game_id: str
    candidate_seat: PlayerId
    terminal: TerminalResult
    game_length: int
    schema_version: int = ARENA_GAME_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.pair_id, "result pair ID")
        _identifier(self.game_id, "result game ID")
        if self.game_length < 0:
            raise ArenaSchemaError("game length cannot be negative")
        _version(self.schema_version, ARENA_GAME_RESULT_SCHEMA_VERSION, "arena game result")

    @property
    def outcome(self) -> CandidateOutcome:
        """Classify terminal winners from the candidate policy's seat."""

        if self.terminal.is_draw:
            return CandidateOutcome.DRAW
        return (
            CandidateOutcome.WIN
            if self.candidate_seat in self.terminal.winners
            else CandidateOutcome.LOSS
        )

    @property
    def utility(self) -> float:
        """Return the canonical win/draw/loss utility (1, .5, 0)."""

        return {CandidateOutcome.WIN: 1.0, CandidateOutcome.DRAW: 0.5, CandidateOutcome.LOSS: 0.0}[
            self.outcome
        ]


@dataclass(frozen=True, slots=True)
class ArenaResult:
    """Immutable raw terminal results tied to one exact manifest digest."""

    arena_id: str
    manifest_sha256: str
    games: tuple[ArenaGameResult, ...]
    format: str = ARENA_RESULT_FORMAT
    schema_version: int = ARENA_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.arena_id, "result arena ID")
        _digest(self.manifest_sha256, "result manifest digest")
        if self.format != ARENA_RESULT_FORMAT:
            raise ArenaSchemaError(f"unsupported arena result format {self.format!r}")
        _version(self.schema_version, ARENA_RESULT_SCHEMA_VERSION, "arena result")
        if len({game.game_id for game in self.games}) != len(self.games):
            raise ArenaSchemaError("arena result has duplicate game IDs")

    @classmethod
    def for_manifest(cls, manifest: ArenaManifest, games: Iterable[ArenaGameResult]) -> ArenaResult:
        """Tie raw results to ``manifest`` and reject incomplete or altered pair plans."""

        result = cls(manifest.arena_id, arena_manifest_digest(manifest), tuple(games))
        validate_arena_result(manifest, result)
        return result


@dataclass(frozen=True, slots=True)
class WdlStatistics:
    """Candidate-perspective raw W/D/L counts and utility sum."""

    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def utility_sum(self) -> float:
        return float(self.wins) + 0.5 * float(self.draws)

    @property
    def mean_utility(self) -> float:
        return self.utility_sum / self.games if self.games else 0.0


@dataclass(frozen=True, slots=True)
class SeatBreakdown:
    """Raw candidate results split by physical player seat."""

    candidate_seat: PlayerId
    wdl: WdlStatistics


@dataclass(frozen=True, slots=True)
class GameLengthStatistics:
    """Deterministic summary of submitted actions per completed game."""

    count: int
    minimum: int
    mean: float
    maximum: int


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A deterministic two-sided 95% percentile bootstrap interval."""

    level: int
    lower: float
    upper: float
    resamples: int
    rng_version: str
    percentile_version: str


@dataclass(frozen=True, slots=True)
class ArenaStatistics:
    """Paired primary utility plus raw game, seat, reason, and length summaries."""

    pair_count: int
    wdl: WdlStatistics
    mean_pair_utility: float
    confidence_interval: ConfidenceInterval
    seat_breakdown: tuple[SeatBreakdown, SeatBreakdown]
    terminal_reasons: tuple[tuple[TerminalReason, int], ...]
    game_lengths: GameLengthStatistics


@dataclass(frozen=True, slots=True)
class OpponentReport:
    """One separately reported opponent stratum from the fixed pool."""

    opponent_policy_id: str
    statistics: ArenaStatistics


@dataclass(frozen=True, slots=True)
class WeightedPoolSummary:
    """Fixed-pool-weight aggregate calculated with stratified pair resampling."""

    mean_pair_utility: float
    confidence_interval: ConfidenceInterval


@dataclass(frozen=True, slots=True)
class ArenaReport:
    """Versioned report data that has both canonical JSON and text-table renderings."""

    arena_id: str
    manifest_sha256: str
    candidate_policy_id: str
    all_pairs: ArenaStatistics
    by_opponent: tuple[OpponentReport, ...]
    weighted_pool: WeightedPoolSummary
    format: str = ARENA_REPORT_FORMAT
    schema_version: int = ARENA_REPORT_SCHEMA_VERSION


def arena_game_result_from_record(pair: MatchPair, record: GameRecord) -> ArenaGameResult:
    """Convert a runner record only when it exactly belongs to one planned pair game.

    The runner record supplies the terminal result and action-count game length; policy labels and
    seats remain exclusively manifest-owned so an executor cannot rewrite an arena schedule.
    """

    if record.setup_seed != pair.setup_seed:
        raise ArenaValidationError(
            f"record {record.game_id!r} setup seed does not match pair {pair.pair_id!r}"
        )
    for planned in pair.games:
        if record.game_id == planned.game_id:
            return ArenaGameResult(
                pair.pair_id,
                record.game_id,
                planned.candidate_seat,
                record.terminal,
                len(record.actions),
            )
    raise ArenaValidationError(
        f"record game ID {record.game_id!r} is not planned in pair {pair.pair_id!r}"
    )


def validate_arena_result(manifest: ArenaManifest, result: ArenaResult) -> None:
    """Reject any missing, duplicate, altered-seat, or altered-seed-pair result plan."""

    if result.arena_id != manifest.arena_id:
        raise ArenaValidationError("result arena ID does not match manifest")
    if result.manifest_sha256 != arena_manifest_digest(manifest):
        raise ArenaValidationError("result manifest digest does not match manifest")
    expected = {
        game.game_id: (pair.pair_id, game.candidate_seat)
        for pair in manifest.match_pairs
        for game in pair.games
    }
    actual = {game.game_id: game for game in result.games}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ArenaValidationError(
            f"arena result game IDs differ: missing={missing}, extra={extra}"
        )
    for game_id, game in actual.items():
        pair_id, seat = expected[game_id]
        if game.pair_id != pair_id or game.candidate_seat is not seat:
            raise ArenaValidationError(f"arena result changes planned pair or seat for {game_id!r}")
    by_pair: dict[str, list[ArenaGameResult]] = defaultdict(list)
    for game in result.games:
        by_pair[game.pair_id].append(game)
    for pair in manifest.match_pairs:
        completed = by_pair[pair.pair_id]
        if len(completed) != 2 or {game.candidate_seat for game in completed} != set(PlayerId):
            raise ArenaValidationError(f"pair {pair.pair_id!r} is incomplete or not seat-swapped")


def arena_manifest_digest(manifest: ArenaManifest) -> str:
    """Return the tagged SHA-256 digest of canonical manifest JSON bytes."""

    return "sha256:" + sha256(dumps_arena_manifest(manifest).encode("ascii")).hexdigest()


def paired_bootstrap_interval(
    pair_means: Iterable[float], config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG
) -> ConfidenceInterval:
    """Bootstrap pair means with a versioned SHA-256 counter stream.

    Percentiles use inclusive order statistics at indices ``floor((B-1)*.025)`` and
    ``ceil((B-1)*.975)``.  This definition deliberately preserves exact score-complement
    symmetry when the candidate and opponent labels are reversed.
    """

    values = tuple(_utility(value, "pair mean") for value in pair_means)
    if not values:
        raise ArenaValidationError("cannot bootstrap an empty pair collection")
    samples = _bootstrap_samples((values,), (1,), config)
    lower_index = floor((config.resamples - 1) * 0.025)
    upper_index = ceil((config.resamples - 1) * 0.975)
    ordered = sorted(samples)
    return ConfidenceInterval(
        95,
        ordered[lower_index],
        ordered[upper_index],
        config.resamples,
        config.rng_version,
        config.percentile_version,
    )


def build_arena_report(manifest: ArenaManifest, result: ArenaResult) -> ArenaReport:
    """Validate results and calculate separate opponent and fixed-weight pooled summaries."""

    validate_arena_result(manifest, result)
    pairs = {pair.pair_id: pair for pair in manifest.match_pairs}
    games_by_opponent: dict[str, list[ArenaGameResult]] = defaultdict(list)
    for game in result.games:
        games_by_opponent[pairs[game.pair_id].opponent_policy_id].append(game)
    reports = tuple(
        OpponentReport(
            entry.policy_id, _statistics(games_by_opponent[entry.policy_id], manifest.bootstrap)
        )
        for entry in manifest.opponent_pool.entries
    )
    grouped_means = tuple(
        _pair_means(games_by_opponent[entry.policy_id]) for entry in manifest.opponent_pool.entries
    )
    weights = tuple(entry.weight for entry in manifest.opponent_pool.entries)
    samples = _bootstrap_samples(grouped_means, weights, manifest.bootstrap)
    lower_index = floor((manifest.bootstrap.resamples - 1) * 0.025)
    upper_index = ceil((manifest.bootstrap.resamples - 1) * 0.975)
    ordered = sorted(samples)
    interval = ConfidenceInterval(
        95,
        ordered[lower_index],
        ordered[upper_index],
        manifest.bootstrap.resamples,
        manifest.bootstrap.rng_version,
        manifest.bootstrap.percentile_version,
    )
    total_weight = sum(weights)
    weighted_mean = sum(
        float(weight) * report.statistics.mean_pair_utility
        for weight, report in zip(weights, reports, strict=True)
    ) / float(total_weight)
    return ArenaReport(
        manifest.arena_id,
        arena_manifest_digest(manifest),
        manifest.candidate_policy_id,
        _statistics(result.games, manifest.bootstrap),
        reports,
        WeightedPoolSummary(weighted_mean, interval),
    )


def _statistics(games: Iterable[ArenaGameResult], config: BootstrapConfig) -> ArenaStatistics:
    completed = tuple(games)
    means = _pair_means(completed)
    outcomes = Counter(game.outcome for game in completed)
    wdl = WdlStatistics(
        outcomes[CandidateOutcome.WIN],
        outcomes[CandidateOutcome.DRAW],
        outcomes[CandidateOutcome.LOSS],
    )
    seats = tuple(
        SeatBreakdown(
            seat,
            _wdl(game for game in completed if game.candidate_seat is seat),
        )
        for seat in PlayerId
    )
    reason_counts = Counter(game.terminal.reason for game in completed)
    lengths = tuple(game.game_length for game in completed)
    if not lengths:
        raise ArenaValidationError("cannot summarize an empty game collection")
    return ArenaStatistics(
        len(means),
        wdl,
        sum(means) / len(means),
        paired_bootstrap_interval(means, config),
        cast(tuple[SeatBreakdown, SeatBreakdown], seats),
        tuple(
            (reason, reason_counts[reason]) for reason in TerminalReason if reason in reason_counts
        ),
        GameLengthStatistics(len(lengths), min(lengths), sum(lengths) / len(lengths), max(lengths)),
    )


def _wdl(games: Iterable[ArenaGameResult]) -> WdlStatistics:
    outcomes = Counter(game.outcome for game in games)
    return WdlStatistics(
        outcomes[CandidateOutcome.WIN],
        outcomes[CandidateOutcome.DRAW],
        outcomes[CandidateOutcome.LOSS],
    )


def _pair_means(games: Iterable[ArenaGameResult]) -> tuple[float, ...]:
    by_pair: dict[str, list[ArenaGameResult]] = defaultdict(list)
    for game in games:
        by_pair[game.pair_id].append(game)
    means: list[float] = []
    for pair_id in sorted(by_pair):
        pair_games = by_pair[pair_id]
        if len(pair_games) != 2 or {game.candidate_seat for game in pair_games} != set(PlayerId):
            raise ArenaValidationError(f"pair {pair_id!r} is incomplete or not seat-swapped")
        means.append(sum(game.utility for game in pair_games) / 2.0)
    return tuple(means)


def _bootstrap_samples(
    strata: tuple[tuple[float, ...], ...], weights: tuple[int, ...], config: BootstrapConfig
) -> tuple[float, ...]:
    if len(strata) != len(weights) or not strata:
        raise ArenaValidationError("bootstrap strata and weights must be non-empty and aligned")
    if any(not values for values in strata):
        raise ArenaValidationError("bootstrap strata cannot be empty")
    total_weight = sum(weights)
    if total_weight < 1:
        raise ArenaValidationError("bootstrap weights must be positive")
    rng = _Sha256Counter(config.seed, config.rng_version)
    samples: list[float] = []
    for _ in range(config.resamples):
        weighted = 0.0
        for values, weight in zip(strata, weights, strict=True):
            weighted += float(weight) * (
                sum(values[rng.randbelow(len(values))] for _ in range(len(values))) / len(values)
            )
        samples.append(weighted / float(total_weight))
    return tuple(samples)


@dataclass(slots=True)
class _Sha256Counter:
    """A tiny, reproducible rejection-sampling generator backed by SHA-256."""

    seed: int
    version: str
    counter: int = 0

    def randbelow(self, stop: int) -> int:
        if stop < 1:
            raise ValueError("random upper bound must be positive")
        bound = (1 << 64) - ((1 << 64) % stop)
        while True:
            value = int.from_bytes(self._next_bytes(), "big")
            if value < bound:
                return value % stop

    def _next_bytes(self) -> bytes:
        payload = f"{self.version}\x00{self.seed}\x00{self.counter}".encode("ascii")
        self.counter += 1
        return sha256(payload).digest()[:8]


def arena_manifest_payload(manifest: ArenaManifest) -> dict[str, object]:
    """Return the canonical versioned JSON-compatible arena-manifest payload."""

    return {
        "format": manifest.format,
        "schema_version": manifest.schema_version,
        "arena_id": manifest.arena_id,
        "candidate_policy_id": manifest.candidate_policy_id,
        "opponent_pool": policy_pool_payload(manifest.opponent_pool),
        "match_pairs": [match_pair_payload(pair) for pair in manifest.match_pairs],
        "bootstrap": _bootstrap_payload(manifest.bootstrap),
        "temperature": manifest.temperature,
    }


def match_pair_payload(pair: MatchPair) -> dict[str, object]:
    """Return one canonical versioned match-pair payload."""

    return {
        "schema_version": pair.schema_version,
        "pair_id": pair.pair_id,
        "setup_seed": pair.setup_seed,
        "candidate_policy_id": pair.candidate_policy_id,
        "opponent_policy_id": pair.opponent_policy_id,
        "games": [
            {"game_id": game.game_id, "candidate_seat": game.candidate_seat.value}
            for game in pair.games
        ],
    }


def arena_game_result_payload(game: ArenaGameResult) -> dict[str, object]:
    """Return one canonical versioned terminal game result payload."""

    return {
        "schema_version": game.schema_version,
        "pair_id": game.pair_id,
        "game_id": game.game_id,
        "candidate_seat": game.candidate_seat.value,
        "terminal": terminal_payload(game.terminal),
        "game_length": game.game_length,
    }


def arena_result_payload(result: ArenaResult) -> dict[str, object]:
    """Return the canonical versioned raw arena-result payload."""

    return {
        "format": result.format,
        "schema_version": result.schema_version,
        "arena_id": result.arena_id,
        "manifest_sha256": result.manifest_sha256,
        "games": [arena_game_result_payload(game) for game in result.games],
    }


def policy_descriptor_payload(descriptor: PolicyDescriptor) -> dict[str, object]:
    """Return one versioned policy descriptor payload."""

    return {
        "schema_version": descriptor.schema_version,
        "policy_id": descriptor.policy_id,
        "policy_kind": descriptor.policy_kind,
        "checkpoint_id": descriptor.checkpoint_id,
    }


def checkpoint_descriptor_payload(descriptor: CheckpointDescriptor) -> dict[str, object]:
    """Return one versioned checkpoint descriptor payload."""

    return {
        "schema_version": descriptor.schema_version,
        "checkpoint_id": descriptor.checkpoint_id,
        "artifact_sha256": descriptor.artifact_sha256,
    }


def policy_pool_payload(pool: PolicyPool) -> dict[str, object]:
    """Return a pool-reference payload containing policy IDs and fixed weights only."""

    return {
        "schema_version": pool.schema_version,
        "pool_id": pool.pool_id,
        "entries": [
            {"policy_id": entry.policy_id, "weight": entry.weight} for entry in pool.entries
        ],
    }


def checkpoint_pool_payload(pool: CheckpointPool) -> dict[str, object]:
    """Return a checkpoint-ID pool payload."""

    return {
        "schema_version": pool.schema_version,
        "pool_id": pool.pool_id,
        "checkpoint_ids": list(pool.checkpoint_ids),
    }


def dumps_arena_manifest(manifest: ArenaManifest) -> str:
    """Serialize a manifest to deterministic compact JSON."""

    return canonical_json(cast(JsonValue, arena_manifest_payload(manifest)))


def dumps_arena_result(result: ArenaResult) -> str:
    """Serialize raw result data to deterministic compact JSON."""

    return canonical_json(cast(JsonValue, arena_result_payload(result)))


def loads_arena_manifest(text: str) -> ArenaManifest:
    """Load a strict arena manifest schema."""

    payload = _object(parse_json(text), "arena manifest")
    _keys(
        payload,
        {
            "format",
            "schema_version",
            "arena_id",
            "candidate_policy_id",
            "opponent_pool",
            "match_pairs",
            "bootstrap",
            "temperature",
        },
        "arena manifest",
    )
    return ArenaManifest(
        _string(payload["arena_id"], "arena_id"),
        _string(payload["candidate_policy_id"], "candidate_policy_id"),
        policy_pool_from_payload(payload["opponent_pool"]),
        tuple(
            match_pair_from_payload(item) for item in _array(payload["match_pairs"], "match_pairs")
        ),
        _bootstrap_from_payload(payload["bootstrap"]),
        _number(payload["temperature"], "temperature"),
        _string(payload["format"], "format"),
        _int(payload["schema_version"], "schema_version"),
    )


def loads_arena_result(text: str) -> ArenaResult:
    """Load a strict raw arena-result schema (then validate it against its manifest separately)."""

    payload = _object(parse_json(text), "arena result")
    _keys(
        payload,
        {"format", "schema_version", "arena_id", "manifest_sha256", "games"},
        "arena result",
    )
    return ArenaResult(
        _string(payload["arena_id"], "arena_id"),
        _string(payload["manifest_sha256"], "manifest_sha256"),
        tuple(arena_game_result_from_payload(item) for item in _array(payload["games"], "games")),
        _string(payload["format"], "format"),
        _int(payload["schema_version"], "schema_version"),
    )


def match_pair_from_payload(value: object) -> MatchPair:
    """Decode one strict match-pair payload."""

    payload = _object(value, "match pair")
    _keys(
        payload,
        {
            "schema_version",
            "pair_id",
            "setup_seed",
            "candidate_policy_id",
            "opponent_policy_id",
            "games",
        },
        "match pair",
    )
    games = tuple(
        _planned_game_from_payload(item) for item in _array(payload["games"], "match pair.games")
    )
    if len(games) != 2:
        raise ArenaSchemaError("match pair must contain exactly two games")
    return MatchPair(
        _string(payload["pair_id"], "pair_id"),
        _int(payload["setup_seed"], "setup_seed"),
        _string(payload["candidate_policy_id"], "candidate_policy_id"),
        _string(payload["opponent_policy_id"], "opponent_policy_id"),
        games,
        _int(payload["schema_version"], "schema_version"),
    )


def arena_game_result_from_payload(value: object) -> ArenaGameResult:
    """Decode one strict terminal game result payload."""

    payload = _object(value, "arena game result")
    _keys(
        payload,
        {"schema_version", "pair_id", "game_id", "candidate_seat", "terminal", "game_length"},
        "arena game result",
    )
    terminal = terminal_from_payload(payload["terminal"])
    return ArenaGameResult(
        _string(payload["pair_id"], "pair_id"),
        _string(payload["game_id"], "game_id"),
        _player(payload["candidate_seat"], "candidate_seat"),
        terminal,
        _int(payload["game_length"], "game_length"),
        _int(payload["schema_version"], "schema_version"),
    )


def policy_pool_from_payload(value: object) -> PolicyPool:
    """Decode a strict policy-ID pool payload."""

    payload = _object(value, "policy pool")
    _keys(payload, {"schema_version", "pool_id", "entries"}, "policy pool")
    entries = tuple(
        _pool_entry_from_payload(item) for item in _array(payload["entries"], "entries")
    )
    return PolicyPool(
        _string(payload["pool_id"], "pool_id"),
        entries,
        _int(payload["schema_version"], "schema_version"),
    )


def policy_descriptor_from_payload(value: object) -> PolicyDescriptor:
    """Decode one strict immutable policy descriptor payload."""

    payload = _object(value, "policy descriptor")
    _keys(
        payload,
        {"schema_version", "policy_id", "policy_kind", "checkpoint_id"},
        "policy descriptor",
    )
    checkpoint_id = payload["checkpoint_id"]
    if checkpoint_id is not None:
        checkpoint_id = _string(checkpoint_id, "checkpoint_id")
    return PolicyDescriptor(
        _string(payload["policy_id"], "policy_id"),
        _string(payload["policy_kind"], "policy_kind"),
        checkpoint_id,
        _int(payload["schema_version"], "schema_version"),
    )


def checkpoint_descriptor_from_payload(value: object) -> CheckpointDescriptor:
    """Decode one strict immutable checkpoint descriptor payload."""

    payload = _object(value, "checkpoint descriptor")
    _keys(payload, {"schema_version", "checkpoint_id", "artifact_sha256"}, "checkpoint descriptor")
    return CheckpointDescriptor(
        _string(payload["checkpoint_id"], "checkpoint_id"),
        _string(payload["artifact_sha256"], "artifact_sha256"),
        _int(payload["schema_version"], "schema_version"),
    )


def checkpoint_pool_from_payload(value: object) -> CheckpointPool:
    """Decode a strict checkpoint-ID pool payload."""

    payload = _object(value, "checkpoint pool")
    _keys(payload, {"schema_version", "pool_id", "checkpoint_ids"}, "checkpoint pool")
    return CheckpointPool(
        _string(payload["pool_id"], "pool_id"),
        tuple(
            _string(item, "checkpoint_ids[]")
            for item in _array(payload["checkpoint_ids"], "checkpoint_ids")
        ),
        _int(payload["schema_version"], "schema_version"),
    )


def dumps_policy_descriptor(descriptor: PolicyDescriptor) -> str:
    """Serialize a policy descriptor to deterministic compact JSON."""

    return canonical_json(cast(JsonValue, policy_descriptor_payload(descriptor)))


def loads_policy_descriptor(text: str) -> PolicyDescriptor:
    """Load a strict policy descriptor schema."""

    return policy_descriptor_from_payload(parse_json(text))


def dumps_checkpoint_descriptor(descriptor: CheckpointDescriptor) -> str:
    """Serialize a checkpoint descriptor to deterministic compact JSON."""

    return canonical_json(cast(JsonValue, checkpoint_descriptor_payload(descriptor)))


def loads_checkpoint_descriptor(text: str) -> CheckpointDescriptor:
    """Load a strict checkpoint descriptor schema."""

    return checkpoint_descriptor_from_payload(parse_json(text))


def dumps_policy_pool(pool: PolicyPool) -> str:
    """Serialize a policy-ID pool to deterministic compact JSON."""

    return canonical_json(cast(JsonValue, policy_pool_payload(pool)))


def loads_policy_pool(text: str) -> PolicyPool:
    """Load a strict policy-ID pool schema."""

    return policy_pool_from_payload(parse_json(text))


def dumps_checkpoint_pool(pool: CheckpointPool) -> str:
    """Serialize a checkpoint-ID pool to deterministic compact JSON."""

    return canonical_json(cast(JsonValue, checkpoint_pool_payload(pool)))


def loads_checkpoint_pool(text: str) -> CheckpointPool:
    """Load a strict checkpoint-ID pool schema."""

    return checkpoint_pool_from_payload(parse_json(text))


def _bootstrap_payload(config: BootstrapConfig) -> dict[str, object]:
    return {
        "seed": config.seed,
        "resamples": config.resamples,
        "rng_version": config.rng_version,
        "percentile_version": config.percentile_version,
    }


def _bootstrap_from_payload(value: object) -> BootstrapConfig:
    payload = _object(value, "bootstrap")
    _keys(payload, {"seed", "resamples", "rng_version", "percentile_version"}, "bootstrap")
    return BootstrapConfig(
        _int(payload["seed"], "bootstrap.seed"),
        _int(payload["resamples"], "bootstrap.resamples"),
        _string(payload["rng_version"], "bootstrap.rng_version"),
        _string(payload["percentile_version"], "bootstrap.percentile_version"),
    )


def arena_report_payload(report: ArenaReport) -> dict[str, object]:
    """Return canonical JSON report data; no display-only rounding is applied."""

    return {
        "format": report.format,
        "schema_version": report.schema_version,
        "arena_id": report.arena_id,
        "manifest_sha256": report.manifest_sha256,
        "candidate_policy_id": report.candidate_policy_id,
        "all_pairs": _statistics_payload(report.all_pairs),
        "by_opponent": [
            {
                "opponent_policy_id": item.opponent_policy_id,
                "statistics": _statistics_payload(item.statistics),
            }
            for item in report.by_opponent
        ],
        "weighted_pool": {
            "mean_pair_utility": report.weighted_pool.mean_pair_utility,
            "confidence_interval": _interval_payload(report.weighted_pool.confidence_interval),
        },
    }


def dumps_arena_report(report: ArenaReport) -> str:
    """Serialize report data to deterministic compact JSON."""

    return canonical_json(cast(JsonValue, arena_report_payload(report)))


def render_arena_report_table(report: ArenaReport) -> str:
    """Render a stable human-readable table without changing report values."""

    rows = [("all-pairs", report.all_pairs)] + [
        (item.opponent_policy_id, item.statistics) for item in report.by_opponent
    ]
    lines = [
        f"Arena {report.arena_id} | candidate {report.candidate_policy_id}",
        "opponent | pairs | W-D-L | utility | 95% CI | P1 W-D-L | P2 W-D-L | length mean/min/max",
        "-" * 108,
    ]
    for opponent, stats in rows:
        p1, p2 = stats.seat_breakdown
        lines.append(
            f"{opponent} | {stats.pair_count} | {_wdl_text(stats.wdl)} | "
            f"{stats.mean_pair_utility:.3f} | "
            f"[{stats.confidence_interval.lower:.3f}, {stats.confidence_interval.upper:.3f}] | "
            f"{_wdl_text(p1.wdl)} | {_wdl_text(p2.wdl)} | "
            f"{stats.game_lengths.mean:.1f}/{stats.game_lengths.minimum}/{stats.game_lengths.maximum}"
        )
    interval = report.weighted_pool.confidence_interval
    lines.append(
        f"weighted pool | utility {report.weighted_pool.mean_pair_utility:.3f} | "
        f"95% CI [{interval.lower:.3f}, {interval.upper:.3f}]"
    )
    return "\n".join(lines) + "\n"


def _statistics_payload(stats: ArenaStatistics) -> dict[str, object]:
    return {
        "pair_count": stats.pair_count,
        "wdl": _wdl_payload(stats.wdl),
        "mean_pair_utility": stats.mean_pair_utility,
        "confidence_interval": _interval_payload(stats.confidence_interval),
        "seat_breakdown": [
            {"candidate_seat": item.candidate_seat.value, "wdl": _wdl_payload(item.wdl)}
            for item in stats.seat_breakdown
        ],
        "terminal_reasons": [
            {"reason": reason.value, "count": count} for reason, count in stats.terminal_reasons
        ],
        "game_lengths": {
            "count": stats.game_lengths.count,
            "minimum": stats.game_lengths.minimum,
            "mean": stats.game_lengths.mean,
            "maximum": stats.game_lengths.maximum,
        },
    }


def _wdl_payload(wdl: WdlStatistics) -> dict[str, int]:
    return {"wins": wdl.wins, "draws": wdl.draws, "losses": wdl.losses}


def _interval_payload(interval: ConfidenceInterval) -> dict[str, object]:
    return {
        "level": interval.level,
        "lower": interval.lower,
        "upper": interval.upper,
        "resamples": interval.resamples,
        "rng_version": interval.rng_version,
        "percentile_version": interval.percentile_version,
    }


def _wdl_text(wdl: WdlStatistics) -> str:
    return f"{wdl.wins}-{wdl.draws}-{wdl.losses}"


def _planned_game_from_payload(value: object) -> PlannedGame:
    payload = _object(value, "planned game")
    _keys(payload, {"game_id", "candidate_seat"}, "planned game")
    return PlannedGame(
        _string(payload["game_id"], "game_id"), _player(payload["candidate_seat"], "candidate_seat")
    )


def _pool_entry_from_payload(value: object) -> PoolEntry:
    payload = _object(value, "pool entry")
    _keys(payload, {"policy_id", "weight"}, "pool entry")
    return PoolEntry(_string(payload["policy_id"], "policy_id"), _int(payload["weight"], "weight"))


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArenaSchemaError(f"{path} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ArenaSchemaError(f"{path} must be an array")
    return cast(list[object], value)


def _keys(payload: Mapping[str, object], expected: set[str], path: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ArenaSchemaError(f"{path} keys differ: missing={missing}, extra={extra}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ArenaSchemaError(f"{path} must be a string")
    return value


def _int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArenaSchemaError(f"{path} must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArenaSchemaError(f"{path} must be a number")
    return float(value)


def _player(value: object, path: str) -> PlayerId:
    try:
        return PlayerId(_string(value, path))
    except ValueError as error:
        raise ArenaSchemaError(f"{path} has an unknown player") from error


def _identifier(value: str, path: str) -> None:
    if not value or value.strip() != value:
        raise ArenaSchemaError(f"{path} must be a non-empty trimmed string")


def _digest(value: str, path: str) -> None:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in value[7:])
    ):
        raise ArenaSchemaError(f"{path} must be a tagged lowercase SHA-256 digest")


def _integer(value: int, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArenaSchemaError(f"{path} must be an integer")


def _utility(value: float, path: str) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ArenaValidationError(f"{path} must be finite and in [0, 1]")
    return value


def _version(actual: int, expected: int, path: str) -> None:
    if actual != expected:
        raise ArenaSchemaError(f"unsupported {path} schema version {actual}; expected {expected}")
