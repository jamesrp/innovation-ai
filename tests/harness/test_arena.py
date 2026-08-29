"""Focused deterministic contracts for torch-free paired arena artifacts."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from innovation_ai.harness.arena import (
    ARENA_MANIFEST_SCHEMA_VERSION,
    BOOTSTRAP_RNG_VERSION,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    ArenaGameResult,
    ArenaManifest,
    ArenaResult,
    ArenaSchemaError,
    ArenaValidationError,
    BootstrapConfig,
    CandidateOutcome,
    CheckpointDescriptor,
    CheckpointPool,
    MatchPair,
    PlannedGame,
    PolicyDescriptor,
    PolicyPool,
    PoolEntry,
    arena_game_result_from_record,
    arena_manifest_digest,
    arena_manifest_payload,
    arena_report_payload,
    build_arena_report,
    checkpoint_descriptor_payload,
    checkpoint_pool_payload,
    dumps_arena_manifest,
    dumps_arena_report,
    dumps_arena_result,
    dumps_checkpoint_descriptor,
    dumps_checkpoint_pool,
    dumps_policy_descriptor,
    dumps_policy_pool,
    loads_arena_manifest,
    loads_arena_result,
    loads_checkpoint_descriptor,
    loads_checkpoint_pool,
    loads_policy_descriptor,
    loads_policy_pool,
    paired_bootstrap_interval,
    plan_match_pair,
    policy_descriptor_payload,
    policy_pool_payload,
    render_arena_report_table,
    validate_arena_result,
)
from innovation_ai.harness.records import GameRecord
from innovation_ai.innovation.state import TerminalReason, TerminalResult
from innovation_ai.innovation.types import PlayerId


def _manifest(*, candidate: str = "candidate") -> ArenaManifest:
    return ArenaManifest(
        "arena-fixture",
        candidate,
        PolicyPool("opponents", (PoolEntry("random", 3), PoolEntry("heuristic", 1))),
        (
            plan_match_pair("seed-101", 101, candidate, "random"),
            plan_match_pair("seed-202", 202, candidate, "heuristic"),
        ),
        BootstrapConfig(seed=17, resamples=101),
    )


def _terminal(winner: PlayerId | None, reason: TerminalReason) -> TerminalResult:
    return TerminalResult(reason, () if winner is None else (winner,))


def _result(manifest: ArenaManifest) -> ArenaResult:
    first, second = manifest.match_pairs
    games = (
        # random pair: win from player 1, loss from player 2 => paired score .5
        ArenaGameResult(
            first.pair_id,
            first.games[0].game_id,
            PlayerId.PLAYER_1,
            _terminal(PlayerId.PLAYER_1, TerminalReason.CARD_EFFECT),
            10,
        ),
        ArenaGameResult(
            first.pair_id,
            first.games[1].game_id,
            PlayerId.PLAYER_2,
            _terminal(PlayerId.PLAYER_1, TerminalReason.DRAW_BEYOND_AGE_10),
            20,
        ),
        # heuristic pair: draw from player 1, win from player 2 => paired score .75
        ArenaGameResult(
            second.pair_id,
            second.games[0].game_id,
            PlayerId.PLAYER_1,
            _terminal(None, TerminalReason.DRAW_BEYOND_AGE_10),
            30,
        ),
        ArenaGameResult(
            second.pair_id,
            second.games[1].game_id,
            PlayerId.PLAYER_2,
            _terminal(PlayerId.PLAYER_2, TerminalReason.ACHIEVEMENT_VICTORY),
            40,
        ),
    )
    return ArenaResult.for_manifest(manifest, games)


def test_manifest_plans_exact_seat_swaps_and_schema_round_trips_byte_identically() -> None:
    manifest = _manifest()
    pair = manifest.match_pairs[0]

    assert pair.setup_seed == 101
    assert pair.games == (
        PlannedGame("seed-101:candidate-player-1", PlayerId.PLAYER_1),
        PlannedGame("seed-101:candidate-player-2", PlayerId.PLAYER_2),
    )
    assert manifest.bootstrap.resamples == 101
    assert BootstrapConfig().resamples == DEFAULT_BOOTSTRAP_RESAMPLES
    assert BootstrapConfig().rng_version == BOOTSTRAP_RNG_VERSION
    assert arena_manifest_payload(manifest)["schema_version"] == ARENA_MANIFEST_SCHEMA_VERSION

    encoded = dumps_arena_manifest(manifest)
    assert dumps_arena_manifest(loads_arena_manifest(encoded)) == encoded
    assert arena_manifest_digest(manifest).startswith("sha256:")

    malformed = json.loads(encoded)
    malformed["unexpected"] = True
    with pytest.raises(ArenaSchemaError, match="keys differ"):
        loads_arena_manifest(json.dumps(malformed))


def test_policy_and_checkpoint_schemas_retain_ids_without_artifact_copying() -> None:
    descriptor = PolicyDescriptor("policy-a", "learned", "checkpoint-a")
    checkpoint = CheckpointDescriptor("checkpoint-a", "sha256:" + "a" * 64)
    policy_pool = PolicyPool("pool", (PoolEntry("policy-a", 2), PoolEntry("policy-b", 1)))
    checkpoint_pool = CheckpointPool("checkpoints", ("checkpoint-a", "checkpoint-b"))

    assert policy_descriptor_payload(descriptor) == {
        "schema_version": 1,
        "policy_id": "policy-a",
        "policy_kind": "learned",
        "checkpoint_id": "checkpoint-a",
    }
    assert checkpoint_descriptor_payload(checkpoint)["artifact_sha256"] == "sha256:" + "a" * 64
    assert policy_pool_payload(policy_pool)["entries"] == [
        {"policy_id": "policy-a", "weight": 2},
        {"policy_id": "policy-b", "weight": 1},
    ]
    assert checkpoint_pool_payload(checkpoint_pool)["checkpoint_ids"] == [
        "checkpoint-a",
        "checkpoint-b",
    ]
    assert policy_pool.weight_for("policy-b") == 1
    assert loads_policy_descriptor(dumps_policy_descriptor(descriptor)) == descriptor
    assert loads_checkpoint_descriptor(dumps_checkpoint_descriptor(checkpoint)) == checkpoint
    assert loads_policy_pool(dumps_policy_pool(policy_pool)) == policy_pool
    assert loads_checkpoint_pool(dumps_checkpoint_pool(checkpoint_pool)) == checkpoint_pool
    with pytest.raises(ArenaValidationError, match="not in pool"):
        policy_pool.weight_for("absent")


def test_complete_pair_validation_rejects_missing_games_and_changed_seats() -> None:
    manifest = _manifest()
    complete = _result(manifest)
    partial = ArenaResult(
        complete.arena_id,
        complete.manifest_sha256,
        complete.games[:-1],
    )
    with pytest.raises(ArenaValidationError, match="game IDs differ"):
        validate_arena_result(manifest, partial)

    changed_seat = replace(complete.games[0], candidate_seat=PlayerId.PLAYER_2)
    altered = ArenaResult(
        complete.arena_id,
        complete.manifest_sha256,
        (changed_seat, *complete.games[1:]),
    )
    with pytest.raises(ArenaValidationError, match="planned pair or seat"):
        validate_arena_result(manifest, altered)

    with pytest.raises(ArenaSchemaError, match="ordered candidate-player-1"):
        MatchPair(
            "invalid",
            3,
            "candidate",
            "opponent",
            (PlannedGame("one", PlayerId.PLAYER_2), PlannedGame("two", PlayerId.PLAYER_1)),
        )


def test_statistics_reports_wdl_seats_reasons_lengths_and_stratified_weights() -> None:
    manifest = _manifest()
    result = _result(manifest)
    report = build_arena_report(manifest, result)
    stats = report.all_pairs

    assert stats.pair_count == 2
    assert stats.wdl.wins == 2
    assert stats.wdl.draws == 1
    assert stats.wdl.losses == 1
    assert stats.mean_pair_utility == 0.625
    assert stats.seat_breakdown[0].wdl.wins == 1
    assert stats.seat_breakdown[0].wdl.draws == 1
    assert stats.seat_breakdown[1].wdl.wins == 1
    assert stats.seat_breakdown[1].wdl.losses == 1
    assert dict(stats.terminal_reasons) == {
        TerminalReason.ACHIEVEMENT_VICTORY: 1,
        TerminalReason.DRAW_BEYOND_AGE_10: 2,
        TerminalReason.CARD_EFFECT: 1,
    }
    assert stats.game_lengths.mean == 25.0
    assert (stats.game_lengths.minimum, stats.game_lengths.maximum) == (10, 40)
    assert tuple(item.opponent_policy_id for item in report.by_opponent) == ("random", "heuristic")
    assert report.by_opponent[0].statistics.mean_pair_utility == 0.5
    assert report.by_opponent[1].statistics.mean_pair_utility == 0.75
    assert report.weighted_pool.mean_pair_utility == pytest.approx(0.5625)

    report_json = dumps_arena_report(report)
    assert dumps_arena_report(report) == report_json
    assert arena_report_payload(report)["weighted_pool"]
    table = render_arena_report_table(report)
    assert "P1 W-D-L" in table
    assert "weighted pool" in table
    assert "random" in table

    encoded_result = dumps_arena_result(result)
    decoded_result = loads_arena_result(encoded_result)
    assert decoded_result == result
    validate_arena_result(manifest, decoded_result)


def test_bootstrap_is_deterministic_and_candidate_label_reversal_complements_interval() -> None:
    config = BootstrapConfig(seed=91, resamples=257)
    direct = paired_bootstrap_interval((0.0, 0.25, 0.5, 0.75, 1.0), config)
    repeated = paired_bootstrap_interval((0.0, 0.25, 0.5, 0.75, 1.0), config)
    reversed_scores = paired_bootstrap_interval((1.0, 0.75, 0.5, 0.25, 0.0), config)
    assert direct == repeated
    assert reversed_scores.lower == pytest.approx(1.0 - direct.upper)
    assert reversed_scores.upper == pytest.approx(1.0 - direct.lower)

    manifest = ArenaManifest(
        "reverse",
        "candidate",
        PolicyPool("one", (PoolEntry("opponent", 1),)),
        (
            plan_match_pair("reverse-1", 1, "candidate", "opponent"),
            plan_match_pair("reverse-2", 2, "candidate", "opponent"),
        ),
        config,
    )
    original_games = (
        ArenaGameResult(
            "reverse-1",
            manifest.match_pairs[0].games[0].game_id,
            PlayerId.PLAYER_1,
            _terminal(PlayerId.PLAYER_1, TerminalReason.CARD_EFFECT),
            4,
        ),
        ArenaGameResult(
            "reverse-1",
            manifest.match_pairs[0].games[1].game_id,
            PlayerId.PLAYER_2,
            _terminal(PlayerId.PLAYER_1, TerminalReason.CARD_EFFECT),
            5,
        ),
        ArenaGameResult(
            "reverse-2",
            manifest.match_pairs[1].games[0].game_id,
            PlayerId.PLAYER_1,
            _terminal(None, TerminalReason.CARD_EFFECT),
            6,
        ),
        ArenaGameResult(
            "reverse-2",
            manifest.match_pairs[1].games[1].game_id,
            PlayerId.PLAYER_2,
            _terminal(PlayerId.PLAYER_2, TerminalReason.CARD_EFFECT),
            7,
        ),
    )
    report = build_arena_report(manifest, ArenaResult.for_manifest(manifest, original_games))
    reverse_manifest = ArenaManifest(
        "reverse-labels",
        "opponent",
        PolicyPool("one", (PoolEntry("candidate", 1),)),
        (
            plan_match_pair("reverse-1", 1, "opponent", "candidate"),
            plan_match_pair("reverse-2", 2, "opponent", "candidate"),
        ),
        config,
    )
    reverse_games = tuple(
        ArenaGameResult(
            pair.pair_id,
            pair.games[seat_index].game_id,
            pair.games[seat_index].candidate_seat,
            original_games[pair_index * 2 + (1 - seat_index)].terminal,
            original_games[pair_index * 2 + (1 - seat_index)].game_length,
        )
        for pair_index, pair in enumerate(reverse_manifest.match_pairs)
        for seat_index in range(2)
    )
    reverse_result = ArenaResult.for_manifest(reverse_manifest, reverse_games)
    reverse_report = build_arena_report(reverse_manifest, reverse_result)
    assert reverse_report.all_pairs.mean_pair_utility == pytest.approx(
        1.0 - report.all_pairs.mean_pair_utility
    )
    assert reverse_report.all_pairs.confidence_interval.lower == pytest.approx(
        1.0 - report.all_pairs.confidence_interval.upper
    )
    assert reverse_report.all_pairs.confidence_interval.upper == pytest.approx(
        1.0 - report.all_pairs.confidence_interval.lower
    )


def test_runner_records_supply_only_terminal_and_length_for_a_planned_game() -> None:
    pair = plan_match_pair("record-pair", 77, "candidate", "opponent")
    record = GameRecord(
        pair.games[0].game_id,
        77,
        "initial",
        (),
        _terminal(PlayerId.PLAYER_1, TerminalReason.CARD_EFFECT),
        "final",
    )
    result = arena_game_result_from_record(pair, record)
    assert result.pair_id == pair.pair_id
    assert result.candidate_seat is PlayerId.PLAYER_1
    assert result.game_length == 0
    with pytest.raises(ArenaValidationError, match="setup seed"):
        arena_game_result_from_record(pair, replace(record, setup_seed=78))
    with pytest.raises(ArenaValidationError, match="not planned"):
        arena_game_result_from_record(pair, replace(record, game_id="unknown"))

    game = ArenaGameResult(
        "pair", "game", PlayerId.PLAYER_1, _terminal(None, TerminalReason.CARD_EFFECT), 0
    )
    assert game.outcome is CandidateOutcome.DRAW
    assert game.utility == 0.5
    with pytest.raises(ArenaSchemaError, match="distinct candidate"):
        plan_match_pair("pair", 1, "same", "same")
    with pytest.raises(ArenaSchemaError, match="non-negative"):
        ArenaManifest(
            "bad",
            "candidate",
            PolicyPool("pool", (PoolEntry("opponent", 1),)),
            (plan_match_pair("pair", 1, "candidate", "opponent"),),
            temperature=-0.1,
        )
    with pytest.raises(ArenaSchemaError, match="schema version"):
        PolicyDescriptor("id", "random", schema_version=999)
