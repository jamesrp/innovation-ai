from __future__ import annotations

import gzip
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

from innovation_ai.harness.diagnostics import (
    DIAGNOSTIC_TRACE_PRIVACY,
    AchievementChange,
    CardMovement,
    DiagnosticJsonObject,
    DiagnosticTraceError,
    DiagnosticTraceFooter,
    DiagnosticTraceHeader,
    LearnedDecisionSummary,
    NoProgressTelemetry,
    PrivateDiagnosticTraceRecorder,
    RepeatedPaidActionWindow,
    SplayChange,
    SupplyChange,
    derive_no_progress_telemetry,
    read_diagnostic_trace,
    redacted_diagnostic_summary,
    write_diagnostic_trace,
)
from innovation_ai.harness.policy import PolicySelection
from innovation_ai.harness.policy_scheduler import PolicyDecisionAudit
from innovation_ai.harness.runner import Submission
from innovation_ai.innovation.actions import Decision, DogmaAction, DrawAction, SemanticAction
from innovation_ai.innovation.protocol import apply_action, current_decision
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GameState,
    TerminalReason,
    TerminalResult,
    build_explicit_state,
    build_setup_state,
)
from innovation_ai.innovation.terminal import apply_terminal
from innovation_ai.innovation.types import (
    CardId,
    Color,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)
from innovation_ai.search.contracts import (
    RootSelectionTelemetry,
    SearchDescriptor,
    SearchRouteTelemetry,
)
from innovation_ai.search.minimax import MinimaxSelection, SearchStatistics, action_key

_DIGEST = "sha256:" + "1" * 64
_VERSIONS = {
    "policy": "policy-v1",
    "checkpoint": _DIGEST,
    "fallback": "simple-v1",
    "search": "search-v1",
    "sampler": "sampler-v1",
}


def _playing_state(seed: int = 401) -> GameState:
    state = build_setup_state(seed)
    for _ in range(2):
        decision = current_decision(state)
        assert decision is not None
        state = apply_action(state, decision.legal_actions[0]).state
    return state


def _header(
    state: GameState,
    *,
    snapshots: bool = False,
    private_debug: bool = False,
) -> DiagnosticTraceHeader:
    return DiagnosticTraceHeader.for_state(
        state,
        source_revision="abc123",
        game_id="game-001",
        setup_id="setup-001",
        manifest_digest=_DIGEST,
        config_digest=_DIGEST,
        versions=_VERSIONS,
        rng_seed_digests={"setup": _DIGEST, "policy": _DIGEST},
        authoritative_snapshots=snapshots,
        private_debug=private_debug,
    )


def _baseline_audit(decision: Decision, action: SemanticAction) -> PolicyDecisionAudit:
    # Every test constructs the production scheduler audit contract.
    return PolicyDecisionAudit(
        Submission("game-001", action),
        decision.chooser,
        None,
        "baseline",
    )


def _gzip(payload: bytes) -> bytes:
    stream = BytesIO()
    with gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    return stream.getvalue()


def _contains_forbidden(value: object) -> bool:
    forbidden = {
        "legal_actions",
        "selected_action",
        "action_keys",
        "principal_variation",
        "card_id",
        "card_ids",
        "before_snapshot",
        "after_snapshot",
    }
    if isinstance(value, dict):
        return any(key in forbidden or _contains_forbidden(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden(child) for child in value)
    return False


def test_trace_is_deterministic_fixed_gzip_jsonl_and_round_trips(tmp_path: Path) -> None:
    state = _playing_state()
    recorder = PrivateDiagnosticTraceRecorder(_header(state), state)
    for _ in range(2):
        decision = current_decision(recorder.state)
        assert decision is not None
        draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
        after = apply_action(recorder.state, draw).state
        recorder.record_step(after, decision, _baseline_audit(decision, draw))
    trace = recorder.finish("action-ceiling", failure=RuntimeError("limit reached"))

    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    assert write_diagnostic_trace(first, trace) == write_diagnostic_trace(second, trace)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    assert read_diagnostic_trace(first) == trace

    records = [json.loads(line) for line in gzip.decompress(first.read_bytes()).splitlines()]
    assert records[0]["privacy"] == DIAGNOSTIC_TRACE_PRIVACY
    assert [record["record_type"] for record in records] == ["header", "step", "step", "footer"]
    assert records[-1]["outcome"] == "action-ceiling"
    assert records[-1]["terminal_result"] is None


def test_step_retains_hashes_actors_legal_selected_handling_and_repeat_window() -> None:
    state = replace(_playing_state(402), paid_actions_remaining=2)
    recorder = PrivateDiagnosticTraceRecorder(_header(state), state)
    first_repeat = None
    for _ in range(2):
        decision = current_decision(recorder.state)
        assert decision is not None
        draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
        after = apply_action(recorder.state, draw).state
        step = recorder.record_step(after, decision, _baseline_audit(decision, draw))
        first_repeat = step.no_progress.repeated_paid_action_window

    assert step.before_state_hash.startswith("sha256:")
    assert step.after_strategic_digest.startswith("sha256:")
    assert decision is not None
    assert step.chooser is decision.chooser
    assert step.executor is decision.executor
    assert step.active_player is decision.chooser
    assert step.decision_kind is decision.kind
    assert step.paid_actions_remaining >= 1
    assert len(step.legal_actions) == len(decision.legal_actions)
    assert step.selected_action in step.legal_actions
    assert step.handling == "baseline"
    assert first_repeat is not None
    assert first_repeat.repeated
    assert first_repeat.matching_prior_count == 1


def test_learned_summary_is_complete_and_checked_against_policy_audit() -> None:
    state = _playing_state(403)
    decision = current_decision(state)
    assert decision is not None
    selected_index = 0
    action = decision.legal_actions[selected_index]
    sample_values = tuple(
        (0.8 - index / 100, 0.6 - index / 100) for index in range(len(decision.legal_actions))
    )
    summary = LearnedDecisionSummary.from_values(sample_values, None, selected_index)
    selection = PolicySelection(
        "learned-policy",
        "game-001",
        decision.decision_id,
        action,
        summary.mean_values[selected_index],
        0.0,
    )
    audit = PolicyDecisionAudit(
        Submission("game-001", action),
        decision.chooser,
        selection,
        "learned",
    )
    after = apply_action(state, action).state
    recorder = PrivateDiagnosticTraceRecorder(_header(state), state)
    step = recorder.record_step(after, decision, audit, learned=summary)

    assert step.learned is not None
    assert step.learned["sample_values"] == [list(values) for values in sample_values]
    assert step.learned["mean_values"] == list(summary.mean_values)
    assert step.learned["selector_scores"] == list(summary.selector_scores)
    assert step.learned["selected_action_index"] == selected_index
    assert "tied_action_indices" in step.learned
    assert "margin" in step.learned

    with pytest.raises(DiagnosticTraceError, match="complete learned"):
        PrivateDiagnosticTraceRecorder(_header(state), state).record_step(after, decision, audit)


def test_search_audit_retains_root_route_and_statistics_telemetry() -> None:
    state = _playing_state(404)
    decision = current_decision(state)
    assert decision is not None
    action = decision.legal_actions[0]
    descriptor = SearchDescriptor()
    means = tuple(0.5 for _ in decision.legal_actions)
    telemetry = RootSelectionTelemetry(
        descriptor.descriptor_id,
        tuple(action_key(item) for item in decision.legal_actions),
        means,
        0,
        (),
        tuple(range(len(decision.legal_actions))),
        _DIGEST,
    )
    statistics = SearchStatistics(0, 7, 6, 1, 0, 2, 3, 4, 5)
    audit = PolicyDecisionAudit(
        Submission("game-001", action),
        decision.chooser,
        None,
        "search",
        search_selection=MinimaxSelection(action, telemetry, statistics),
    )
    step = PrivateDiagnosticTraceRecorder(_header(state), state).record_step(
        apply_action(state, action).state,
        decision,
        audit,
    )

    assert step.search is not None
    root = step.search["telemetry"]
    assert isinstance(root, dict)
    assert root["action_mean_values"] == list(means)
    assert root["selected_action_key"] == action_key(action)
    statistics_payload = step.search["statistics"]
    assert isinstance(statistics_payload, dict)
    assert statistics_payload["nodes"] == 7


def test_no_progress_telemetry_detects_noop_dogma_and_zone_changes() -> None:
    machinery = CardId.from_name("Machinery")
    state = build_explicit_state(
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(board=((Color.YELLOW, (machinery,)),)),
            ),
        ),
    )
    decision = current_decision(state)
    assert decision is not None
    dogma = next(action for action in decision.legal_actions if isinstance(action, DogmaAction))

    no_op = derive_no_progress_telemetry(state, state, decision, dogma)
    assert no_op.no_op_dogma
    assert no_op.card_movements == ()
    assert no_op.splay_changes == ()

    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
    after = apply_action(state, draw).state
    movement = derive_no_progress_telemetry(state, after, decision, draw)
    assert len(movement.card_movements) == 1
    assert movement.supply_changes
    assert movement.no_op_dogma is False


def test_authoritative_snapshots_require_two_explicit_markers_and_verify(
    tmp_path: Path,
) -> None:
    state = _playing_state(405)
    with pytest.raises(ValueError, match="private-debug"):
        _header(state, snapshots=True)

    header = _header(state, snapshots=True, private_debug=True)
    recorder = PrivateDiagnosticTraceRecorder(header, state)
    decision = current_decision(state)
    assert decision is not None
    action = decision.legal_actions[0]
    recorder.record_step(
        apply_action(state, action).state,
        decision,
        _baseline_audit(decision, action),
    )
    trace = recorder.finish("stopped")
    path = tmp_path / "snapshots.jsonl.gz"
    write_diagnostic_trace(path, trace)
    loaded = read_diagnostic_trace(path)
    assert loaded == trace
    assert loaded.steps[0].before_snapshot is not None
    assert loaded.steps[0].after_snapshot is not None


def test_strict_reader_rejects_noncanonical_unknown_and_concatenated_members(
    tmp_path: Path,
) -> None:
    state = _playing_state(406)
    trace = PrivateDiagnosticTraceRecorder(_header(state), state).finish("stopped")
    path = tmp_path / "trace.jsonl.gz"
    write_diagnostic_trace(path, trace)
    raw = gzip.decompress(path.read_bytes())

    noncanonical = tmp_path / "noncanonical.gz"
    records = [json.loads(line) for line in raw.splitlines()]
    pretty = "\n".join(json.dumps(record) for record in records).encode("ascii") + b"\n"
    noncanonical.write_bytes(_gzip(pretty))
    with pytest.raises(DiagnosticTraceError, match="canonical"):
        read_diagnostic_trace(noncanonical)

    unknown = tmp_path / "unknown.gz"
    records[0]["unexpected"] = True
    altered = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
    ).encode("ascii")
    unknown.write_bytes(_gzip(altered))
    with pytest.raises(DiagnosticTraceError, match="keys differ"):
        read_diagnostic_trace(unknown)

    concatenated = tmp_path / "concatenated.gz"
    concatenated.write_bytes(path.read_bytes() + path.read_bytes())
    with pytest.raises(DiagnosticTraceError, match="exactly one gzip member"):
        read_diagnostic_trace(concatenated)


def test_redacted_summary_excludes_private_action_card_and_snapshot_fields() -> None:
    state = _playing_state(407)
    header = _header(state, snapshots=True, private_debug=True)
    recorder = PrivateDiagnosticTraceRecorder(header, state)
    decision = current_decision(state)
    assert decision is not None
    action = decision.legal_actions[0]
    recorder.record_step(
        apply_action(state, action).state,
        decision,
        _baseline_audit(decision, action),
    )
    summary = redacted_diagnostic_summary(recorder.finish("stopped"))

    assert not _contains_forbidden(summary)
    encoded = json.dumps(summary)
    for private_card in state.players[0].hand:
        assert private_card.value not in encoded
    assert "setup_seed" not in summary
    assert "terminal_result" not in summary["footer"]  # type: ignore[operator]


def test_footer_seals_chain_and_aggregates_without_turning_failure_into_draw() -> None:
    state = _playing_state(408)
    recorder = PrivateDiagnosticTraceRecorder(_header(state), state)
    decision = current_decision(state)
    assert decision is not None
    draw = next(action for action in decision.legal_actions if isinstance(action, DrawAction))
    after = apply_action(state, draw).state
    recorder.record_step(after, decision, _baseline_audit(decision, draw))
    trace = recorder.finish("invariant-failure", failure=AssertionError("broken"))

    assert trace.footer.final_state_hash == trace.steps[-1].after_state_hash
    assert trace.footer.final_strategic_digest == trace.steps[-1].after_strategic_digest
    assert trace.footer.step_count == 1
    assert trace.footer.failure == {"type": "AssertionError", "message": "broken"}
    assert trace.footer.terminal_result is None
    assert trace.footer.records_digest.startswith("sha256:")
    assert trace.footer.no_progress_totals["card_movements"] == 1


def test_rich_learned_and_search_trace_round_trips_and_redacts(tmp_path: Path) -> None:
    state = replace(_playing_state(410), paid_actions_remaining=2)
    recorder = PrivateDiagnosticTraceRecorder(_header(state), state)

    learned_decision = current_decision(state)
    assert learned_decision is not None
    learned_action = learned_decision.legal_actions[0]
    sample_values = tuple(
        (0.4 + index / 100, 0.6 + index / 100)
        for index in range(len(learned_decision.legal_actions))
    )
    mean_values = tuple(sum(values) / len(values) for values in sample_values)
    learned_selection = PolicySelection(
        "learned-policy",
        "game-001",
        learned_decision.decision_id,
        learned_action,
        mean_values[0],
        0.0,
        "temperature-softmax-v1",
        sample_values,
        mean_values,
        mean_values,
        0,
        (len(mean_values) - 1,),
        mean_values[-1] - mean_values[-2] if len(mean_values) > 1 else None,
        _DIGEST,
        None,
    )
    learned_audit = PolicyDecisionAudit(
        Submission("game-001", learned_action),
        learned_decision.chooser,
        learned_selection,
        "learned",
    )
    after_learned = apply_action(state, learned_action).state
    recorder.record_step(after_learned, learned_decision, learned_audit)

    search_decision = current_decision(after_learned)
    assert search_decision is not None
    search_action = search_decision.legal_actions[0]
    descriptor = SearchDescriptor()
    keys = tuple(action_key(item) for item in search_decision.legal_actions)
    means = tuple(0.25 for _ in keys)
    route = SearchRouteTelemetry(
        0,
        0,
        0.25,
        1,
        2,
        3,
        1,
        1,
        True,
        False,
        (keys[0],),
    )
    telemetry = RootSelectionTelemetry(
        descriptor.descriptor_id,
        keys,
        means,
        0,
        (route,),
        tuple(range(len(keys))),
        _DIGEST,
    )
    statistics = SearchStatistics(1, 2, 3, 1, 0, 1, 1, 1, 0)
    search_audit = PolicyDecisionAudit(
        Submission("game-001", search_action),
        search_decision.chooser,
        None,
        "search",
        search_selection=MinimaxSelection(search_action, telemetry, statistics),
    )
    recorder.record_step(
        apply_action(after_learned, search_action).state,
        search_decision,
        search_audit,
    )
    trace = recorder.finish("stopped")
    path = tmp_path / "rich.jsonl.gz"
    write_diagnostic_trace(path, trace)

    loaded = read_diagnostic_trace(path)
    assert loaded == trace
    assert loaded.steps[0].learned is not None
    assert loaded.steps[1].search is not None
    redacted = redacted_diagnostic_summary(loaded)
    first = redacted["steps"][0]  # type: ignore[index]
    second = redacted["steps"][1]  # type: ignore[index]
    assert isinstance(first, dict) and first["learned_aggregates"] is not None
    assert isinstance(second, dict) and second["search_aggregates"] is not None


def test_terminal_and_snapshot_trace_round_trip(tmp_path: Path) -> None:
    terminal = TerminalResult(TerminalReason.CARD_EFFECT, (PlayerId.PLAYER_1,))
    state = apply_terminal(build_explicit_state(), terminal)
    recorder = PrivateDiagnosticTraceRecorder(
        _header(state, snapshots=True, private_debug=True), state
    )
    trace = recorder.finish("card-effect", terminal_result=terminal)
    path = tmp_path / "terminal.jsonl.gz"
    write_diagnostic_trace(path, trace)

    loaded = read_diagnostic_trace(path)
    assert loaded.footer.terminal_result is not None
    assert loaded.footer.failure is None
    assert loaded.steps == ()


def test_no_progress_tracks_splay_and_special_achievement_changes() -> None:
    state = build_explicit_state(
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    board=((Color.RED, (CardId("archery"), CardId("metalworking"))),),
                ),
            ),
        )
    )
    player = state.player(PlayerId.PLAYER_1)
    red = player.board.stack(Color.RED)
    changed_board = player.board.replace_stack(replace(red, splay=SplayDirection.LEFT))
    changed_player = replace(
        player,
        board=changed_board,
        special_achievements=(SpecialAchievementId.EMPIRE,),
    )
    after = state.replace_player(changed_player)
    decision = current_decision(state)
    assert decision is not None
    action = decision.legal_actions[0]

    progress = derive_no_progress_telemetry(state, after, decision, action)

    assert len(progress.splay_changes) == 1
    assert progress.splay_changes[0].after == SplayDirection.LEFT.value
    assert len(progress.achievements) == 1
    assert progress.achievements[0].achievement_id == SpecialAchievementId.EMPIRE.value


def test_diagnostic_value_contracts_reject_malformed_data() -> None:
    state = _playing_state(411)
    header = _header(state)
    with pytest.raises(ValueError, match="source revision"):
        replace(header, source_revision="")
    with pytest.raises(ValueError, match="setup seed"):
        replace(header, setup_seed=True)
    with pytest.raises(ValueError, match="manifest digest"):
        replace(header, manifest_digest="bad")
    with pytest.raises(ValueError, match="version values"):
        replace(header, versions={**_VERSIONS, "policy": ""})
    with pytest.raises(ValueError, match="labels"):
        replace(header, rng_seed_digests={"": _DIGEST})
    with pytest.raises(ValueError, match="RNG seed digest"):
        replace(header, rng_seed_digests={"policy": "bad"})

    valid_summary = LearnedDecisionSummary(((0.5,),), (0.5,), (0.5,), 0, (0,), 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        LearnedDecisionSummary((), (), (), 0, (), 0.0)
    with pytest.raises(ValueError, match="per-determinization"):
        LearnedDecisionSummary(((),), (0.5,), (0.5,), 0, (0,), 0.0)
    with pytest.raises(ValueError, match="equal determinization"):
        LearnedDecisionSummary(((0.5,), (0.4, 0.6)), (0.5, 0.5), (0.5, 0.5), 0, (0,), 0.0)
    with pytest.raises(ValueError, match="finite"):
        replace(valid_summary, margin=float("nan"))
    with pytest.raises(ValueError, match="selected action index"):
        replace(valid_summary, selected_action_index=1)
    with pytest.raises(ValueError, match="tied action indices"):
        replace(valid_summary, tied_action_indices=(0, 0))
    with pytest.raises(ValueError, match="margin"):
        replace(valid_summary, margin=-0.1)
    with pytest.raises(ValueError, match="mean"):
        replace(valid_summary, mean_values=(0.4,))

    with pytest.raises(ValueError, match="change direction"):
        SplayChange(PlayerId.PLAYER_1, Color.RED, "none", "none")
    with pytest.raises(ValueError, match="achievement kind"):
        AchievementChange(PlayerId.PLAYER_1, "empire", "other")
    with pytest.raises(ValueError, match="age"):
        SupplyChange(0, 1, 0, None, None)
    with pytest.raises(ValueError, match="counts"):
        SupplyChange(1, -1, 0, None, None)
    with pytest.raises(ValueError, match="must alter"):
        SupplyChange(1, 1, 1, None, None)

    pattern: DiagnosticJsonObject = {"kind": "draw"}
    with pytest.raises(ValueError, match="invalid size"):
        RepeatedPaidActionWindow(0, (pattern,), 0, False)
    with pytest.raises(ValueError, match="match count"):
        RepeatedPaidActionWindow(4, (pattern,), 1, True)
    with pytest.raises(ValueError, match="flag"):
        RepeatedPaidActionWindow(4, (pattern,), 0, True)
    with pytest.raises(ValueError, match="omit"):
        RepeatedPaidActionWindow(
            4,
            ({"kind": "draw", "decision_id": 1},),
            0,
            False,
        )

    movement = CardMovement(CardId("agriculture"), "player-1.hand", "player-1.score")
    with pytest.raises(ValueError, match="non-negative"):
        NoProgressTelemetry((movement,), (), (), (), -1, 0, 0, 0, False, None)
    with pytest.raises(ValueError, match="repeat a card"):
        NoProgressTelemetry((movement, movement), (), (), (), 0, 0, 0, 0, False, None)

    footer = DiagnosticTraceFooter(
        0,
        _DIGEST,
        _DIGEST,
        "stopped",
        None,
        None,
        {
            key: 0
            for key in (
                "card_movements",
                "splay_changes",
                "achievements",
                "supply_changes",
                "scored",
                "melded",
                "tucked",
                "returned",
                "no_op_dogmas",
                "repeated_paid_actions",
            )
        },
        _DIGEST,
    )
    with pytest.raises(ValueError, match="step count"):
        replace(footer, step_count=-1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(footer, terminal_result={"reason": "x"}, failure={"type": "x"})
    with pytest.raises(ValueError, match="totals"):
        replace(footer, no_progress_totals={})


def test_header_rejects_missing_version_provenance() -> None:
    state = _playing_state(409)
    with pytest.raises(ValueError, match="versions"):
        replace(_header(state), versions={"policy": "only-one"})
