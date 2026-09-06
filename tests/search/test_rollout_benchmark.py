from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from innovation_ai.innovation.actions import DogmaAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.zones import ValidationLevel
from innovation_ai.search.rollout_benchmark import (
    DEFAULT_ROLLOUT_CORPUS_NAMES,
    EXPERIMENTAL_ROUTE_TRANSITION_CEILING,
    FROZEN_DETERMINIZATION_COUNTS,
    GateStatus,
    RolloutFeasibilityConfig,
    RolloutFeasibilityResult,
    TimingSummary,
    _rollout_corpus,
    complete_file_digest,
    run_rollout_feasibility,
    write_public_redacted_rollout_feasibility_json,
    write_public_redacted_rollout_feasibility_markdown,
    write_trusted_private_rollout_feasibility_json,
    write_trusted_private_rollout_feasibility_markdown,
)
from innovation_ai.search.rollout_validation import (
    RolloutArenaSmokeConfig,
    RolloutGenerationSmokeConfig,
    RolloutRouteLengthRootMeasurement,
    RolloutRouteLengthSoakConfig,
    RolloutRouteLengthSoakResult,
    RolloutThroughputSmokeResult,
    run_rollout_route_length_soak,
)


def _tiny_config(
    *names: str,
    determinizations: tuple[int, ...] = (1,),
) -> RolloutFeasibilityConfig:
    return RolloutFeasibilityConfig(
        determinization_counts=determinizations,
        corpus_names=names or ("early-first-paid-turn",),
        warmup_repetitions=0,
        measurement_repetitions=1,
    )


def test_reference_defaults_reuse_eight_roots_and_add_rollout_stress_cases() -> None:
    config = RolloutFeasibilityConfig()

    assert config.determinization_counts == FROZEN_DETERMINIZATION_COUNTS == (1, 2, 4)
    assert config.route_transition_ceiling == EXPERIMENTAL_ROUTE_TRANSITION_CEILING == 1024
    ceiling_status = config.public_redacted_payload()["route_transition_ceiling_status"]
    assert isinstance(ceiling_status, str)
    assert not ceiling_status.startswith("production")
    assert config.validation_level is ValidationLevel.FULL
    assert config.strict
    assert config.warmup_repetitions == 1
    assert config.measurement_repetitions == 5
    assert DEFAULT_ROLLOUT_CORPUS_NAMES[:8] == (
        "both-pending-starting-meld",
        "one-latent-starting-meld",
        "early-first-paid-turn",
        "explicit-second-paid-action",
        "own-turn-effect-choice",
        "opponent-demand-effect-choice",
        "opponent-shared-effect-choice",
        "late-high-branching-position",
    )
    assert DEFAULT_ROLLOUT_CORPUS_NAMES[8:] == (
        "high-choice-first-paid-action",
        "nested-effect-decisions",
        "continuation-terminal",
        "no-op-no-progress-continuation",
    )
    assert config.config_digest.startswith("sha256:")


def test_expanded_corpus_exercises_high_choice_nested_terminal_and_no_progress() -> None:
    registry = load_card_registry()
    programs = load_effect_programs()
    entries = {item.name: item for item in _rollout_corpus(4404, registry, programs)[8:]}

    high_choice = entries["high-choice-first-paid-action"]
    assert high_choice.state.paid_actions_remaining == 2
    high_dogma = next(
        action
        for action in high_choice.decision.legal_actions
        if isinstance(action, DogmaAction) and action.card_id.value == "empiricism"
    )
    high_pause = apply_action(high_choice.state, high_dogma, registry, programs)
    assert high_pause.decision is not None
    assert len(high_pause.decision.legal_actions) == 10

    nested = entries["nested-effect-decisions"]
    assert "nested-card-execution" in nested.features
    assert len(nested.decision.legal_actions) >= 2

    no_progress = entries["no-op-no-progress-continuation"]
    first_dogma = next(
        action for action in no_progress.decision.legal_actions if isinstance(action, DogmaAction)
    )
    after_first = apply_action(no_progress.state, first_dogma, registry, programs).state
    second = current_decisions(after_first, registry, programs)[0]
    second_dogma = next(
        action for action in second.legal_actions if isinstance(action, DogmaAction)
    )
    after_second = apply_action(after_first, second_dogma, registry, programs).state
    assert after_first.players == no_progress.state.players
    assert after_second.players == no_progress.state.players
    assert after_second.supply == no_progress.state.supply

    report = run_rollout_feasibility(
        _tiny_config(
            "nested-effect-decisions",
            "continuation-terminal",
            "no-op-no-progress-continuation",
        )
    )
    measurements = {item.corpus_name: item for item in report.measurements}
    assert measurements["nested-effect-decisions"].rollout_steps > 0
    assert measurements["continuation-terminal"].terminal_routes >= 1
    assert measurements["no-op-no-progress-continuation"].maximum_route_transitions >= 2
    assert all(item.completed_routes == item.routes for item in report.measurements)


def test_tiny_run_reports_counters_timing_rss_variance_and_immediate_rerun() -> None:
    report = run_rollout_feasibility(_tiny_config("early-first-paid-turn", determinizations=(1, 2)))

    assert not report.failures
    assert len(report.measurements) == 2
    for item in report.measurements:
        assert item.root_actions >= 2
        assert item.routes == item.root_actions * item.determinization_count
        assert item.total_engine_transitions == (
            item.root_engine_transitions
            + item.setup_response_engine_transitions
            + item.continuation_engine_transitions
        )
        assert item.completed_routes == item.routes
        assert item.safety_ceiling_hits == 0
        assert item.maximum_route_transitions <= item.route_transition_ceiling
        assert item.timing.median_seconds >= 0
        assert item.timing.p95_seconds >= item.timing.median_seconds
        assert item.timing.max_seconds >= item.timing.p95_seconds
        assert item.median_roots_per_second >= 0
        assert item.peak_rss_bytes >= 0
        assert len(item.action_value_variances) == item.root_actions
        assert item.deterministic_run_digest == item.deterministic_rerun_digest
        assert item.deterministic_rerun_match

    assert len(report.aggregates) == 2
    diagnostic = report.agreement_diagnostics[0]
    assert tuple(count for count, _ in diagnostic.selected_actions) == (1, 2)
    assert diagnostic.action_value_variances


def test_content_digest_excludes_timing_rates_rss_and_complete_file_digest_does_not() -> None:
    report = run_rollout_feasibility(_tiny_config())
    original = report.measurements[0]
    changed = replace(
        original,
        timing=TimingSummary((original.timing.wall_seconds[0] + 10.0,)),
        median_roots_per_second=0.01,
        peak_rss_bytes=1,
    )
    rebuilt = RolloutFeasibilityResult(
        report.config,
        report.corpus_digest,
        (changed,),
        report.agreement_diagnostics,
        report.failures,
    )

    assert rebuilt.content_digest == report.content_digest
    assert rebuilt.to_trusted_private_json() != report.to_trusted_private_json()
    assert rebuilt.trusted_private_json_file_digest() != report.trusted_private_json_file_digest()


def test_explicit_private_and_public_writers_cover_bytes_and_redact_sensitive_data(
    tmp_path: Path,
) -> None:
    report = run_rollout_feasibility(_tiny_config())
    private_json = write_trusted_private_rollout_feasibility_json(
        report, tmp_path / "private" / "rollout.json"
    )
    private_markdown = write_trusted_private_rollout_feasibility_markdown(
        report, tmp_path / "private" / "rollout.md"
    )
    public_json = write_public_redacted_rollout_feasibility_json(
        report, tmp_path / "public" / "rollout.json"
    )
    public_markdown = write_public_redacted_rollout_feasibility_markdown(
        report, tmp_path / "public" / "rollout.md"
    )

    assert complete_file_digest(private_json) == report.trusted_private_json_file_digest()
    assert complete_file_digest(private_markdown) == report.trusted_private_markdown_file_digest()
    assert complete_file_digest(public_json) == report.public_redacted_json_file_digest()
    assert complete_file_digest(public_markdown) == report.public_redacted_markdown_file_digest()

    private_payload = json.loads(private_json.read_text())
    private_telemetry = private_payload["measurements"][0]["selection_telemetry"]
    assert private_payload["privacy"] == "trusted-private"
    assert "action_keys" in private_telemetry
    assert "action_sample_values" in private_telemetry
    assert "rollout_path" in private_telemetry["routes"][0]

    public_text = public_json.read_text()
    public_payload = json.loads(public_text)
    assert public_payload["privacy"] == "public-redacted"
    for forbidden in (
        "action_keys",
        "action_key",
        "action_mean_values",
        "sample_values",
        "selected_action_index",
        "tied_action_indices",
        "chooser",
        "rollout_path",
        "information_set_digest",
        "sampler_seed_digest",
        "continuation_agent_descriptor_id",
        "rollout_descriptor_id",
        str(tmp_path),
    ):
        assert forbidden not in public_text
    public_markdown_text = public_markdown.read_text()
    assert "Aggregate scaling" not in public_markdown_text
    assert "Gate evaluation" in public_markdown_text


def _passing_route_soak() -> RolloutRouteLengthSoakResult:
    config = RolloutRouteLengthSoakConfig()
    seeded = len(config.seeded_game_seeds) * len(config.seeded_decision_indices)
    cards = len(load_card_registry().cards)
    high_choice = 2
    planned = seeded + cards + high_choice
    measurements = tuple(
        RolloutRouteLengthRootMeasurement(
            f"root-{index:04d}",
            "injected",
            "turn-action",
            1,
            2,
            2,
            0,
            0,
            512,
        )
        for index in range(planned)
    )
    return RolloutRouteLengthSoakResult(
        config,
        planned,
        seeded,
        cards,
        high_choice,
        measurements,
        (),
        1.0,
        0,
    )


def _generation_smoke(rate: float) -> RolloutThroughputSmokeResult:
    config = RolloutGenerationSmokeConfig()
    committed = 200
    return RolloutThroughputSmokeResult(
        "generation",
        config,
        config.games,
        committed,
        committed / rate,
    )


def _arena_smoke(rate: float) -> RolloutThroughputSmokeResult:
    config = RolloutArenaSmokeConfig()
    committed = 200
    return RolloutThroughputSmokeResult(
        "arena",
        config,
        config.games,
        committed,
        committed / rate,
        rollout_decisions=20,
        maximum_route_transitions=8,
    )


def test_subset_core_pass_does_not_claim_overall_and_smokes_need_measurement() -> None:
    report = run_rollout_feasibility(
        _tiny_config("no-op-no-progress-continuation", determinizations=(2,))
    )
    measurement = replace(
        report.measurements[0],
        timing=TimingSummary((0.1,)),
        median_roots_per_second=10.0,
        peak_rss_bytes=0,
    )
    rebuilt = RolloutFeasibilityResult(
        report.config,
        report.corpus_digest,
        (measurement,),
        report.agreement_diagnostics,
    )
    checks = {item.name: item for item in rebuilt.gates.checks}

    assert rebuilt.gates.target_determinization_count == 2
    assert rebuilt.gates.core_search_passed
    assert not rebuilt.gates.passed
    assert rebuilt.gates.status is GateStatus.FAIL
    assert checks["complete-reference-configuration"].status is GateStatus.FAIL
    assert checks["all-card-route-length-soak"].status is GateStatus.PENDING
    assert checks["generation-smoke-throughput"].status is GateStatus.PENDING
    assert checks["arena-smoke-throughput"].status is GateStatus.PENDING
    assert checks["all-routes-complete"].requirement.startswith("100%")
    assert checks["maximum-route-length"].requirement.startswith("<= 512 transitions")
    assert checks["target-median-throughput"].requirement == ">= 2.5 roots/s"
    assert checks["target-per-root-p95"].requirement == "<= 1.0 seconds"

    smoke_failed = replace(
        rebuilt,
        route_length_soak=_passing_route_soak(),
        generation_smoke=_generation_smoke(1.9),
        arena_smoke=_arena_smoke(0.9),
    )
    failed_checks = {item.name: item for item in smoke_failed.gates.checks}
    assert failed_checks["generation-smoke-throughput"].status is GateStatus.FAIL
    assert failed_checks["arena-smoke-throughput"].status is GateStatus.FAIL

    smoke_measured = replace(
        rebuilt,
        route_length_soak=_passing_route_soak(),
        generation_smoke=_generation_smoke(2.0),
        arena_smoke=_arena_smoke(1.0),
    )
    measured_checks = {item.name: item for item in smoke_measured.gates.checks}
    assert measured_checks["generation-smoke-throughput"].status is GateStatus.PASS
    assert measured_checks["arena-smoke-throughput"].status is GateStatus.PASS
    assert not smoke_measured.gates.passed


def test_small_route_length_soak_covers_seeded_and_selected_card_roots() -> None:
    config = RolloutRouteLengthSoakConfig(
        seeded_game_seeds=(17,),
        seeded_decision_indices=(0,),
        include_all_cards=False,
        selected_card_ids=("agriculture",),
        include_high_choice_roots=False,
        determinization_count=1,
        validation_level=ValidationLevel.CHEAP,
    )

    result = run_rollout_route_length_soak(config)

    assert result.planned_roots == 2
    assert len(result.measurements) == 2
    assert not result.failures
    assert result.all_routes_complete
    assert result.safety_ceiling_hits == 0
    assert result.maximum_route_transitions < EXPERIMENTAL_ROUTE_TRANSITION_CEILING
    public = json.loads(result.to_json(public=True))
    assert "seeded_game_seeds" not in public["config"]
    assert "sampler_seed" not in public["config"]


def test_small_high_choice_soak_exercises_wide_paid_and_effect_roots() -> None:
    result = run_rollout_route_length_soak(
        RolloutRouteLengthSoakConfig(
            seeded_game_seeds=(),
            seeded_decision_indices=(),
            include_all_cards=False,
            include_high_choice_roots=True,
            determinization_count=1,
            validation_level=ValidationLevel.CHEAP,
        )
    )

    assert not result.failures
    assert result.planned_roots == 2
    by_id = {item.root_id: item for item in result.measurements}
    assert by_id["high-choice-paid-action"].root_actions == 46
    assert by_id["high-choice-empiricism-effect"].root_actions == 10
    assert result.all_routes_complete


def test_overall_reference_gate_requires_and_accepts_all_three_reference_measurements() -> None:
    sample = run_rollout_feasibility(_tiny_config()).measurements[0]
    config = RolloutFeasibilityConfig()
    measurements = tuple(
        replace(
            sample,
            corpus_name=name,
            determinization_count=determinizations,
            timing=TimingSummary((0.01,) * config.measurement_repetitions),
            median_roots_per_second=100.0,
            maximum_route_transitions=10,
            deterministic_rerun_match=True,
        )
        for name in config.corpus_names
        for determinizations in config.determinization_counts
    )
    core = RolloutFeasibilityResult(config, "sha256:" + "0" * 64, measurements, ())

    assert core.gates.core_search_passed
    assert core.gates.status is GateStatus.PENDING
    with_soak = replace(core, route_length_soak=_passing_route_soak())
    assert with_soak.gates.status is GateStatus.PENDING
    with_generation = replace(with_soak, generation_smoke=_generation_smoke(2.0))
    assert with_generation.gates.status is GateStatus.PENDING
    complete = replace(with_generation, arena_smoke=_arena_smoke(1.0))
    checks = {item.name: item for item in complete.gates.checks}

    assert complete.gates.status is GateStatus.PASS
    assert complete.gates.passed
    assert checks["all-card-route-length-soak"].status is GateStatus.PASS
    assert checks["generation-smoke-throughput"].status is GateStatus.PASS
    assert checks["arena-smoke-throughput"].status is GateStatus.PASS

    soak = complete.route_length_soak
    assert soak is not None
    over_headroom = replace(
        soak,
        measurements=(
            replace(
                soak.measurements[0],
                maximum_route_transitions=513,
            ),
            *soak.measurements[1:],
        ),
    )
    assert replace(complete, route_length_soak=over_headroom).gates.status is GateStatus.FAIL


def test_config_and_timing_reject_non_reference_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="only frozen determinization counts"):
        RolloutFeasibilityConfig(determinization_counts=(3,))
    with pytest.raises(ValueError, match="provisional 1024-transition ceiling"):
        RolloutFeasibilityConfig(route_transition_ceiling=512)
    with pytest.raises(ValueError, match="unknown rollout corpus"):
        _tiny_config("unknown-root")
    with pytest.raises(ValueError, match="non-negative"):
        RolloutFeasibilityConfig(warmup_repetitions=-1)
    with pytest.raises(ValueError, match="positive"):
        RolloutFeasibilityConfig(measurement_repetitions=0)
    with pytest.raises(ValueError, match="finite non-negative"):
        TimingSummary((-1.0,))
