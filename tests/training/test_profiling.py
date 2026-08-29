from __future__ import annotations

import json
import time

import pytest

from innovation_ai.harness.engine import InnovationEngineAdapter
from innovation_ai.training.profiling import (
    IntegrityConfig,
    ProfileCategory,
    ProfileConfig,
    ProfileError,
    ProfileInvocation,
    ProfileSchemaError,
    ScaleRecommendation,
    ScenarioWork,
    analyze_bottlenecks,
    encoding_scenario,
    engine_baseline_play_scenario,
    inference_scenario,
    loads_profile_config,
    loads_profile_report,
    run_profile,
)


def test_profile_runner_warms_up_sweeps_and_round_trips_canonical_json() -> None:
    calls: list[tuple[bool, int]] = []

    def encode(invocation: ProfileInvocation) -> ScenarioWork:
        # The public invocation is intentionally opaque to this synthetic speed test except for
        # the two fields needed to prove warmups are excluded and batch sweep points are resolved.
        warmup = invocation.warmup
        batch_size = invocation.batch_size
        calls.append((warmup, batch_size))
        return ScenarioWork(4 * batch_size, "positions", metrics=(("encoded", 4 * batch_size),))

    config = ProfileConfig(
        "synthetic-profile",
        ("innovation-ai", "profile", "--synthetic"),
        warmup_samples=2,
        timed_samples=3,
        batch_sizes=(1, 8),
        integrity=IntegrityConfig(correctness_spot_checks=True, strict_spot_checks=True),
        environment_overrides=(("profile_mode", "test"),),
    )
    report = run_profile(
        config,
        [encoding_scenario("encoding", encode, correctness_spot_check=lambda _: "matched full")],
    )

    assert len(report.samples) == 6
    assert calls.count((True, 1)) == 2
    assert calls.count((True, 8)) == 2
    assert calls.count((False, 1)) == 3
    assert calls.count((False, 8)) == 3
    assert report.correctness_spot_checks[0].passed
    assert report.samples[0].parameters == (
        ("batch_size", 1),
        ("determinizations", 1),
        ("games_in_flight", 1),
        ("torch_num_threads", 1),
    )
    encoded = report.to_json()
    assert encoded == json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":"))
    assert loads_profile_report(encoded).to_json() == encoded
    assert "Largest measured wall-time share" in report.to_markdown()


def test_profile_schema_rejects_unknown_or_inconsistent_fields() -> None:
    config = ProfileConfig("schema", ("profile",))
    payload = json.loads(json.dumps(config.payload()))
    payload["unknown"] = True
    with pytest.raises(ProfileSchemaError, match="extra"):
        loads_profile_config(json.dumps(payload))

    with pytest.raises(ProfileError, match="requires correctness_spot_checks"):
        IntegrityConfig(strict_spot_checks=True)


def test_bottleneck_recommendations_are_driven_by_measured_categories() -> None:
    def inference(invocation: ProfileInvocation) -> ScenarioWork:
        batch_size = invocation.batch_size
        # Fixed dispatch overhead makes the measured large batch materially more efficient.
        time.sleep(0.001)
        return ScenarioWork(batch_size * batch_size, "positions")

    report = run_profile(
        ProfileConfig("inference", ("profile",), batch_sizes=(1, 32), timed_samples=1),
        [inference_scenario("inference", inference)],
    )
    analysis = analyze_bottlenecks(report.samples)
    assert analysis.largest.category is ProfileCategory.INFERENCE
    assert analysis.recommendation is ScaleRecommendation.GPU_BATCHING


def test_real_engine_baseline_adapter_honors_games_in_flight_and_hash_flag() -> None:
    scenario = engine_baseline_play_scenario(
        "engine-baseline",
        InnovationEngineAdapter(),
        games=1,
        run_seed=17,
        max_actions_per_game=1_000,
    )
    report = run_profile(
        ProfileConfig(
            "engine",
            ("profile",),
            timed_samples=1,
            games_in_flight=(1, 2),
            integrity=IntegrityConfig(transition_hashes=False),
        ),
        [scenario],
    )

    assert len(report.samples) == 2
    assert all(sample.work_unit == "actions" and sample.work_items > 0 for sample in report.samples)
    assert {dict(sample.parameters)["games_in_flight"] for sample in report.samples} == {1, 2}
