import os

import pytest

from innovation_ai.innovation.fuzz import (
    ProtocolFuzzError,
    run_protocol_fuzz,
    run_protocol_fuzz_seeds,
)
from innovation_ai.innovation.state import TerminalReason


@pytest.mark.fuzz
def test_seeded_protocol_fuzz_is_deterministic_fast_and_reaches_terminal() -> None:
    first = run_protocol_fuzz(2001)
    second = run_protocol_fuzz(2001)

    assert first == second
    assert first.trace_digest == second.trace_digest
    assert first.steps
    assert first.steps[-1].after_hash == first.final_state_hash
    assert first.terminal.reason in {
        TerminalReason.ACHIEVEMENT_VICTORY,
        TerminalReason.DRAW_BEYOND_AGE_10,
    }
    assert len(first.steps) < 512


@pytest.mark.fuzz
def test_small_seed_batch_has_stable_golden_trace_records() -> None:
    results = run_protocol_fuzz_seeds((0, 1, 2))

    assert tuple(result.seed for result in results) == (0, 1, 2)
    assert tuple(result.trace_digest for result in results) == (
        "sha256:d1848784015221babe6d20df9573ef3ff9c0d02bd8dd3b70808e7b9784c5a280",
        "sha256:7a915a0e7b23352271ee831d078083fd6c93ee62c4206324ce753710679b644a",
        "sha256:ed84f836f3d555b43af1a5eada95d40659e58912e9e405055ba2524d0c39234f",
    )


@pytest.mark.fuzz
@pytest.mark.slow
def test_large_seeded_protocol_fuzz_batch() -> None:
    """Opt-in larger run: INNOVATION_LARGE_FUZZ_SEEDS=100 pytest -m fuzz."""

    count = int(os.environ.get("INNOVATION_LARGE_FUZZ_SEEDS", "0"))
    if count < 1:
        pytest.skip("set INNOVATION_LARGE_FUZZ_SEEDS to opt into the larger deterministic run")
    results = run_protocol_fuzz_seeds(range(10_000, 10_000 + count))
    assert len(results) == count
    assert len({result.final_state_hash for result in results}) > 1


def test_protocol_fuzz_rejects_invalid_ceiling_and_reports_step_ceiling() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        run_protocol_fuzz(3, max_steps=0)
    with pytest.raises(ProtocolFuzzError, match="step ceiling"):
        run_protocol_fuzz(3, max_steps=1)
