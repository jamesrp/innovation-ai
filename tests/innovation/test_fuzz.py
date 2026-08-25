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
        "sha256:dafe98ad9e272ea4ab9f6aa0c103fed15db6632dc55953965ee46b0e0b3f5017",
        "sha256:fa7787405e3b7ea74e9d1940f97e239f79405b6a55b5acbcf63318ff8dfe6497",
        "sha256:82e662bdbd993b8cec48cb7543f99df6915bd3ff8715386b82085253330f6e06",
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
