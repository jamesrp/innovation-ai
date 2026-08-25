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
        TerminalReason.CARD_EFFECT,
    }
    assert len(first.steps) < 512


@pytest.mark.fuzz
def test_small_seed_batch_has_stable_golden_trace_records() -> None:
    """Golden digests pin deterministic behaviour including selected Dogma actions.

    These changed deliberately when WP5 made Dogma executable: the fuzzer now selects and fully
    resolves dogma actions for every implemented card instead of skipping them.
    """

    results = run_protocol_fuzz_seeds((0, 1, 2))

    assert tuple(result.seed for result in results) == (0, 1, 2)
    assert tuple(result.trace_digest for result in results) == (
        "sha256:e271f2eaa8c021bee3e7f176e6b5bbf870fc5d72a8388d07d4a789503ea27495",
        "sha256:07b8857c17629f359e13b059862b67989df391dadb65c61e9cd50ba9d5a600f4",
        "sha256:b2f22a691795477c5fc2b00e51cd16bfe9df32452ec2ffc619da41e8a87a80c9",
    )


@pytest.mark.fuzz
def test_fuzzing_actually_selects_and_resolves_dogma_actions() -> None:
    """WP5 gate: Dogma is no longer filtered out of the fuzzer's action set."""

    from innovation_ai.innovation.actions import DogmaAction
    from innovation_ai.innovation.effects import implemented_card_ids

    implemented = implemented_card_ids()
    selected = tuple(
        step.action
        for result in run_protocol_fuzz_seeds(range(0, 12))
        for step in result.steps
        if isinstance(step.action, DogmaAction)
    )
    assert selected, "the fuzzer must be able to take a Dogma action"
    assert all(action.card_id in implemented for action in selected)


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
