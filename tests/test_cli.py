from __future__ import annotations

from pathlib import Path

import pytest

from innovation_ai.cli import main


def test_doctor_reports_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    """The starter CLI should confirm the supported training device."""
    assert main(["doctor"]) == 0
    # Avoid coupling the test to whether the optional AI extra is installed.
    output = capsys.readouterr().out
    assert "device cpu" in output
    assert "python " in output


def test_play_writes_log_and_replay_verifies_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The minimal baseline CLI should produce a complete replayable log."""
    path = tmp_path / "game.json"
    assert main(["play", "--seed", "77", "--log", str(path)]) == 0
    assert path.is_file()
    assert "saved 95-transition game" in capsys.readouterr().out

    assert main(["replay", str(path)]) == 0
    assert "hash replay matched" in capsys.readouterr().out


def test_replay_cli_reports_corrupt_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI replay failures should be loud and return a nonzero status."""
    path = tmp_path / "bad.json"
    path.write_text('{"truncated":', encoding="utf-8")
    assert main(["replay", str(path)]) == 2
    assert "error: invalid JSON" in capsys.readouterr().err


def test_ml_workflow_commands_are_registered_and_encoding_is_inspectable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["inspect-encoding", "--seed", "7", "--steps", "2"]) == 0
    payload = capsys.readouterr().out
    assert '"dimension": 4690' in payload
    assert '"fingerprint": "sha256:' in payload
