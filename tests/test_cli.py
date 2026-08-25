from __future__ import annotations

import pytest

from card_game_ai.cli import main


def test_doctor_reports_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    """The starter CLI should confirm the supported training device."""
    assert main(["doctor"]) == 0
    # Avoid coupling the test to whether the optional AI extra is installed.
    output = capsys.readouterr().out
    assert "device cpu" in output
    assert "python " in output
