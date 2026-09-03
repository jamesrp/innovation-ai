from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from innovation_ai.search.benchmark import (
    FROZEN_OPPONENT_TURN_HORIZON,
    FROZEN_ROOT_TURN_HORIZON,
    FROZEN_STARTING_MELD_HORIZON,
    SearchFeasibilityConfig,
    SearchFeasibilityResult,
    run_search_feasibility,
    write_search_feasibility_json,
    write_search_feasibility_markdown,
)


def _tiny_config(*names: str) -> SearchFeasibilityConfig:
    return SearchFeasibilityConfig(
        route_transition_budgets=(1,),
        determinization_counts=(1,),
        corpus_names=names or ("early-first-paid-turn",),
    )


def test_tiny_search_feasibility_run_has_complete_counter_schema() -> None:
    report = run_search_feasibility(_tiny_config())

    assert not report.failures
    assert len(report.measurements) == 1
    item = report.measurements[0]
    assert item.corpus_name == "early-first-paid-turn"
    assert item.root_decisions == 1
    assert item.root_actions >= 2
    assert item.routes == item.root_actions
    assert item.total_engine_transitions == (
        item.recursive_engine_transitions
        + item.root_engine_transitions
        + item.setup_engine_transitions
    )
    assert sum(part.routes for part in item.completed_depth_distribution) == item.routes
    assert item.wall_seconds >= 0
    assert item.transitions_per_second >= 0
    assert item.decisions_per_second >= 0
    assert item.peak_rss_bytes >= 0

    payload = report.payload()
    assert payload["schema_version"] == 1
    assert payload["content_digest"] == report.content_digest
    assert payload["measurements"] == [item.payload()]
    assert "state" not in json.dumps(payload)
    config = cast(dict[str, object], payload["config"])
    assert cast(dict[str, int], config["horizons"]) == {
        "root_turn": FROZEN_ROOT_TURN_HORIZON,
        "opponent_turn": FROZEN_OPPONENT_TURN_HORIZON,
        "starting_meld": FROZEN_STARTING_MELD_HORIZON,
    }


def test_setup_subset_measures_mandatory_response_transitions() -> None:
    report = run_search_feasibility(_tiny_config("both-pending-starting-meld"))
    item = report.measurements[0]

    assert item.category == "setup"
    assert item.decision_kind == "starting-meld"
    assert item.setup_engine_transitions > 0
    assert item.routes == item.root_actions


def test_content_digest_excludes_live_performance_fields() -> None:
    report = run_search_feasibility(_tiny_config())
    changed = replace(
        report.measurements[0],
        wall_seconds=report.measurements[0].wall_seconds + 99.0,
        transitions_per_second=1.0,
        decisions_per_second=2.0,
        peak_rss_bytes=3,
    )
    rebuilt = SearchFeasibilityResult(report.config, (changed,))

    assert rebuilt.content_digest == report.content_digest
    assert rebuilt.to_json() != report.to_json()


def test_json_and_markdown_writers_emit_schema(tmp_path: Path) -> None:
    report = run_search_feasibility(_tiny_config())
    json_path = write_search_feasibility_json(report, tmp_path / "search.json")
    markdown_path = write_search_feasibility_markdown(report, tmp_path / "search.md")

    assert json.loads(json_path.read_text())["content_digest"] == report.content_digest
    markdown = markdown_path.read_text()
    assert "# Milestone 4 search feasibility" in markdown
    assert "early-first-paid-turn" in markdown
    assert "depth distribution" in markdown


def test_config_rejects_unknown_or_invalid_sweep_values() -> None:
    with pytest.raises(ValueError, match="unknown search corpus"):
        _tiny_config("not-a-corpus-entry")
    with pytest.raises(ValueError, match="positive integers"):
        SearchFeasibilityConfig(route_transition_budgets=(0,))
    with pytest.raises(ValueError, match="must be unique"):
        SearchFeasibilityConfig(determinization_counts=(1, 1))
