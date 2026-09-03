from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from innovation_ai.search import (
    PRODUCTION_SEARCH_DESCRIPTOR,
    RootSelectionTelemetry,
    SearchDescriptor,
    SearchRouteTelemetry,
)


def test_search_descriptor_is_immutable_content_addressed_and_round_trips() -> None:
    descriptor = SearchDescriptor()

    assert descriptor.root_turn_horizon == 1
    assert descriptor.opponent_turn_horizon == 1
    assert descriptor.starting_meld_horizon == 1
    assert descriptor.determinization_count == 1
    assert descriptor.route_transition_budget == 400
    assert descriptor.descriptor_id.startswith("sha256:")
    assert SearchDescriptor.loads(descriptor.dumps()) == descriptor
    assert json.loads(descriptor.dumps()) == descriptor.payload()
    assert descriptor == PRODUCTION_SEARCH_DESCRIPTOR

    with pytest.raises(FrozenInstanceError):
        descriptor.route_transition_budget = 1  # type: ignore[misc]


def test_every_frozen_plan_field_changes_search_identity() -> None:
    baseline = SearchDescriptor()
    for field_name, replacement in (
        ("search_version", "search-v2"),
        ("evaluator_version", "evaluator-v2"),
        ("information_set_spec_version", "spec-v2"),
        ("sampling_algorithm", "sampling-v2"),
        ("hidden_allocation_algorithm", "allocation-v2"),
        ("sampler_seed_derivation", "sampler-seed-v2"),
        ("selector_seed_derivation", "selector-seed-v2"),
        ("root_turn_horizon", 5),
        ("opponent_turn_horizon", 4),
        ("starting_meld_horizon", 5),
        ("determinization_count", 2),
        ("sample_aggregation", "median-v1"),
        ("route_transition_budget", 401),
        ("budget_accounting", "nodes-v1"),
        ("iterative_deepening_completion_rule", "completion-v2"),
        ("move_ordering", "move-order-v2"),
        ("transposition_key_version", "key-v2"),
        ("transposition_cache_scope", "search-global-v1"),
        ("transposition_entry_semantics", "lower-upper-bounds-v1"),
        ("alpha_beta", "disabled-v1"),
        ("starting_meld_aggregation", "mean-min-max-v1"),
        ("root_transition_budgeting", "all-transitions-count-v1"),
        ("tie_break", "reverse-v1"),
        ("cycle_cutoff", "cycle-v2"),
        ("fallback_policy", "fallback-v2"),
    ):
        changes: Any = {field_name: replacement}
        changed = replace(baseline, **changes)
        assert changed.descriptor_id != baseline.descriptor_id, field_name


def test_descriptor_rejects_tampering_and_invalid_budgets() -> None:
    payload = SearchDescriptor().payload()
    payload["route_transition_budget"] = 401
    with pytest.raises(ValueError, match="identity"):
        SearchDescriptor.from_payload(payload)
    with pytest.raises(ValueError, match="positive"):
        SearchDescriptor(route_transition_budget=0)
    with pytest.raises(ValueError, match="positive"):
        SearchDescriptor(determinization_count=0)


def test_route_and_root_telemetry_are_finite_json_values() -> None:
    route = SearchRouteTelemetry(
        root_action_index=1,
        determinization_index=0,
        value=-0.25,
        completed_turn_depth=2,
        nodes=12,
        engine_transitions=20,
        transposition_hits=3,
        repeated_position_cutoffs=1,
        budget_cutoff=True,
        principal_variation=("dogma:tools", "draw"),
    )
    root = RootSelectionTelemetry(
        search_descriptor_id=SearchDescriptor().descriptor_id,
        action_keys=("draw", "dogma:tools"),
        action_mean_values=(-0.5, -0.25),
        selected_action_index=1,
        routes=(route,),
        tied_action_indices=(1,),
    )

    assert root.selected_action_key == "dogma:tools"
    assert (
        json.loads(json.dumps(root.payload(), allow_nan=False))["routes"][0]["engine_transitions"]
        == 20
    )
    with pytest.raises(ValueError, match="finite"):
        replace(route, value=float("inf"))
