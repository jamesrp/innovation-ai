from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from innovation_ai.agents.descriptors import SIMPLE_HEURISTIC_AGENT_DESCRIPTOR
from innovation_ai.search import (
    EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR,
    MINIMAX_SEARCH_DESCRIPTOR_KIND,
    PRODUCTION_SEARCH_DESCRIPTOR,
    ROLLOUT_SEARCH_DESCRIPTOR_KIND,
    RolloutFailureTrace,
    RolloutLeafTelemetry,
    RolloutPathStep,
    RolloutRootSelectionTelemetry,
    RolloutRouteTelemetry,
    RolloutSearchDescriptor,
    RootSelectionTelemetry,
    SearchDescriptor,
    SearchDescriptorProtocol,
    SearchRouteTelemetry,
    decode_search_descriptor,
    dumps_search_descriptor,
    loads_search_descriptor,
)


def _rollout_route(
    *,
    action_index: int = 0,
    sample_index: int = 0,
    action_key: str = "draw",
) -> RolloutRouteTelemetry:
    return RolloutRouteTelemetry(
        root_action_index=action_index,
        determinization_index=sample_index,
        root_action_key=action_key,
        leaf=RolloutLeafTelemetry(0.25),
        root_engine_transitions=1,
        setup_response_engine_transitions=0,
        continuation_engine_transitions=0,
        completed_turn_count=1,
    )


def _rollout_root() -> RolloutRootSelectionTelemetry:
    descriptor = RolloutSearchDescriptor(determinization_count=1)
    return RolloutRootSelectionTelemetry(
        rollout_descriptor_id=descriptor.descriptor_id,
        continuation_agent_descriptor_id=descriptor.continuation_agent_descriptor_id,
        information_set_digest="sha256:" + "1" * 64,
        determinization_count=1,
        action_keys=("draw",),
        action_mean_values=(0.25,),
        selected_action_index=0,
        routes=(_rollout_route(),),
    )


def _root_telemetry() -> RootSelectionTelemetry:
    return RootSelectionTelemetry(
        search_descriptor_id=SearchDescriptor().descriptor_id,
        action_keys=("draw", "dogma:tools"),
        action_mean_values=(0.0, 1.0),
        selected_action_index=1,
        routes=(),
        tied_action_indices=(1,),
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


def test_rollout_descriptor_is_distinct_tagged_and_content_addressed() -> None:
    descriptor = RolloutSearchDescriptor()

    assert descriptor == EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR
    assert descriptor.descriptor_kind == ROLLOUT_SEARCH_DESCRIPTOR_KIND
    assert descriptor.algorithm == "root-sampled-deterministic-turn-rollout-v1"
    assert descriptor.determinization_count == 2
    assert descriptor.route_transition_ceiling == 1024
    assert descriptor.ceiling_status == "experimental-pending-all-card-route-soak-v1"
    assert descriptor.implementation_version == "canonical-current-decisions-apply-action-v1"
    assert descriptor.continuation_agent_descriptor_id.startswith(
        "simple-heuristic-printed-card-observation-v1-"
    )
    assert (
        descriptor.continuation_agent_descriptor_json
        == SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.canonical_json()
    )
    assert (
        descriptor.continuation_agent_descriptor_id
        == SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.descriptor_id
    )
    assert "alpha_beta" not in descriptor.payload()
    assert "transposition_cache_scope" not in descriptor.payload()
    assert RolloutSearchDescriptor.loads(descriptor.dumps()) == descriptor
    assert isinstance(descriptor, SearchDescriptorProtocol)

    changed = replace(descriptor, determinization_count=1)
    assert changed.descriptor_id != descriptor.descriptor_id
    with pytest.raises(ValueError, match="identity"):
        replace(descriptor, continuation_agent_descriptor_id="not-the-canonical-id")


def test_generic_descriptor_decoder_requires_explicit_kind_and_preserves_legacy_payload() -> None:
    minimax = SearchDescriptor()
    rollout = RolloutSearchDescriptor()
    legacy_json = minimax.dumps()

    assert minimax.descriptor_kind == MINIMAX_SEARCH_DESCRIPTOR_KIND
    assert minimax.algorithm == minimax.search_version
    assert "descriptor_kind" not in minimax.payload()
    assert minimax.descriptor_id == (
        "sha256:a3de2116646c297b30b75e25fc4ab065ef1a0674f89b932eecd9ef5eea2b067e"
    )
    assert SearchDescriptor.loads(legacy_json) == minimax
    with pytest.raises(ValueError, match="explicit descriptor_kind"):
        decode_search_descriptor(minimax.payload())

    assert loads_search_descriptor(dumps_search_descriptor(minimax)) == minimax
    assert loads_search_descriptor(dumps_search_descriptor(rollout)) == rollout
    assert decode_search_descriptor(rollout.payload()) == rollout

    mismatched = json.loads(dumps_search_descriptor(rollout))
    mismatched["descriptor_kind"] = MINIMAX_SEARCH_DESCRIPTOR_KIND
    with pytest.raises(ValueError):
        decode_search_descriptor(mismatched)


def test_descriptor_decoders_reject_wrong_shapes_types_and_invalid_json() -> None:
    descriptor = SearchDescriptor()
    assert descriptor.digest == descriptor.descriptor_id

    with pytest.raises(ValueError, match="unsupported search descriptor schema"):
        SearchDescriptor(schema_version=2)
    with pytest.raises(ValueError, match="cannot be empty"):
        SearchDescriptor(search_version="")
    invalid_payloads: tuple[object, ...] = (None, [], {1: "value"})
    for invalid in invalid_payloads:
        with pytest.raises(ValueError, match="JSON object"):
            SearchDescriptor.from_payload(invalid)
    missing = descriptor.payload()
    missing.pop("search_version")
    with pytest.raises(ValueError, match="fields differ"):
        SearchDescriptor.from_payload(missing)
    wrong_integer = descriptor.payload()
    wrong_integer["determinization_count"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        SearchDescriptor.from_payload(wrong_integer)
    wrong_string = descriptor.payload()
    wrong_string["search_version"] = 1
    with pytest.raises(ValueError, match="must be a string"):
        SearchDescriptor.from_payload(wrong_string)
    with pytest.raises(ValueError, match="invalid search descriptor JSON"):
        SearchDescriptor.loads("{")


def test_rollout_descriptor_rejects_wrong_shapes_types_and_agent_identity_documents() -> None:
    descriptor = RolloutSearchDescriptor()
    assert descriptor.digest == descriptor.descriptor_id

    descriptor_changes: tuple[tuple[dict[str, Any], str], ...] = (
        ({"descriptor_kind": "other"}, "unsupported rollout descriptor kind"),
        ({"schema_version": 2}, "unsupported rollout search descriptor schema"),
        ({"algorithm": ""}, "cannot be empty"),
        ({"determinization_count": False}, "positive integer"),
        ({"route_transition_ceiling": 0}, "positive integer"),
        ({"action_schema_version": 0}, "positive integer"),
    )
    for changes, message in descriptor_changes:
        with pytest.raises(ValueError, match=message):
            replace(descriptor, **changes)

    invalid_payloads: tuple[object, ...] = (None, [], {1: "value"})
    for invalid in invalid_payloads:
        with pytest.raises(ValueError, match="JSON object"):
            RolloutSearchDescriptor.from_payload(invalid)
    extra = descriptor.payload()
    extra["unexpected"] = "field"
    with pytest.raises(ValueError, match="fields differ"):
        RolloutSearchDescriptor.from_payload(extra)
    wrong_integer = descriptor.payload()
    wrong_integer["schema_version"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        RolloutSearchDescriptor.from_payload(wrong_integer)
    wrong_string = descriptor.payload()
    wrong_string["algorithm"] = 1
    with pytest.raises(ValueError, match="must be a string"):
        RolloutSearchDescriptor.from_payload(wrong_string)
    tampered = descriptor.payload()
    tampered["descriptor_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="content-derived identity"):
        RolloutSearchDescriptor.from_payload(tampered)
    with pytest.raises(ValueError, match="invalid rollout search descriptor JSON"):
        RolloutSearchDescriptor.loads("{")

    for document, message in (
        ("{", "invalid continuation agent descriptor JSON"),
        ("[]", "must contain an object"),
        ('{"version":"v1","name":"agent"}', "must be canonical"),
        ('{"name":"","version":"v1"}', "requires non-empty name and version"),
        ('{"name":"agent","version":""}', "requires non-empty name and version"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(descriptor, continuation_agent_descriptor_json=document)


def test_generic_descriptor_decoder_rejects_ambiguous_and_malformed_envelopes() -> None:
    invalid_payloads: tuple[object, ...] = (None, [], {1: "value"})
    for invalid in invalid_payloads:
        with pytest.raises(ValueError, match="tagged search descriptor must be a JSON object"):
            decode_search_descriptor(invalid)
    with pytest.raises(ValueError, match="body must be a JSON object"):
        decode_search_descriptor(
            {"descriptor_kind": ROLLOUT_SEARCH_DESCRIPTOR_KIND, "descriptor": 1}
        )

    rollout = RolloutSearchDescriptor().payload()
    rollout["descriptor_kind"] = MINIMAX_SEARCH_DESCRIPTOR_KIND
    with pytest.raises(ValueError, match="rollout descriptor kind tag differs"):
        decode_search_descriptor(
            {"descriptor_kind": ROLLOUT_SEARCH_DESCRIPTOR_KIND, "descriptor": rollout}
        )

    minimax = SearchDescriptor().payload()
    minimax["descriptor_kind"] = ROLLOUT_SEARCH_DESCRIPTOR_KIND
    with pytest.raises(ValueError, match="minimax descriptor kind tag differs"):
        decode_search_descriptor(
            {"descriptor_kind": MINIMAX_SEARCH_DESCRIPTOR_KIND, "descriptor": minimax}
        )
    with pytest.raises(ValueError, match="unsupported search descriptor kind"):
        decode_search_descriptor({"descriptor_kind": "unknown", "descriptor": {}})
    with pytest.raises(ValueError, match="invalid tagged search descriptor JSON"):
        loads_search_descriptor("{")


def test_rollout_leaf_path_and_route_contracts_reject_incomplete_or_impossible_data() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        RolloutLeafTelemetry(0.0, achievement=0.0)
    with pytest.raises(ValueError, match="component 4 must be finite"):
        RolloutLeafTelemetry(
            0.0,
            achievement=0.0,
            score=0.0,
            icons=0.0,
            board=0.0,
            hand=float("nan"),
        )
    with pytest.raises(ValueError, match="terminal reason cannot be empty"):
        RolloutLeafTelemetry(0.0, terminal_reason="")
    step = RolloutPathStep("draw", "player-1", "turn-action")
    path_changes: tuple[dict[str, str], ...] = (
        {"action_key": ""},
        {"chooser": ""},
        {"decision_kind": ""},
    )
    for path_change in path_changes:
        with pytest.raises(ValueError, match="fields cannot be empty"):
            replace(step, **path_change)

    route = _rollout_route()
    invalid_routes: tuple[tuple[dict[str, Any], str], ...] = (
        ({"schema_version": 2}, "unsupported rollout telemetry schema"),
        ({"root_action_index": -1}, "non-negative integer"),
        ({"determinization_index": False}, "non-negative integer"),
        ({"root_action_key": ""}, "requires a root action key"),
        ({"root_engine_transitions": 2}, "cannot exceed one"),
        ({"setup_response_engine_transitions": 2}, "cannot exceed one"),
        ({"completed_turn_count": 2}, "cannot complete more than one turn"),
        (
            {"setup_response_action_key": "", "setup_response_engine_transitions": 1},
            "cannot be empty",
        ),
        ({"setup_response_action_key": "meld:tools"}, "transition count differ"),
        ({"setup_response_engine_transitions": 1}, "transition count differ"),
        ({"continuation_engine_transitions": 1}, "rollout path.*differ"),
        ({"safety_ceiling_hit": True}, "cannot contain a leaf"),
        ({"leaf": None}, "requires a leaf"),
    )
    for route_changes, message in invalid_routes:
        with pytest.raises(ValueError, match=message):
            replace(route, **route_changes)

    failed = replace(route, leaf=None, safety_ceiling_hit=True, completed_turn_count=0)
    assert failed.value is None
    assert failed.terminal_reason is None
    assert failed.payload()["leaf"] is None


def test_rollout_root_telemetry_rejects_inconsistent_aggregate_data() -> None:
    root = _rollout_root()
    failed_route = replace(
        root.routes[0], leaf=None, safety_ceiling_hit=True, completed_turn_count=0
    )
    wrong_key_route = replace(root.routes[0], root_action_key="meld:tools")
    invalid_roots: tuple[tuple[dict[str, Any], str], ...] = (
        ({"schema_version": 2}, "unsupported rollout telemetry schema"),
        ({"rollout_descriptor_id": "bad"}, "rollout descriptor ID must be"),
        ({"information_set_digest": "bad"}, "information-set digest must be"),
        ({"continuation_agent_descriptor_id": ""}, "cannot be empty"),
        ({"determinization_count": 0}, "positive integer"),
        ({"action_keys": ()}, "non-empty action keys"),
        ({"action_keys": ("draw", "draw"), "action_mean_values": (0.25, 0.25)}, "unique"),
        ({"action_mean_values": ()}, "equal length"),
        ({"action_mean_values": (float("nan"),)}, "must be finite"),
        ({"selected_action_index": 1}, "out of range"),
        ({"tied_action_indices": (0, 0)}, "unique stable order"),
        ({"tied_action_indices": (1,)}, "out of range"),
        ({"routes": ()}, "cover every action/sample pair"),
        ({"routes": (failed_route,)}, "cannot contain a failed"),
        ({"routes": (wrong_key_route,)}, "root action key differs"),
        ({"action_mean_values": (0.5,)}, "mean differs"),
        ({"sampler_seed_digest": "bad"}, "sampler seed digest must be"),
    )
    for root_changes, message in invalid_roots:
        with pytest.raises(ValueError, match=message):
            replace(root, **root_changes)

    two_action_routes = (
        _rollout_route(action_index=0, action_key="draw"),
        _rollout_route(action_index=1, action_key="meld:tools"),
    )
    two_action = replace(
        root,
        action_keys=("draw", "meld:tools"),
        action_mean_values=(0.25, 0.25),
        routes=two_action_routes,
        tied_action_indices=(0, 1),
    )
    with pytest.raises(ValueError, match="unique stable order"):
        replace(two_action, tied_action_indices=(1, 0))
    with pytest.raises(ValueError, match="among tied best actions"):
        replace(two_action, selected_action_index=1, tied_action_indices=(0,))
    assert two_action.search_descriptor_id == two_action.rollout_descriptor_id


def test_rollout_failure_trace_requires_a_complete_failed_route_identity() -> None:
    descriptor = RolloutSearchDescriptor()
    failed_route = replace(
        _rollout_route(), leaf=None, safety_ceiling_hit=True, completed_turn_count=0
    )
    trace = RolloutFailureTrace(
        descriptor.descriptor_id,
        descriptor.continuation_agent_descriptor_id,
        "sha256:" + "1" * 64,
        failed_route,
        "route-transition-ceiling",
    )
    assert json.loads(trace.dumps())["route"]["safety_ceiling_hit"] is True

    trace_changes: tuple[tuple[dict[str, Any], str], ...] = (
        ({"schema_version": 2}, "unsupported rollout failure trace schema"),
        ({"failure": ""}, "failure reason cannot be empty"),
        ({"route": _rollout_route()}, "requires a ceiling-hit route"),
        ({"rollout_descriptor_id": "bad"}, "descriptor ID must be a tagged digest"),
        ({"information_set_digest": "bad"}, "information-set digest must be tagged"),
        ({"continuation_agent_descriptor_id": ""}, "continuation descriptor cannot be empty"),
    )
    for trace_change, message in trace_changes:
        with pytest.raises(ValueError, match=message):
            replace(trace, **trace_change)


def test_legacy_search_telemetry_rejects_invalid_route_and_root_invariants() -> None:
    route = SearchRouteTelemetry(0, 0, 0.0, 0, 0, 0)
    with pytest.raises(ValueError, match="unsupported search telemetry schema"):
        replace(route, schema_version=2)
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(route, nodes=-1)
    with pytest.raises(ValueError, match="principal-variation entries cannot be empty"):
        replace(route, principal_variation=("",))

    root = _root_telemetry()
    invalid_legacy_roots: tuple[tuple[dict[str, Any], str], ...] = (
        ({"schema_version": 2}, "unsupported search telemetry schema"),
        ({"search_descriptor_id": "bad"}, "tagged SHA-256 digest"),
        ({"action_keys": ()}, "non-empty action keys"),
        ({"action_mean_values": (0.0,)}, "equal length"),
        ({"action_keys": ("draw", "draw")}, "must be unique"),
        ({"action_mean_values": (0.0, float("inf"))}, "must be finite"),
        ({"selected_action_index": 2}, "out of range"),
        ({"tied_action_indices": (1, 1)}, "cannot repeat"),
        ({"tied_action_indices": (1, 0)}, "stable action order"),
        ({"tied_action_indices": (2,)}, "out of range"),
        ({"selected_action_index": 0, "tied_action_indices": (1,)}, "among the tied"),
        ({"routes": (replace(route, root_action_index=2),)}, "route root action index"),
        ({"selector_seed_digest": "bad"}, "selector seed digest"),
    )
    for root_changes, message in invalid_legacy_roots:
        with pytest.raises(ValueError, match=message):
            replace(root, **root_changes)


def test_rollout_telemetry_serializes_stable_samples_paths_and_leaf_components() -> None:
    descriptor = RolloutSearchDescriptor(determinization_count=1)
    leaf = RolloutLeafTelemetry(
        0.25,
        achievement=0.1,
        score=0.05,
        icons=0.05,
        board=0.03,
        hand=0.02,
    )
    route = RolloutRouteTelemetry(
        root_action_index=0,
        determinization_index=0,
        root_action_key="draw",
        leaf=leaf,
        root_engine_transitions=1,
        setup_response_engine_transitions=0,
        continuation_engine_transitions=1,
        completed_turn_count=1,
        rollout_path=(RolloutPathStep("meld:tools", "player-1", "turn-action"),),
    )
    telemetry = RolloutRootSelectionTelemetry(
        rollout_descriptor_id=descriptor.descriptor_id,
        continuation_agent_descriptor_id=descriptor.continuation_agent_descriptor_id,
        information_set_digest="sha256:" + "1" * 64,
        sampler_seed_digest="sha256:" + "2" * 64,
        determinization_count=1,
        action_keys=("draw",),
        action_mean_values=(0.25,),
        selected_action_index=0,
        tied_action_indices=(0,),
        routes=(route,),
    )

    payload = json.loads(telemetry.dumps())
    assert payload["action_sample_values"] == [[0.25]]
    assert payload["routes"][0]["total_engine_transitions"] == 2
    assert payload["routes"][0]["leaf"]["components"]["achievement"] == 0.1
    assert json.loads(route.dumps())["rollout_path"][0]["chooser"] == "player-1"
