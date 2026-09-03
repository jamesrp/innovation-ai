"""Immutable contracts shared by the Milestone 4 sampled-search implementation.

This module deliberately defines identity and telemetry only.  It does not expand a game tree or
select an action; later search code must execute under one exact :class:`SearchDescriptor`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import cast

from innovation_ai.innovation.strategic import STRATEGIC_STATE_DIGEST_VERSION

SEARCH_DESCRIPTOR_SCHEMA_VERSION = 1
SEARCH_TELEMETRY_SCHEMA_VERSION = 1
PRODUCTION_DETERMINIZATION_COUNT = 1
PRODUCTION_ROUTE_TRANSITION_BUDGET = 400

# These strings are policy identity, not prose labels.  Changing any search behavior requires a
# new value (and therefore a new descriptor digest), even if the dataclass schema does not change.
DEFAULT_SEARCH_VERSION = "root-sampled-minimax-one-completed-turn-v1"
DEFAULT_EVALUATOR_VERSION = "hand-engineered-leaf-v1"
DEFAULT_INFORMATION_SET_SPEC_VERSION = "player-safe-search-spec-v1"
DEFAULT_SAMPLING_ALGORITHM = "fixed-count-root-determinization-v1"
DEFAULT_HIDDEN_ALLOCATION_ALGORITHM = "without-replacement-hidden-allocation-v1"
DEFAULT_SAMPLER_SEED_DERIVATION = "sha256-search-route-seed-v1"
DEFAULT_SELECTOR_SEED_DERIVATION = "sha256-search-selector-seed-v1"
DEFAULT_SAMPLE_AGGREGATION = "arithmetic-mean-by-root-action-v1"
DEFAULT_BUDGET_ACCOUNTING = "engine-transitions-per-root-action-determinization-v1"
DEFAULT_ITERATIVE_DEEPENING_COMPLETION_RULE = (
    "deepest-fully-completed-turn-depth-else-immediate-leaf-v1"
)
DEFAULT_MOVE_ORDERING = "stable-legal-action-order-v1"
DEFAULT_TRANSPOSITION_CACHE_SCOPE = "route-local-v1"
DEFAULT_TRANSPOSITION_ENTRY_SEMANTICS = "exact-values-only-after-complete-subtree-v1"
DEFAULT_ALPHA_BETA = "stable-order-alpha-beta-v1"
DEFAULT_STARTING_MELD_AGGREGATION = "max-root-mean-samples-min-legal-opponent-v1"
DEFAULT_ROOT_TRANSITION_BUDGETING = "root-and-mandatory-setup-response-outside-budget-v1"
DEFAULT_TIE_BREAK = "stable-legal-action-order-v1"
DEFAULT_CYCLE_CUTOFF = "repeated-strategic-state-to-leaf-v1"
DEFAULT_FALLBACK_POLICY = "deterministic-immediate-leaf-v1"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _tagged_sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('ascii')).hexdigest()}"


def _require_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class SearchDescriptor:
    """Complete content-addressed identity of one deterministic search policy.

    Every behavioral choice called out by the Milestone 4 plan is represented explicitly.  The
    numeric defaults are the current one-turn feasibility candidate; callers may construct separate
    immutable descriptors for measured budget and determinization alternatives.
    """

    search_version: str = DEFAULT_SEARCH_VERSION
    evaluator_version: str = DEFAULT_EVALUATOR_VERSION
    information_set_spec_version: str = DEFAULT_INFORMATION_SET_SPEC_VERSION
    sampling_algorithm: str = DEFAULT_SAMPLING_ALGORITHM
    hidden_allocation_algorithm: str = DEFAULT_HIDDEN_ALLOCATION_ALGORITHM
    sampler_seed_derivation: str = DEFAULT_SAMPLER_SEED_DERIVATION
    selector_seed_derivation: str = DEFAULT_SELECTOR_SEED_DERIVATION
    root_turn_horizon: int = 1
    opponent_turn_horizon: int = 1
    starting_meld_horizon: int = 1
    determinization_count: int = PRODUCTION_DETERMINIZATION_COUNT
    sample_aggregation: str = DEFAULT_SAMPLE_AGGREGATION
    route_transition_budget: int = PRODUCTION_ROUTE_TRANSITION_BUDGET
    budget_accounting: str = DEFAULT_BUDGET_ACCOUNTING
    iterative_deepening_completion_rule: str = DEFAULT_ITERATIVE_DEEPENING_COMPLETION_RULE
    move_ordering: str = DEFAULT_MOVE_ORDERING
    transposition_key_version: str = STRATEGIC_STATE_DIGEST_VERSION
    transposition_cache_scope: str = DEFAULT_TRANSPOSITION_CACHE_SCOPE
    transposition_entry_semantics: str = DEFAULT_TRANSPOSITION_ENTRY_SEMANTICS
    alpha_beta: str = DEFAULT_ALPHA_BETA
    starting_meld_aggregation: str = DEFAULT_STARTING_MELD_AGGREGATION
    root_transition_budgeting: str = DEFAULT_ROOT_TRANSITION_BUDGETING
    tie_break: str = DEFAULT_TIE_BREAK
    cycle_cutoff: str = DEFAULT_CYCLE_CUTOFF
    fallback_policy: str = DEFAULT_FALLBACK_POLICY
    schema_version: int = SEARCH_DESCRIPTOR_SCHEMA_VERSION
    descriptor_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError(f"unsupported search descriptor schema {self.schema_version}")
        for item in fields(self):
            if item.name == "descriptor_id":
                continue
            value = getattr(self, item.name)
            if isinstance(value, str) and not value:
                raise ValueError(f"search descriptor {item.name} cannot be empty")
        for name in (
            "root_turn_horizon",
            "opponent_turn_horizon",
            "starting_meld_horizon",
            "determinization_count",
            "route_transition_budget",
        ):
            _require_positive_integer(getattr(self, name), f"search descriptor {name}")
        object.__setattr__(self, "descriptor_id", _tagged_sha256(self.identity_json()))

    def identity_payload(self) -> dict[str, str | int]:
        """Return the canonical JSON-compatible fields covered by ``descriptor_id``."""

        return {
            item.name: cast(str | int, getattr(self, item.name))
            for item in fields(self)
            if item.name != "descriptor_id"
        }

    def identity_json(self) -> str:
        """Return stable canonical JSON for the identity-bearing fields."""

        return _canonical_json(self.identity_payload())

    @property
    def digest(self) -> str:
        """Alias used by complete policy descriptors that reference this search contract."""

        return self.descriptor_id

    def payload(self) -> dict[str, str | int]:
        """Return canonical JSON-compatible descriptor content plus its derived identity."""

        payload = self.identity_payload()
        payload["descriptor_id"] = self.descriptor_id
        return payload

    def dumps(self) -> str:
        """Serialize the descriptor canonically."""

        return _canonical_json(self.payload())

    @classmethod
    def from_payload(cls, value: object) -> SearchDescriptor:
        """Decode an exact schema-v1 descriptor and verify its content-derived identity."""

        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError("search descriptor must be a JSON object")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise ValueError("search descriptor fields differ from schema")
        kwargs: dict[str, str | int] = {}
        integer_fields = {
            "root_turn_horizon",
            "opponent_turn_horizon",
            "starting_meld_horizon",
            "determinization_count",
            "route_transition_budget",
            "schema_version",
        }
        for name in expected - {"descriptor_id"}:
            raw = value[name]
            if name in integer_fields:
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise ValueError(f"search descriptor {name} must be an integer")
            elif not isinstance(raw, str):
                raise ValueError(f"search descriptor {name} must be a string")
            kwargs[name] = raw
        descriptor = cls(**kwargs)  # type: ignore[arg-type]
        if value["descriptor_id"] != descriptor.descriptor_id:
            raise ValueError("search descriptor content-derived identity is invalid")
        return descriptor

    @classmethod
    def loads(cls, text: str) -> SearchDescriptor:
        """Deserialize and verify a canonical or ordinary JSON descriptor document."""

        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid search descriptor JSON: {error}") from error
        return cls.from_payload(value)


@dataclass(frozen=True, slots=True)
class SearchRouteTelemetry:
    """Audit data for one independently budgeted ``(action, determinization)`` route."""

    root_action_index: int
    determinization_index: int
    value: float
    completed_turn_depth: int
    nodes: int
    engine_transitions: int
    transposition_hits: int = 0
    repeated_position_cutoffs: int = 0
    budget_cutoff: bool = False
    immediate_leaf_fallback: bool = False
    principal_variation: tuple[str, ...] = ()
    schema_version: int = SEARCH_TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported search telemetry schema version")
        _require_nonnegative_integer(self.root_action_index, "root action index")
        _require_nonnegative_integer(self.determinization_index, "determinization index")
        _require_finite(self.value, "route value")
        for name in (
            "completed_turn_depth",
            "nodes",
            "engine_transitions",
            "transposition_hits",
            "repeated_position_cutoffs",
        ):
            _require_nonnegative_integer(getattr(self, name), name.replace("_", " "))
        if any(not item for item in self.principal_variation):
            raise ValueError("principal-variation entries cannot be empty")

    def payload(self) -> dict[str, object]:
        """Return a JSON-compatible representation with tuple order preserved."""

        return {
            "schema_version": self.schema_version,
            "root_action_index": self.root_action_index,
            "determinization_index": self.determinization_index,
            "value": float(self.value),
            "completed_turn_depth": self.completed_turn_depth,
            "nodes": self.nodes,
            "engine_transitions": self.engine_transitions,
            "transposition_hits": self.transposition_hits,
            "repeated_position_cutoffs": self.repeated_position_cutoffs,
            "budget_cutoff": self.budget_cutoff,
            "immediate_leaf_fallback": self.immediate_leaf_fallback,
            "principal_variation": list(self.principal_variation),
        }


@dataclass(frozen=True, slots=True)
class RootSelectionTelemetry:
    """JSON-safe aggregate values and exact stable-order result for one search root."""

    search_descriptor_id: str
    action_keys: tuple[str, ...]
    action_mean_values: tuple[float, ...]
    selected_action_index: int
    routes: tuple[SearchRouteTelemetry, ...]
    tied_action_indices: tuple[int, ...] = ()
    selector_seed_digest: str | None = None
    schema_version: int = SEARCH_TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported search telemetry schema version")
        if not self.search_descriptor_id.startswith("sha256:"):
            raise ValueError("search descriptor ID must be a tagged SHA-256 digest")
        if not self.action_keys or any(not key for key in self.action_keys):
            raise ValueError("root telemetry requires non-empty action keys")
        if len(self.action_keys) != len(self.action_mean_values):
            raise ValueError("root action keys and mean values must have equal length")
        if len(set(self.action_keys)) != len(self.action_keys):
            raise ValueError("root action keys must be unique")
        for index, value in enumerate(self.action_mean_values):
            _require_finite(value, f"action mean value {index}")
        _require_nonnegative_integer(self.selected_action_index, "selected action index")
        if self.selected_action_index >= len(self.action_keys):
            raise ValueError("selected action index is out of range")
        if len(set(self.tied_action_indices)) != len(self.tied_action_indices):
            raise ValueError("tied action indices cannot repeat")
        if tuple(sorted(self.tied_action_indices)) != self.tied_action_indices:
            raise ValueError("tied action indices must be in stable action order")
        if any(index < 0 or index >= len(self.action_keys) for index in self.tied_action_indices):
            raise ValueError("tied action index is out of range")
        if self.tied_action_indices and self.selected_action_index not in self.tied_action_indices:
            raise ValueError("selected action must be among the tied best actions")
        if any(route.root_action_index >= len(self.action_keys) for route in self.routes):
            raise ValueError("route root action index is out of range")
        if self.selector_seed_digest is not None and not self.selector_seed_digest.startswith(
            "sha256:"
        ):
            raise ValueError("selector seed digest must be a tagged SHA-256 digest")

    @property
    def selected_action_key(self) -> str:
        """Return the selected semantic action key without duplicating it in telemetry."""

        return self.action_keys[self.selected_action_index]

    def payload(self) -> dict[str, object]:
        """Return a JSON-compatible representation suitable for diagnostic traces."""

        return {
            "schema_version": self.schema_version,
            "search_descriptor_id": self.search_descriptor_id,
            "action_keys": list(self.action_keys),
            "action_mean_values": [float(value) for value in self.action_mean_values],
            "selected_action_index": self.selected_action_index,
            "selected_action_key": self.selected_action_key,
            "tied_action_indices": list(self.tied_action_indices),
            "selector_seed_digest": self.selector_seed_digest,
            "routes": [route.payload() for route in self.routes],
        }


# Short aliases keep call sites readable while the longer public names remain self-documenting.
RouteTelemetry = SearchRouteTelemetry
RootTelemetry = RootSelectionTelemetry

PRODUCTION_SEARCH_DESCRIPTOR = SearchDescriptor()
DEFAULT_SEARCH_DESCRIPTOR = PRODUCTION_SEARCH_DESCRIPTOR
