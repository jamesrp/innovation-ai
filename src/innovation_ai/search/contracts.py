"""Immutable contracts shared by the Milestone 4 sampled-search implementation.

This module deliberately defines identity and telemetry only.  It does not expand a game tree or
select an action; later search code must execute under one exact :class:`SearchDescriptor`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Protocol, cast, runtime_checkable

from innovation_ai import __version__
from innovation_ai.innovation.actions import ACTION_SCHEMA_VERSION, DECISION_SCHEMA_VERSION
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects.registry import effects_fingerprint
from innovation_ai.innovation.state import (
    PUBLIC_COVERED_INFORMATION_POLICY_VERSION,
    RULES_VERSION,
)
from innovation_ai.innovation.strategic import STRATEGIC_STATE_DIGEST_VERSION

SEARCH_DESCRIPTOR_SCHEMA_VERSION = 1
SEARCH_TELEMETRY_SCHEMA_VERSION = 1
ROLLOUT_SEARCH_DESCRIPTOR_KIND = "rollout-search"
MINIMAX_SEARCH_DESCRIPTOR_KIND = "minimax-search"
ROLLOUT_SEARCH_DESCRIPTOR_SCHEMA_VERSION = 1
ROLLOUT_TELEMETRY_SCHEMA_VERSION = 1
PRODUCTION_DETERMINIZATION_COUNT = 1
PRODUCTION_ROUTE_TRANSITION_BUDGET = 400
EXPERIMENTAL_ROLLOUT_DETERMINIZATION_COUNT = 2
EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING = 1024
# The exact R4-tested values are now production-frozen. Keep the experimental aliases because
# they are embedded in the immutable R1-R4 implementation and evidence.
PRODUCTION_ROLLOUT_DETERMINIZATION_COUNT = EXPERIMENTAL_ROLLOUT_DETERMINIZATION_COUNT
PRODUCTION_ROLLOUT_ROUTE_TRANSITION_CEILING = EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING
ROLLOUT_ROUTE_TRANSITION_CEILING = PRODUCTION_ROLLOUT_ROUTE_TRANSITION_CEILING

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

# The rollout contract is deliberately separate from the historical minimax payload. These values
# identify the versioned R1 experiment; the 1,024 ceiling remains provisional pending all-card soak.
DEFAULT_ROLLOUT_ALGORITHM = "root-sampled-deterministic-turn-rollout-v1"
DEFAULT_ROLLOUT_IMPLEMENTATION = "canonical-current-decisions-apply-action-v1"
DEFAULT_ROLLOUT_CONTINUATION_BRANCHING = "single-action-per-decision-v1"
DEFAULT_ROLLOUT_HYPOTHETICAL_LIVE_MISMATCH = "simple-rollout-then-reroot-v1"
DEFAULT_ROLLOUT_TURN_BOUNDARY = "anchored-one-completed-active-player-turn-v1"
DEFAULT_ROLLOUT_SAMPLE_AGGREGATION = "stable-sample-order-math-fsum-mean-v1"
DEFAULT_ROLLOUT_STARTING_MELD = "preselect-sampled-opponent-simple-choice-then-rollout-v1"
DEFAULT_ROLLOUT_ACTION_KEY = "canonical-semantic-action-without-decision-id-v1"
DEFAULT_ROLLOUT_CEILING_ACCOUNTING = "all-in-semantic-engine-transitions-per-route-v1"
DEFAULT_ROLLOUT_CEILING_FAILURE = "strict-failure-before-transition-over-ceiling-v1"
DEFAULT_ROLLOUT_CEILING_STATUS = "experimental-pending-all-card-route-soak-v1"
DEFAULT_ROLLOUT_INFORMATION_POLICY = PUBLIC_COVERED_INFORMATION_POLICY_VERSION
DEFAULT_ROLLOUT_INFORMATION_SET_SPEC_VERSION = DEFAULT_INFORMATION_SET_SPEC_VERSION
DEFAULT_ROLLOUT_INFORMATION_SET_SAMPLER_VERSION = "player-safe-search-sampler-v1"
DEFAULT_ROLLOUT_INFORMATION_SET_RNG_VERSION = "sha256-counter-v1"
DEFAULT_ROLLOUT_SYNTHETIC_SETUP_RNG_VERSION = "search-information-set-synthetic-setup-v1"
DEFAULT_CONTINUATION_AGENT_DESCRIPTOR_JSON = (
    '{"name":"simple-heuristic","parameters":{},"schema_version":1,'
    '"version":"printed-card-observation-v1"}'
)
DEFAULT_CONTINUATION_AGENT_DESCRIPTOR_ID = (
    "simple-heuristic-printed-card-observation-v1-0ffaec5049436bcb"
)


@runtime_checkable
class SearchDescriptorProtocol(Protocol):
    """Small common contract consumed by scheduling and diagnostic boundaries.

    ``SearchDescriptor`` exposes ``descriptor_kind`` and ``algorithm`` as properties so its
    historical schema-v1 identity payload remains byte-for-byte unchanged.
    """

    @property
    def descriptor_kind(self) -> str: ...

    @property
    def algorithm(self) -> str: ...

    @property
    def descriptor_id(self) -> str: ...

    @property
    def digest(self) -> str: ...

    @property
    def determinization_count(self) -> int: ...

    @property
    def evaluator_version(self) -> str: ...

    @property
    def information_set_spec_version(self) -> str: ...

    @property
    def sampler_seed_derivation(self) -> str: ...

    def payload(self) -> dict[str, str | int]: ...

    def dumps(self) -> str: ...


# Alternate descriptive name for callers that treat minimax and rollout as decision algorithms.
SearchAlgorithmDescriptor = SearchDescriptorProtocol


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
    def descriptor_kind(self) -> str:
        """Return the generic algorithm tag without changing the legacy identity payload."""

        return MINIMAX_SEARCH_DESCRIPTOR_KIND

    @property
    def algorithm(self) -> str:
        """Return the minimax algorithm identity through the common descriptor protocol."""

        return self.search_version

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
class RolloutSearchDescriptor:
    """Complete content-addressed identity of the deterministic rollout experiment.

    Unlike :class:`SearchDescriptor`, this contract contains no minimax-only alpha/beta,
    transposition, or iterative-deepening fields.  The explicit ``descriptor_kind`` makes generic
    decoding unambiguous and prevents historical minimax artifacts from being reinterpreted.
    """

    descriptor_kind: str = ROLLOUT_SEARCH_DESCRIPTOR_KIND
    algorithm: str = DEFAULT_ROLLOUT_ALGORITHM
    implementation_version: str = DEFAULT_ROLLOUT_IMPLEMENTATION
    evaluator_version: str = DEFAULT_EVALUATOR_VERSION
    information_policy_version: str = DEFAULT_ROLLOUT_INFORMATION_POLICY
    information_set_spec_version: str = DEFAULT_ROLLOUT_INFORMATION_SET_SPEC_VERSION
    information_set_sampler_version: str = DEFAULT_ROLLOUT_INFORMATION_SET_SAMPLER_VERSION
    information_set_rng_version: str = DEFAULT_ROLLOUT_INFORMATION_SET_RNG_VERSION
    synthetic_setup_rng_version: str = DEFAULT_ROLLOUT_SYNTHETIC_SETUP_RNG_VERSION
    sampling_algorithm: str = DEFAULT_SAMPLING_ALGORITHM
    hidden_allocation_algorithm: str = DEFAULT_HIDDEN_ALLOCATION_ALGORITHM
    sampler_seed_derivation: str = DEFAULT_SAMPLER_SEED_DERIVATION
    determinization_count: int = EXPERIMENTAL_ROLLOUT_DETERMINIZATION_COUNT
    continuation_agent_descriptor_id: str = DEFAULT_CONTINUATION_AGENT_DESCRIPTOR_ID
    continuation_agent_descriptor_json: str = DEFAULT_CONTINUATION_AGENT_DESCRIPTOR_JSON
    continuation_branching: str = DEFAULT_ROLLOUT_CONTINUATION_BRANCHING
    hypothetical_live_mismatch: str = DEFAULT_ROLLOUT_HYPOTHETICAL_LIVE_MISMATCH
    turn_boundary: str = DEFAULT_ROLLOUT_TURN_BOUNDARY
    sample_aggregation: str = DEFAULT_ROLLOUT_SAMPLE_AGGREGATION
    starting_meld_semantics: str = DEFAULT_ROLLOUT_STARTING_MELD
    root_ordering: str = DEFAULT_MOVE_ORDERING
    action_key_version: str = DEFAULT_ROLLOUT_ACTION_KEY
    tie_break: str = DEFAULT_TIE_BREAK
    route_transition_ceiling: int = EXPERIMENTAL_ROLLOUT_ROUTE_TRANSITION_CEILING
    ceiling_status: str = DEFAULT_ROLLOUT_CEILING_STATUS
    ceiling_accounting: str = DEFAULT_ROLLOUT_CEILING_ACCOUNTING
    ceiling_failure: str = DEFAULT_ROLLOUT_CEILING_FAILURE
    engine_version: str = __version__
    rules_version: str = RULES_VERSION
    action_schema_version: int = ACTION_SCHEMA_VERSION
    decision_schema_version: int = DECISION_SCHEMA_VERSION
    card_data_fingerprint: str = field(
        default_factory=lambda: load_card_registry().data_fingerprint
    )
    effects_fingerprint: str = field(default_factory=effects_fingerprint)
    schema_version: int = ROLLOUT_SEARCH_DESCRIPTOR_SCHEMA_VERSION
    descriptor_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.descriptor_kind != ROLLOUT_SEARCH_DESCRIPTOR_KIND:
            raise ValueError(f"unsupported rollout descriptor kind {self.descriptor_kind!r}")
        if self.schema_version != ROLLOUT_SEARCH_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError(f"unsupported rollout search descriptor schema {self.schema_version}")
        for item in fields(self):
            if item.name == "descriptor_id":
                continue
            value = getattr(self, item.name)
            if isinstance(value, str) and not value:
                raise ValueError(f"rollout search descriptor {item.name} cannot be empty")
        _require_positive_integer(
            self.determinization_count, "rollout search descriptor determinization_count"
        )
        _require_positive_integer(
            self.route_transition_ceiling,
            "rollout search descriptor route_transition_ceiling",
        )
        for name in ("action_schema_version", "decision_schema_version"):
            _require_positive_integer(getattr(self, name), f"rollout search descriptor {name}")
        if self.continuation_agent_descriptor_id != _agent_descriptor_id(
            self.continuation_agent_descriptor_json
        ):
            raise ValueError("continuation agent descriptor identity is invalid")
        object.__setattr__(self, "descriptor_id", _tagged_sha256(self.identity_json()))

    def identity_payload(self) -> dict[str, str | int]:
        """Return every identity-bearing scalar except the derived digest."""

        return {
            item.name: cast(str | int, getattr(self, item.name))
            for item in fields(self)
            if item.name != "descriptor_id"
        }

    def identity_json(self) -> str:
        return _canonical_json(self.identity_payload())

    @property
    def digest(self) -> str:
        return self.descriptor_id

    def payload(self) -> dict[str, str | int]:
        payload = self.identity_payload()
        payload["descriptor_id"] = self.descriptor_id
        return payload

    def dumps(self) -> str:
        return _canonical_json(self.payload())

    @classmethod
    def from_payload(cls, value: object) -> RolloutSearchDescriptor:
        """Decode only an explicitly tagged rollout schema and verify its identity."""

        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError("rollout search descriptor must be a JSON object")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise ValueError("rollout search descriptor fields differ from schema")
        integer_fields = {
            "determinization_count",
            "route_transition_ceiling",
            "action_schema_version",
            "decision_schema_version",
            "schema_version",
        }
        kwargs: dict[str, str | int] = {}
        for name in expected - {"descriptor_id"}:
            raw = value[name]
            if name in integer_fields:
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise ValueError(f"rollout search descriptor {name} must be an integer")
            elif not isinstance(raw, str):
                raise ValueError(f"rollout search descriptor {name} must be a string")
            kwargs[name] = raw
        descriptor = cls(**kwargs)  # type: ignore[arg-type]
        if value["descriptor_id"] != descriptor.descriptor_id:
            raise ValueError("rollout search descriptor content-derived identity is invalid")
        return descriptor

    @classmethod
    def loads(cls, text: str) -> RolloutSearchDescriptor:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid rollout search descriptor JSON: {error}") from error
        return cls.from_payload(value)


def _agent_descriptor_id(canonical_json: str) -> str:
    """Verify and derive the existing framework-free ``AgentDescriptor`` identity."""

    try:
        value = json.loads(canonical_json)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid continuation agent descriptor JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("continuation agent descriptor JSON must contain an object")
    if _canonical_json(value) != canonical_json:
        raise ValueError("continuation agent descriptor JSON must be canonical")
    name = value.get("name")
    version = value.get("version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ValueError("continuation agent descriptor requires non-empty name and version")
    digest = hashlib.sha256(canonical_json.encode("ascii")).hexdigest()[:16]
    return f"{name}-{version}-{digest}"


type AnySearchDescriptor = SearchDescriptor | RolloutSearchDescriptor


def tagged_search_descriptor_payload(
    descriptor: SearchDescriptorProtocol,
) -> dict[str, object]:
    """Wrap either descriptor in an explicit generic dispatch envelope.

    The nested minimax payload is exactly the historical payload and therefore keeps its original
    digest.  Rollout payloads retain their own required kind tag as an additional consistency check.
    """

    return {
        "descriptor_kind": descriptor.descriptor_kind,
        "descriptor": descriptor.payload(),
    }


def decode_search_descriptor(value: object) -> AnySearchDescriptor:
    """Decode a descriptor only when an explicit algorithm-kind tag is present."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("tagged search descriptor must be a JSON object")
    if set(value) == {"descriptor_kind", "descriptor"}:
        kind = value["descriptor_kind"]
        body = value["descriptor"]
        if not isinstance(body, dict):
            raise ValueError("tagged search descriptor body must be a JSON object")
    else:
        kind = value.get("descriptor_kind")
        body = value
    if not isinstance(kind, str) or not kind:
        raise ValueError("generic search descriptor requires an explicit descriptor_kind")
    if kind == ROLLOUT_SEARCH_DESCRIPTOR_KIND:
        if body.get("descriptor_kind") != kind:
            raise ValueError("rollout descriptor kind tag differs from its payload")
        return RolloutSearchDescriptor.from_payload(body)
    if kind == MINIMAX_SEARCH_DESCRIPTOR_KIND:
        legacy = dict(body)
        embedded_kind = legacy.pop("descriptor_kind", kind)
        if embedded_kind != kind:
            raise ValueError("minimax descriptor kind tag differs from its payload")
        return SearchDescriptor.from_payload(legacy)
    raise ValueError(f"unsupported search descriptor kind {kind!r}")


def loads_search_descriptor(text: str) -> AnySearchDescriptor:
    """Deserialize the explicitly tagged generic descriptor representation."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid tagged search descriptor JSON: {error}") from error
    return decode_search_descriptor(value)


def dumps_search_descriptor(descriptor: SearchDescriptorProtocol) -> str:
    """Serialize either descriptor with the generic explicit-kind envelope."""

    return _canonical_json(tagged_search_descriptor_payload(descriptor))


# Verb-first aliases are convenient at manifest decoding boundaries.
search_descriptor_from_payload = decode_search_descriptor
search_descriptor_payload = tagged_search_descriptor_payload


@dataclass(frozen=True, slots=True)
class RolloutLeafTelemetry:
    """Final route value and optional additive components of the frozen evaluator."""

    value: float
    terminal_reason: str | None = None
    achievement: float | None = None
    score: float | None = None
    icons: float | None = None
    board: float | None = None
    hand: float | None = None

    def __post_init__(self) -> None:
        _require_finite(self.value, "rollout leaf value")
        components = (self.achievement, self.score, self.icons, self.board, self.hand)
        if any(item is None for item in components) and any(
            item is not None for item in components
        ):
            raise ValueError("rollout leaf components must be supplied together")
        for index, item in enumerate(components):
            if item is not None:
                _require_finite(item, f"rollout leaf component {index}")
        if self.terminal_reason is not None and not self.terminal_reason:
            raise ValueError("rollout terminal reason cannot be empty")

    def payload(self) -> dict[str, object]:
        components = None
        if self.achievement is not None:
            components = {
                "achievement": self.achievement,
                "score": self.score,
                "icons": self.icons,
                "board": self.board,
                "hand": self.hand,
            }
        return {
            "value": float(self.value),
            "terminal_reason": self.terminal_reason,
            "components": components,
        }


@dataclass(frozen=True, slots=True)
class RolloutPathStep:
    """One continuation decision/action selected from a synthetic chooser's Decision."""

    action_key: str
    chooser: str
    decision_kind: str

    def __post_init__(self) -> None:
        if not self.action_key or not self.chooser or not self.decision_kind:
            raise ValueError("rollout path step fields cannot be empty")

    def payload(self) -> dict[str, str]:
        return {
            "action_key": self.action_key,
            "chooser": self.chooser,
            "decision_kind": self.decision_kind,
        }


@dataclass(frozen=True, slots=True)
class RolloutRouteTelemetry:
    """Trusted-private telemetry for one root-action/determinization rollout path."""

    root_action_index: int
    determinization_index: int
    root_action_key: str
    leaf: RolloutLeafTelemetry | None
    root_engine_transitions: int
    setup_response_engine_transitions: int
    continuation_engine_transitions: int
    completed_turn_count: int
    rollout_path: tuple[RolloutPathStep, ...] = ()
    setup_response_action_key: str | None = None
    safety_ceiling_hit: bool = False
    schema_version: int = ROLLOUT_TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROLLOUT_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported rollout telemetry schema version")
        _require_nonnegative_integer(self.root_action_index, "root action index")
        _require_nonnegative_integer(self.determinization_index, "determinization index")
        if not self.root_action_key:
            raise ValueError("rollout route requires a root action key")
        for name in (
            "root_engine_transitions",
            "setup_response_engine_transitions",
            "continuation_engine_transitions",
            "completed_turn_count",
        ):
            _require_nonnegative_integer(getattr(self, name), name.replace("_", " "))
        if self.root_engine_transitions > 1 or self.setup_response_engine_transitions > 1:
            raise ValueError("root and setup-response transition counts cannot exceed one")
        if self.completed_turn_count > 1:
            raise ValueError("version-1 rollout cannot complete more than one turn")
        if self.setup_response_action_key is not None and not self.setup_response_action_key:
            raise ValueError("setup response action key cannot be empty")
        if (self.setup_response_action_key is None) != (
            self.setup_response_engine_transitions == 0
        ):
            raise ValueError("setup response action key and transition count differ")
        if len(self.rollout_path) != self.continuation_engine_transitions:
            raise ValueError("rollout path and continuation transition count differ")
        if self.safety_ceiling_hit:
            if self.leaf is not None:
                raise ValueError("a ceiling-failed rollout route cannot contain a leaf value")
        elif self.leaf is None:
            raise ValueError("a successful rollout route requires a leaf value")

    @property
    def value(self) -> float | None:
        return None if self.leaf is None else self.leaf.value

    @property
    def terminal_reason(self) -> str | None:
        return None if self.leaf is None else self.leaf.terminal_reason

    @property
    def total_engine_transitions(self) -> int:
        return (
            self.root_engine_transitions
            + self.setup_response_engine_transitions
            + self.continuation_engine_transitions
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root_action_index": self.root_action_index,
            "determinization_index": self.determinization_index,
            "root_action_key": self.root_action_key,
            "value": self.value,
            "terminal_reason": self.terminal_reason,
            "leaf": None if self.leaf is None else self.leaf.payload(),
            "root_engine_transitions": self.root_engine_transitions,
            "setup_response_engine_transitions": self.setup_response_engine_transitions,
            "continuation_engine_transitions": self.continuation_engine_transitions,
            "total_engine_transitions": self.total_engine_transitions,
            "completed_turn_count": self.completed_turn_count,
            "setup_response_action_key": self.setup_response_action_key,
            "rollout_path": [item.payload() for item in self.rollout_path],
            "safety_ceiling_hit": self.safety_ceiling_hit,
        }

    def dumps(self) -> str:
        """Serialize this trusted-private route telemetry canonically."""

        return _canonical_json(self.payload())


@dataclass(frozen=True, slots=True)
class RolloutRootSelectionTelemetry:
    """Deterministic aggregate and trusted-private route data for one rollout root."""

    rollout_descriptor_id: str
    continuation_agent_descriptor_id: str
    information_set_digest: str
    determinization_count: int
    action_keys: tuple[str, ...]
    action_mean_values: tuple[float, ...]
    selected_action_index: int
    routes: tuple[RolloutRouteTelemetry, ...]
    tied_action_indices: tuple[int, ...] = ()
    sampler_seed_digest: str | None = None
    schema_version: int = ROLLOUT_TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROLLOUT_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported rollout telemetry schema version")
        for digest_value, name in (
            (self.rollout_descriptor_id, "rollout descriptor ID"),
            (self.information_set_digest, "information-set digest"),
        ):
            if not digest_value.startswith("sha256:"):
                raise ValueError(f"{name} must be a tagged SHA-256 digest")
        if not self.continuation_agent_descriptor_id:
            raise ValueError("continuation agent descriptor ID cannot be empty")
        _require_positive_integer(self.determinization_count, "determinization count")
        if not self.action_keys or any(not key for key in self.action_keys):
            raise ValueError("rollout root telemetry requires non-empty action keys")
        if len(self.action_keys) != len(self.action_mean_values):
            raise ValueError("rollout action keys and means must have equal length")
        if len(set(self.action_keys)) != len(self.action_keys):
            raise ValueError("rollout root action keys must be unique")
        for index, mean_value in enumerate(self.action_mean_values):
            _require_finite(mean_value, f"rollout action mean value {index}")
        _require_nonnegative_integer(self.selected_action_index, "selected action index")
        if self.selected_action_index >= len(self.action_keys):
            raise ValueError("selected rollout action index is out of range")
        if tuple(sorted(set(self.tied_action_indices))) != self.tied_action_indices:
            raise ValueError("tied rollout action indices must be unique stable order")
        if any(index < 0 or index >= len(self.action_keys) for index in self.tied_action_indices):
            raise ValueError("tied rollout action index is out of range")
        if self.tied_action_indices and self.selected_action_index not in self.tied_action_indices:
            raise ValueError("selected rollout action must be among tied best actions")
        expected_pairs = tuple(
            (action_index, sample_index)
            for action_index in range(len(self.action_keys))
            for sample_index in range(self.determinization_count)
        )
        actual_pairs = tuple(
            (route.root_action_index, route.determinization_index) for route in self.routes
        )
        if actual_pairs != expected_pairs:
            raise ValueError("rollout routes must cover every action/sample pair in stable order")
        if any(route.safety_ceiling_hit or route.leaf is None for route in self.routes):
            raise ValueError("successful root telemetry cannot contain a failed rollout route")
        if any(
            route.root_action_key != self.action_keys[route.root_action_index]
            for route in self.routes
        ):
            raise ValueError("rollout route root action key differs from root telemetry")
        for action_index, values in enumerate(self.action_sample_values):
            mean = math.fsum(values) / self.determinization_count
            if mean != self.action_mean_values[action_index]:
                raise ValueError("rollout action mean differs from stable math.fsum aggregation")
        if self.sampler_seed_digest is not None and not self.sampler_seed_digest.startswith(
            "sha256:"
        ):
            raise ValueError("rollout sampler seed digest must be a tagged SHA-256 digest")

    @property
    def search_descriptor_id(self) -> str:
        """Common diagnostic name for the rollout descriptor digest."""

        return self.rollout_descriptor_id

    @property
    def selected_action_key(self) -> str:
        return self.action_keys[self.selected_action_index]

    @property
    def action_sample_values(self) -> tuple[tuple[float, ...], ...]:
        values: list[tuple[float, ...]] = []
        for action_index in range(len(self.action_keys)):
            action_values = tuple(
                cast(float, route.value)
                for route in self.routes
                if route.root_action_index == action_index
            )
            values.append(action_values)
        return tuple(values)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rollout_descriptor_id": self.rollout_descriptor_id,
            "continuation_agent_descriptor_id": self.continuation_agent_descriptor_id,
            "information_set_digest": self.information_set_digest,
            "sampler_seed_digest": self.sampler_seed_digest,
            "determinization_count": self.determinization_count,
            "action_keys": list(self.action_keys),
            "action_sample_values": [list(values) for values in self.action_sample_values],
            "action_mean_values": [float(value) for value in self.action_mean_values],
            "selected_action_index": self.selected_action_index,
            "selected_action_key": self.selected_action_key,
            "tied_action_indices": list(self.tied_action_indices),
            "routes": [route.payload() for route in self.routes],
        }

    def dumps(self) -> str:
        return _canonical_json(self.payload())


@dataclass(frozen=True, slots=True)
class RolloutFailureTrace:
    """Trusted-private deterministic trace attached to a strict rollout failure."""

    rollout_descriptor_id: str
    continuation_agent_descriptor_id: str
    information_set_digest: str
    route: RolloutRouteTelemetry
    failure: str
    schema_version: int = ROLLOUT_TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROLLOUT_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported rollout failure trace schema version")
        if not self.failure:
            raise ValueError("rollout failure reason cannot be empty")
        if not self.route.safety_ceiling_hit:
            raise ValueError("version-1 rollout failure trace requires a ceiling-hit route")
        if not self.rollout_descriptor_id.startswith("sha256:"):
            raise ValueError("rollout failure descriptor ID must be a tagged digest")
        if not self.information_set_digest.startswith("sha256:"):
            raise ValueError("rollout failure information-set digest must be tagged")
        if not self.continuation_agent_descriptor_id:
            raise ValueError("rollout failure continuation descriptor cannot be empty")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rollout_descriptor_id": self.rollout_descriptor_id,
            "continuation_agent_descriptor_id": self.continuation_agent_descriptor_id,
            "information_set_digest": self.information_set_digest,
            "failure": self.failure,
            "route": self.route.payload(),
        }

    def dumps(self) -> str:
        return _canonical_json(self.payload())


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
RolloutRootTelemetry = RolloutRootSelectionTelemetry

PRODUCTION_SEARCH_DESCRIPTOR = SearchDescriptor()
DEFAULT_SEARCH_DESCRIPTOR = PRODUCTION_SEARCH_DESCRIPTOR
# This identity was measured throughout R1-R4. Its historical ceiling-status string is identity-
# bearing and therefore remains unchanged after production approval on September 6, 2026.
EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR = RolloutSearchDescriptor()
PRODUCTION_ROLLOUT_SEARCH_DESCRIPTOR = EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR
DEFAULT_ROLLOUT_SEARCH_DESCRIPTOR = PRODUCTION_ROLLOUT_SEARCH_DESCRIPTOR
