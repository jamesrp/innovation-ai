"""Stable, framework-free descriptors for baseline decision policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from innovation_ai.search.contracts import PRODUCTION_SEARCH_DESCRIPTOR

type AgentParameter = str | int | bool | None
AGENT_DESCRIPTOR_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    """A portable identity for an agent implementation, not a live agent instance.

    Runtime RNG seeds are intentionally separate. This lets a manifest identify the policy
    independently of the game/seat seed that drives a particular actor instance.
    """

    name: str
    version: str
    parameters: tuple[tuple[str, AgentParameter], ...] = ()
    schema_version: int = AGENT_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AGENT_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError("unsupported agent descriptor schema version")
        if not self.name or not self.version:
            raise ValueError("agent descriptor name and version cannot be empty")
        keys = tuple(key for key, _ in self.parameters)
        if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
            raise ValueError("agent descriptor parameters must have unique sorted keys")
        for key, value in self.parameters:
            if not key:
                raise ValueError("agent descriptor parameter name cannot be empty")
            if not isinstance(value, (str, int, bool)) and value is not None:
                raise TypeError("agent descriptor parameters must be JSON scalars")

    def payload(self) -> dict[str, object]:
        """Return the canonical JSON-compatible manifest representation."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "parameters": {key: value for key, value in self.parameters},
        }

    def canonical_json(self) -> str:
        """Return a stable byte representation suitable for manifests and hashing."""

        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def descriptor_id(self) -> str:
        """Return a content-addressed identifier without Python object identity."""

        digest = sha256(self.canonical_json().encode("ascii")).hexdigest()[:16]
        return f"{self.name}-{self.version}-{digest}"


RANDOM_AGENT_DESCRIPTOR = AgentDescriptor(
    name="random",
    version="python-mt19937-randrange-v1",
)
SIMPLE_HEURISTIC_AGENT_DESCRIPTOR = AgentDescriptor(
    name="simple-heuristic",
    version="printed-card-observation-v1",
)
SAMPLED_MINIMAX_AGENT_DESCRIPTOR = AgentDescriptor(
    name="sampled-minimax-heuristic",
    version="v1",
    parameters=(("search_descriptor_id", PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id),),
)
