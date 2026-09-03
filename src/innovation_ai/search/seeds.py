"""Deterministic private seed derivation for sampled-search policy routing.

The derived bytes are supplied only to the information-set sampler.  Audit
telemetry records a one-way digest, never the seed itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from innovation_ai.innovation.types import PlayerId
from innovation_ai.search.contracts import DEFAULT_SAMPLER_SEED_DERIVATION

SEARCH_RNG_VERSION = DEFAULT_SAMPLER_SEED_DERIVATION


class SearchSeedError(ValueError):
    """A sampled-search seed identity or derivation version is invalid."""


def _seed_bytes(seed: int | str | bytes) -> bytes:
    if isinstance(seed, bool):
        raise SearchSeedError("run seed cannot be boolean")
    if isinstance(seed, int):
        return f"int:{seed}".encode("ascii")
    if isinstance(seed, str):
        return b"str:" + seed.encode("utf-8")
    if isinstance(seed, bytes):
        return b"bytes:" + seed
    raise SearchSeedError("run seed must be an int, string, or bytes")


def seed_digest(seed: bytes) -> str:
    """Return the tagged digest safe to retain in search telemetry."""

    if not isinstance(seed, bytes) or not seed:
        raise SearchSeedError("search seed must be non-empty bytes")
    return f"sha256:{hashlib.sha256(seed).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SearchRngFactory:
    """Derive batch-order-independent sampler bytes for one search root."""

    run_seed: int | str | bytes
    generation: int
    version: str = SEARCH_RNG_VERSION

    def __post_init__(self) -> None:
        _seed_bytes(self.run_seed)
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise SearchSeedError("generation must be an integer")
        if self.generation < 0:
            raise SearchSeedError("generation cannot be negative")
        if self.version != DEFAULT_SAMPLER_SEED_DERIVATION:
            raise SearchSeedError(f"unsupported search seed derivation {self.version!r}")

    def seed_for_decision(
        self,
        *,
        game_id: str,
        chooser: PlayerId,
        decision_id: int,
        policy_id: str,
        search_descriptor_id: str,
    ) -> bytes:
        """Return private sampler bytes for one fully identified search decision."""

        if not game_id:
            raise SearchSeedError("game ID cannot be empty")
        if isinstance(decision_id, bool) or not isinstance(decision_id, int) or decision_id < 1:
            raise SearchSeedError("decision ID must be positive")
        if not policy_id:
            raise SearchSeedError("policy ID cannot be empty")
        if not search_descriptor_id:
            raise SearchSeedError("search descriptor ID cannot be empty")
        payload = b"\0".join(
            (
                b"innovation-ai",
                self.version.encode("ascii"),
                _seed_bytes(self.run_seed),
                str(self.generation).encode("ascii"),
                game_id.encode("utf-8"),
                chooser.value.encode("ascii"),
                str(decision_id).encode("ascii"),
                policy_id.encode("utf-8"),
                search_descriptor_id.encode("ascii"),
            )
        )
        return hashlib.sha256(payload).digest()


__all__ = [
    "SEARCH_RNG_VERSION",
    "SearchRngFactory",
    "SearchSeedError",
    "seed_digest",
]
