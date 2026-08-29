"""Domain-separated deterministic seed derivation for run manifests."""

from __future__ import annotations

from hashlib import blake2b

type SeedPart = str | int
SEED_DERIVATION_VERSION = "blake2b-128-length-prefixed-v1"


def _encoded_part(part: SeedPart) -> bytes:
    if isinstance(part, str):
        kind = b"s"
        value = part.encode("utf-8")
    elif isinstance(part, int) and not isinstance(part, bool):
        kind = b"i"
        value = str(part).encode("ascii")
    else:
        raise TypeError("seed parts must be strings or integers")
    return kind + len(value).to_bytes(4, "big") + value


def derive_seed(root_seed: int, domain: str, /, *parts: SeedPart) -> int:
    """Derive one non-negative 64-bit seed from an explicit domain and stable parts.

    The domain is part of the hash input rather than a naming convention, so using a game setup
    seed at an agent-RNG call site cannot accidentally reproduce the same random stream.
    """

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root seed must be an integer")
    if not domain:
        raise ValueError("seed derivation domain cannot be empty")
    material = b"innovation-ai.seed\0" + _encoded_part(domain) + _encoded_part(root_seed)
    material += b"".join(_encoded_part(part) for part in parts)
    return int.from_bytes(blake2b(material, digest_size=16).digest()[:8], "big")


def setup_seed(run_seed: int, game_id: str, /) -> int:
    """Derive a game setup seed in the reserved Stage-0 setup domain."""

    return derive_seed(run_seed, "stage0/setup", game_id)


def agent_seed(run_seed: int, game_id: str, seat: str, agent_id: str, /) -> int:
    """Derive one baseline-agent seed distinct from setup and other seats."""

    return derive_seed(run_seed, "stage0/agent", game_id, seat, agent_id)
