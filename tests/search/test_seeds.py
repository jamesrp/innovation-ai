from __future__ import annotations

import hashlib

import pytest

from innovation_ai.innovation.types import PlayerId
from innovation_ai.search import (
    SEARCH_RNG_VERSION,
    SearchDescriptor,
    SearchRngFactory,
    SearchSeedError,
    seed_digest,
)
from innovation_ai.search.contracts import DEFAULT_SAMPLER_SEED_DERIVATION


def test_search_seed_contract_matches_descriptor_and_is_deterministic() -> None:
    descriptor = SearchDescriptor()
    factory = SearchRngFactory("run", 4)
    first = factory.seed_for_decision(
        game_id="game-1",
        chooser=PlayerId.PLAYER_2,
        decision_id=17,
        policy_id="policy",
        search_descriptor_id=descriptor.descriptor_id,
    )
    second = factory.seed_for_decision(
        game_id="game-1",
        chooser=PlayerId.PLAYER_2,
        decision_id=17,
        policy_id="policy",
        search_descriptor_id=descriptor.descriptor_id,
    )

    assert SEARCH_RNG_VERSION == DEFAULT_SAMPLER_SEED_DERIVATION
    assert first == second
    assert len(first) == hashlib.sha256().digest_size
    assert seed_digest(first) == f"sha256:{hashlib.sha256(first).hexdigest()}"


def test_search_seed_is_domain_separated_by_every_route_identity() -> None:
    descriptor = SearchDescriptor()
    baseline = SearchRngFactory(9, 2).seed_for_decision(
        game_id="game",
        chooser=PlayerId.PLAYER_1,
        decision_id=3,
        policy_id="policy",
        search_descriptor_id=descriptor.descriptor_id,
    )
    variants = (
        SearchRngFactory(10, 2).seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=3,
            policy_id="policy",
            search_descriptor_id=descriptor.descriptor_id,
        ),
        SearchRngFactory(9, 3).seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=3,
            policy_id="policy",
            search_descriptor_id=descriptor.descriptor_id,
        ),
        SearchRngFactory(9, 2).seed_for_decision(
            game_id="other",
            chooser=PlayerId.PLAYER_1,
            decision_id=3,
            policy_id="policy",
            search_descriptor_id=descriptor.descriptor_id,
        ),
        SearchRngFactory(9, 2).seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_2,
            decision_id=3,
            policy_id="policy",
            search_descriptor_id=descriptor.descriptor_id,
        ),
        SearchRngFactory(9, 2).seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=4,
            policy_id="policy",
            search_descriptor_id=descriptor.descriptor_id,
        ),
        SearchRngFactory(9, 2).seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=3,
            policy_id="other-policy",
            search_descriptor_id=descriptor.descriptor_id,
        ),
        SearchRngFactory(9, 2).seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=3,
            policy_id="policy",
            search_descriptor_id=SearchDescriptor(route_transition_budget=401).descriptor_id,
        ),
    )
    assert baseline not in variants
    assert len(set(variants)) == len(variants)


def test_search_seed_rejects_wrong_contract_version_and_invalid_identity() -> None:
    descriptor = SearchDescriptor()
    assert SearchRngFactory(b"bytes", 0).seed_for_decision(
        game_id="bytes",
        chooser=PlayerId.PLAYER_1,
        decision_id=1,
        policy_id="policy",
        search_descriptor_id=descriptor.descriptor_id,
    )
    with pytest.raises(SearchSeedError, match="non-empty bytes"):
        seed_digest(b"")
    with pytest.raises(SearchSeedError, match="int, string, or bytes"):
        SearchRngFactory(1.5, 0)  # type: ignore[arg-type]
    with pytest.raises(SearchSeedError, match="generation must be an integer"):
        SearchRngFactory(1, True)
    with pytest.raises(SearchSeedError, match="generation cannot be negative"):
        SearchRngFactory(1, -1)
    with pytest.raises(SearchSeedError, match="unsupported"):
        SearchRngFactory(1, 0, version="other")
    with pytest.raises(SearchSeedError, match="boolean"):
        SearchRngFactory(True, 0)
    factory = SearchRngFactory(1, 0)
    with pytest.raises(SearchSeedError, match="game ID"):
        factory.seed_for_decision(
            game_id="",
            chooser=PlayerId.PLAYER_1,
            decision_id=1,
            policy_id="policy",
            search_descriptor_id=descriptor.descriptor_id,
        )
    with pytest.raises(SearchSeedError, match="decision ID"):
        factory.seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=0,
            policy_id="policy",
            search_descriptor_id=descriptor.descriptor_id,
        )
    with pytest.raises(SearchSeedError, match="policy ID"):
        factory.seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=1,
            policy_id="",
            search_descriptor_id=descriptor.descriptor_id,
        )
    with pytest.raises(SearchSeedError, match="search descriptor ID"):
        factory.seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=1,
            policy_id="policy",
            search_descriptor_id="",
        )
