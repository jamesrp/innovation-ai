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
    arguments = {
        "game_id": "game-1",
        "chooser": PlayerId.PLAYER_2,
        "decision_id": 17,
        "policy_id": "policy",
        "search_descriptor_id": descriptor.descriptor_id,
    }

    first = factory.seed_for_decision(**arguments)
    second = factory.seed_for_decision(**arguments)

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
    with pytest.raises(SearchSeedError, match="unsupported"):
        SearchRngFactory(1, 0, version="other")
    with pytest.raises(SearchSeedError, match="boolean"):
        SearchRngFactory(True, 0)  # type: ignore[arg-type]
    with pytest.raises(SearchSeedError, match="policy ID"):
        SearchRngFactory(1, 0).seed_for_decision(
            game_id="game",
            chooser=PlayerId.PLAYER_1,
            decision_id=1,
            policy_id="",
            search_descriptor_id=SearchDescriptor().descriptor_id,
        )
