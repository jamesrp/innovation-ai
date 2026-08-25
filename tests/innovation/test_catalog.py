from __future__ import annotations

from collections import Counter
from importlib import resources
from pathlib import Path

import pytest

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    IconSlot,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)

ROOT = Path(__file__).parents[2]
EXPECTED_FINGERPRINT = "sha256:23fc84b70f401bba3e8d0abaaad0c28978cdaf076aef45e9e0cfc4b5810d5e86"


def test_stable_value_types_and_card_id_normalization() -> None:
    assert tuple(PlayerId) == (PlayerId.PLAYER_1, PlayerId.PLAYER_2)
    assert len(Color) == 5
    assert len(Icon) == 6
    assert len(IconSlot) == 4
    assert tuple(SplayDirection) == (
        SplayDirection.NONE,
        SplayDirection.LEFT,
        SplayDirection.RIGHT,
        SplayDirection.UP,
    )
    assert len(NormalAchievementId) == 9
    assert len(SpecialAchievementId) == 5
    assert CardId.from_name("A.I.") == CardId("a-i")
    assert CardId.from_name(" THE WHEEL ") == CardId("the-wheel")
    assert CardId.from_name("CITY STATES") == CardId("city-states")
    assert CardId.from_name("Publication") == CardId("publications")
    assert str(CardId("the-wheel")) == "the-wheel"
    assert DogmaEffectId(CardId("pottery"), 2).ordinal == 2

    with pytest.raises(ValueError, match="invalid canonical card ID"):
        CardId("A.I.")
    with pytest.raises(ValueError, match="ordinal must be 1-3"):
        DogmaEffectId(CardId("pottery"), 0)


def test_registry_has_complete_expected_histograms() -> None:
    registry = load_card_registry()

    assert len(registry.cards) == 105
    assert len(registry.by_id) == 105
    assert Counter(card.age for card in registry.cards) == Counter(
        {1: 15, **{age: 10 for age in range(2, 11)}}
    )
    assert Counter(card.color for card in registry.cards) == Counter({color: 21 for color in Color})
    assert Counter(len(card.dogma_effects) for card in registry.cards) == Counter(
        {1: 55, 2: 47, 3: 3}
    )
    assert {card.name for card in registry.cards if len(card.dogma_effects) == 3} == {
        "COAL",
        "SATELLITES",
        "THE INTERNET",
    }
    assert registry.card("POTTERY").dogma_effects[1].id == DogmaEffectId(CardId("pottery"), 2)
    assert registry.card("POTTERY").dogma_effects[1].text == "Draw a 1."


def test_every_card_has_frozen_face_geometry() -> None:
    registry = load_card_registry()

    for card in registry.cards:
        assert len(card.functional_icons) == 3
        assert card.icon_at(card.image_slot) is None
        assert sum(card.icon_at(slot) is None for slot in IconSlot) == 1
        assert card.featured_icon in Icon

    archery = registry.card("ARCHERY")
    assert archery.icon_at(IconSlot.TOP_LEFT) is Icon.CASTLE
    assert archery.icon_at(IconSlot.BOTTOM_LEFT) is Icon.LIGHTBULB
    assert archery.icon_at(IconSlot.BOTTOM_CENTER) is None
    assert archery.icon_at(IconSlot.BOTTOM_RIGHT) is Icon.CASTLE


def test_registry_lookup_is_semantic_and_immutable() -> None:
    registry = load_card_registry()

    assert registry.card(CardId("a-i")).name == "A.I."
    assert registry.card("Publication").name == "PUBLICATIONS"
    assert registry.card("PUBLICATIONS").id == CardId("publications")
    with pytest.raises(KeyError, match="unknown card"):
        registry.card("not a card")
    with pytest.raises(TypeError):
        registry.by_id[CardId("new-card")] = registry.cards[0]  # type: ignore[index]


def test_special_predicates_and_linked_routes_are_separate() -> None:
    registry = load_card_registry()

    assert set(registry.special_achievements) == set(SpecialAchievementId)
    assert set(registry.linked_achievement_routes) == set(SpecialAchievementId)
    monument = registry.special_achievements[SpecialAchievementId.MONUMENT]
    monument_route = registry.linked_achievement_routes[SpecialAchievementId.MONUMENT]
    assert monument.source_condition == "Tuck six or Score six in one turn"
    assert monument_route.source_card_id == CardId("masonry")
    assert monument_route.source_effect_id.ordinal == 1
    assert registry.card(monument_route.source_card_id).name == "MASONRY"


def test_packaged_reference_data_is_byte_identical_to_supplied_data() -> None:
    package = resources.files("innovation_ai.innovation.data")
    for filename in ("cards.csv", "special_achievements.csv"):
        supplied = (ROOT / "game-rules-plaintext" / filename).read_bytes()
        assert package.joinpath(filename).read_bytes() == supplied


def test_data_fingerprint_is_stable() -> None:
    assert load_card_registry().data_fingerprint == EXPECTED_FINGERPRINT
