"""Immutable value types shared by the Innovation engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_CARD_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_CARD_ID_ALIASES = {"publication": "publications"}


class PlayerId(StrEnum):
    """Stable identities for the two supported players."""

    PLAYER_1 = "player-1"
    PLAYER_2 = "player-2"


class Color(StrEnum):
    """Innovation card colors."""

    BLUE = "blue"
    GREEN = "green"
    PURPLE = "purple"
    RED = "red"
    YELLOW = "yellow"


class Icon(StrEnum):
    """Functional card icons, using canonical engine names."""

    CASTLE = "castle"
    CROWN = "crown"
    LEAF = "leaf"
    LIGHTBULB = "lightbulb"
    FACTORY = "factory"
    CLOCK = "clock"


class IconSlot(StrEnum):
    """The four meaningful card-image positions in the supplied CSV."""

    TOP_LEFT = "top-left"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"


class SplayDirection(StrEnum):
    """Possible stack splay states."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"


class NormalAchievementId(StrEnum):
    """Stable identities for the nine hidden normal achievements."""

    AGE_1 = "normal-age-1"
    AGE_2 = "normal-age-2"
    AGE_3 = "normal-age-3"
    AGE_4 = "normal-age-4"
    AGE_5 = "normal-age-5"
    AGE_6 = "normal-age-6"
    AGE_7 = "normal-age-7"
    AGE_8 = "normal-age-8"
    AGE_9 = "normal-age-9"


class SpecialAchievementId(StrEnum):
    """Stable identities for the five public special achievements."""

    EMPIRE = "empire"
    MONUMENT = "monument"
    UNIVERSE = "universe"
    WONDER = "wonder"
    WORLD = "world"


@dataclass(frozen=True, slots=True, order=True)
class CardId:
    """A display-name-independent semantic card identifier."""

    value: str

    def __post_init__(self) -> None:
        if _CARD_ID_PATTERN.fullmatch(self.value) is None:
            raise ValueError(f"invalid canonical card ID: {self.value!r}")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_name(cls, name: str) -> CardId:
        """Normalize a printed or clarified card name to its canonical ID."""

        slug = _NON_ALPHANUMERIC.sub("-", name.strip().casefold()).strip("-")
        slug = _CARD_ID_ALIASES.get(slug, slug)
        return cls(slug)


@dataclass(frozen=True, slots=True, order=True)
class DogmaEffectId:
    """Stable identity for one printed effect on a card."""

    card_id: CardId
    ordinal: int

    def __post_init__(self) -> None:
        if not 1 <= self.ordinal <= 3:
            raise ValueError(f"dogma effect ordinal must be 1-3, got {self.ordinal}")
