"""EMPIRICISM - choose an unordered pair of colors, draw and reveal a 9, meld a
matching draw and optionally splay that color up; twenty visible bulbs wins immediately.

The ten branch choices are the ten legal unordered pairs, so selecting the same color twice is
not representable in the program.
"""

from __future__ import annotations

from typing import Final

from innovation_ai.innovation.effects.program import (
    EXECUTOR,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    Cmp,
    ConditionNode,
    DrawNode,
    EffectProgram,
    LetNode,
    MovementKind,
    MoveNode,
    Predicate,
    ProgramEffect,
    RevealNode,
    SequenceNode,
    SplayNode,
    ValueRef,
    WinNode,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, Icon, SplayDirection

CARD_ID: Final[CardId] = CardId("empiricism")

_COLOR_PAIRS: Final[tuple[tuple[str, tuple[Color, Color]], ...]] = (
    ("blue-green", (Color.BLUE, Color.GREEN)),
    ("blue-purple", (Color.BLUE, Color.PURPLE)),
    ("blue-red", (Color.BLUE, Color.RED)),
    ("blue-yellow", (Color.BLUE, Color.YELLOW)),
    ("green-purple", (Color.GREEN, Color.PURPLE)),
    ("green-red", (Color.GREEN, Color.RED)),
    ("green-yellow", (Color.GREEN, Color.YELLOW)),
    ("purple-red", (Color.PURPLE, Color.RED)),
    ("purple-yellow", (Color.PURPLE, Color.YELLOW)),
    ("red-yellow", (Color.RED, Color.YELLOW)),
)

_PAIR_DISPATCH: Final[tuple[ConditionNode, ...]] = tuple(
    ConditionNode(
        f"dispatch-{pair_id}",
        Predicate.equals("color-pair", pair_id),
        f"match-{pair_id}",
        f"dispatch-{_COLOR_PAIRS[index + 1][0]}" if index + 1 < len(_COLOR_PAIRS) else None,
    )
    for index, (pair_id, _colors) in enumerate(_COLOR_PAIRS)
)

_PAIR_MATCHES: Final[tuple[ConditionNode, ...]] = tuple(
    ConditionNode(
        f"match-{pair_id}",
        Predicate.card_color_in("drawn-nine", colors),
        "matched-nine",
    )
    for pair_id, colors in _COLOR_PAIRS
)

EFFECTS: Final[EffectProgram] = EffectProgram(
    "empiricism-v1",
    CARD_ID,
    (
        ProgramEffect(DogmaEffectId(CARD_ID, 1), False, "empiricism-reveal"),
        ProgramEffect(DogmaEffectId(CARD_ID, 2), False, "empiricism-win"),
    ),
    (
        SequenceNode(
            "empiricism-reveal",
            ("choose-color-pair", "draw-nine", "reveal-nine", "dispatch-blue-green"),
        ),
        ChoiceNode(
            "choose-color-pair",
            ChoiceKind.BRANCH,
            "color-pair",
            chooser=EXECUTOR,
            branches=tuple(pair_id for pair_id, _colors in _COLOR_PAIRS),
        ),
        DrawNode("draw-nine", ValueRef.literal(9), "drawn-nine", player=EXECUTOR),
        RevealNode("reveal-nine", CardSelector.from_variable("drawn-nine")),
        *_PAIR_DISPATCH,
        *_PAIR_MATCHES,
        SequenceNode(
            "matched-nine",
            ("bind-drawn-color", "meld-nine", "choose-splay", "if-splay"),
        ),
        LetNode("bind-drawn-color", "drawn-color", color_of="drawn-nine"),
        MoveNode(
            "meld-nine",
            MovementKind.MELD,
            CardSelector.from_variable("drawn-nine"),
            destination_player=EXECUTOR,
        ),
        ChoiceNode(
            "choose-splay",
            ChoiceKind.BRANCH,
            "splay-choice",
            chooser=EXECUTOR,
            branches=("splay",),
            optional=True,
        ),
        ConditionNode("if-splay", Predicate.truthy("splay-choice"), "splay-drawn-color"),
        SplayNode(
            "splay-drawn-color",
            EXECUTOR,
            color_variable="drawn-color",
            direction=SplayDirection.UP,
        ),
        ConditionNode(
            "empiricism-win",
            Predicate.count(
                ValueRef.icon_count(Icon.LIGHTBULB, EXECUTOR),
                Cmp.GE,
                ValueRef.literal(20),
            ),
            "win",
        ),
        WinNode("win", player=EXECUTOR),
    ),
)
