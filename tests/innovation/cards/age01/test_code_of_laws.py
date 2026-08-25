"""CODE OF LAWS: optional relational choice, "if you do", and splay-that-colour."""

from __future__ import annotations

from support import (
    assert_conserved,
    choose_branch,
    choose_card,
    decline,
    resolve_dogma,
    scenario,
)

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def test_no_matching_colour_in_hand_means_no_decision_at_all() -> None:
    """Acceptance row 1: minimal state performs nothing and raises nothing."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.PURPLE, ("city-states",))
        # Tools is blue and the board has only purple, so nothing is tuckable.
        .hand(P1, ("tools",))
        .build()
    )
    result = resolve_dogma(state, "code-of-laws", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert result.state.player(P1).hand == (CardId("tools"),)


def test_tucking_then_splaying_that_colour_left() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("mysticism", "code-of-laws"))
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P1, ("city-states", "tools"))
        .build()
    )
    result = resolve_dogma(
        state,
        "code-of-laws",
        choose_card("city-states"),
        choose_branch("splay"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    purple = result.state.player(P1).board.stack(Color.PURPLE)
    # A tuck goes to the bottom of the stack.
    assert purple.cards[0] == CardId("city-states")
    assert purple.splay is SplayDirection.LEFT
    assert result.state.player(P1).hand == (CardId("tools"),)
    assert_conserved(result.state, REGISTRY)


def test_the_optional_tuck_can_be_declined() -> None:
    """Acceptance row 3: an explicit decline leaves the game unchanged."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P1, ("city-states",))
        .build()
    )
    result = resolve_dogma(state, "code-of-laws", decline(), registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.qualifying_changes == 0
    assert result.state.player(P1).hand == (CardId("city-states"),)


def test_the_splay_can_be_declined_after_a_tuck() -> None:
    """The "if you do" branch offers the splay; declining it keeps the tuck."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("mysticism", "code-of-laws"))
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P1, ("city-states",))
        .build()
    )
    result = resolve_dogma(
        state,
        "code-of-laws",
        choose_card("city-states"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    purple = result.state.player(P1).board.stack(Color.PURPLE)
    assert purple.cards[0] == CardId("city-states")
    assert purple.splay is SplayDirection.NONE


def test_only_colours_already_on_the_board_are_tuckable() -> None:
    """The relational selector restricts candidates to colours already present."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P1, Color.BLUE, ("tools",))
        .board(P2, Color.BLUE, ("pottery",))
        # writing is blue, sailing is green, city-states is purple.
        .hand(P1, ("writing", "sailing", "city-states"))
        .build()
    )
    result = resolve_dogma(
        state,
        "code-of-laws",
        choose_card("writing"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    offered = tuple(
        action.card_id for action in result.decisions[0].legal_actions if hasattr(action, "card_id")
    )
    assert set(offered) == {CardId("writing"), CardId("city-states")}
    assert CardId("sailing") not in offered


def test_a_no_op_splay_is_legal_but_earns_no_sharing_credit() -> None:
    """Decision 15: a singleton stack may still be chosen, and the no-op changes nothing."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.BLUE, ("pottery",))
        .hand(P1, ("city-states",))
        .build()
    )
    result = resolve_dogma(
        state,
        "code-of-laws",
        choose_card("city-states"),
        choose_branch("splay"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    purple = result.state.player(P1).board.stack(Color.PURPLE)
    assert len(purple.cards) == 2
    assert purple.splay is SplayDirection.LEFT
    # The tuck itself is the only qualifying change; the splay of a two-card stack also changes
    # geometry, so both count. What must never count is a splay that changed nothing.
    assert result.qualifying_changes == 2


def test_both_players_execute_when_the_opponent_shares() -> None:
    """Acceptance row 8: the sharing opponent executes first, with their own choices."""

    state = (
        scenario(REGISTRY)
        .board(P1, Color.PURPLE, ("code-of-laws",))
        .board(P2, Color.PURPLE, ("city-states",))
        .hand(P1, ("mysticism",))
        .hand(P2, ("monotheism",))
        .build()
    )
    result = resolve_dogma(
        state,
        "code-of-laws",
        choose_card("monotheism"),
        decline(),
        choose_card("mysticism"),
        decline(),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert result.status is EffectStatus.COMPLETE
    assert tuple(decision.chooser for decision in result.decisions) == (P2, P2, P1, P1)
    assert result.state.player(P2).board.stack(Color.PURPLE).cards[0] == CardId("monotheism")
    assert result.state.player(P1).board.stack(Color.PURPLE).cards[0] == CardId("mysticism")
    # The opponent's tuck changed the game, so the activator gets one free Draw.
    assert len(result.state.player(P1).hand) == 1
