"""Rules decision 13: two-stage hidden-zone choices and the deterministic fallback.

When the executor cannot legally inspect the affected zone, the choice splits: the executor fixes
the public projection (a value), then the zone's owner disambiguates identities without seeing the
alternatives. When the executor *is* the owner, both stages collapse into one ordinary choice.
"""

from __future__ import annotations

import pytest

from innovation_ai.innovation.actions import ChooseCardAction, ChooseValueAction
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import (
    ACTIVATOR,
    EXECUTOR,
    OPPONENT,
    CardSelector,
    ChoiceKind,
    ChoiceNode,
    EffectContext,
    EffectInvariantError,
    EffectProgram,
    EffectProgramRegistry,
    EffectStatus,
    Extreme,
    ExtremeScope,
    MovementKind,
    MoveNode,
    ProgramEffect,
    RevealNode,
    SequenceNode,
    current_effect_decision,
    start_effect,
    submit_effect_action,
)
from innovation_ai.innovation.serialization import dumps_state, loads_state
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GameState,
    build_explicit_state,
    state_hash,
)
from innovation_ai.innovation.types import CardId, Color, DogmaEffectId, PlayerId
from innovation_ai.innovation.zones import ZoneKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
CARD = CardId("rocketry")


def _program(*, owner_disambiguates: bool) -> EffectProgram:
    """A "return a card in the opponent's score pile" shape, the canonical hidden-zone case.

    ``owner_disambiguates`` toggles whether the node names an explicit owner or relies on the
    selector's own player reference.
    """

    return EffectProgram(
        "hidden-choice-v1",
        CARD,
        (ProgramEffect(DogmaEffectId(CARD, 1), False, "sequence"),),
        (
            SequenceNode("sequence", ("choose", "return")),
            ChoiceNode(
                "choose",
                ChoiceKind.HIDDEN_CARD,
                "target",
                chooser=EXECUTOR,
                cards=CardSelector.score(OPPONENT),
                owner=OPPONENT if owner_disambiguates else None,
            ),
            MoveNode("return", MovementKind.RETURN, CardSelector.from_variable("target")),
        ),
    )


PROGRAMS = EffectProgramRegistry((_program(owner_disambiguates=True),))


def _context(state: GameState) -> EffectContext:
    return EffectContext(
        actor=P1,
        chooser=P1,
        executor=P1,
        dogma_activator=P1,
        source_card_id=CARD,
        source_effect_id=None,
        turn_id=state.turn_number,
        dogma_action_id=1,
    )


def _state(opponent_score: tuple[str, ...]) -> GameState:
    return build_explicit_state(
        REGISTRY,
        positions=(
            (P1, ExplicitPlayerPosition(board=((Color.BLUE, (CardId("rocketry"),)),))),
            (
                P2,
                ExplicitPlayerPosition(
                    board=((Color.BLUE, (CardId("pottery"),)),),
                    score_pile=tuple(CardId(name) for name in opponent_score),
                ),
            ),
        ),
    )


def test_the_executor_first_chooses_the_public_value() -> None:
    state = _state(("tools", "canal-building", "construction"))
    started = start_effect(state, "hidden-choice-v1", _context(state), PROGRAMS, REGISTRY)
    assert started.status is EffectStatus.AWAIT_DECISION
    decision = started.decision
    assert decision is not None
    # Stage one is a value choice made by the executor, who cannot see the identities.
    assert decision.chooser is P1
    assert all(isinstance(action, ChooseValueAction) for action in decision.legal_actions)
    assert {
        action.value for action in decision.legal_actions if isinstance(action, ChooseValueAction)
    } == {
        1,
        2,
    }


def test_the_zone_owner_then_disambiguates_without_seeing_alternatives() -> None:
    state = _state(("tools", "canal-building", "construction"))
    started = start_effect(state, "hidden-choice-v1", _context(state), PROGRAMS, REGISTRY)
    assert started.decision is not None
    value = next(
        action
        for action in started.decision.legal_actions
        if isinstance(action, ChooseValueAction) and action.value == 2
    )
    second = submit_effect_action(started.state, value, PROGRAMS, REGISTRY)
    assert second.status is EffectStatus.AWAIT_DECISION
    decision = second.decision
    assert decision is not None
    # Stage two belongs to the score pile's owner, not the executor.
    assert decision.chooser is P2
    assert decision.executor is P1
    offered = {
        action.card_id for action in decision.legal_actions if isinstance(action, ChooseCardAction)
    }
    # Only the cards matching the publicly fixed value are offered.
    assert offered == {CardId("canal-building"), CardId("construction")}

    final = submit_effect_action(
        second.state,
        next(
            action
            for action in decision.legal_actions
            if isinstance(action, ChooseCardAction) and action.card_id == CardId("construction")
        ),
        PROGRAMS,
        REGISTRY,
    )
    assert final.status is EffectStatus.COMPLETE
    assert CardId("construction") not in final.state.player(P2).score_pile
    assert CardId("canal-building") in final.state.player(P2).score_pile


def test_a_single_matching_card_needs_no_disambiguation_stage() -> None:
    state = _state(("tools", "canal-building"))
    started = start_effect(state, "hidden-choice-v1", _context(state), PROGRAMS, REGISTRY)
    assert started.decision is not None
    value = next(
        action
        for action in started.decision.legal_actions
        if isinstance(action, ChooseValueAction) and action.value == 1
    )
    result = submit_effect_action(started.state, value, PROGRAMS, REGISTRY)
    # Exactly one card has value 1, so the owner has nothing to choose.
    assert result.status is EffectStatus.COMPLETE
    assert CardId("tools") not in result.state.player(P2).score_pile


def test_a_single_candidate_overall_resolves_with_no_decision_at_all() -> None:
    """Decision 13's fallback: when nobody can choose, take the only card."""

    state = _state(("tools",))
    result = start_effect(state, "hidden-choice-v1", _context(state), PROGRAMS, REGISTRY)
    assert result.status is EffectStatus.COMPLETE
    assert not result.state.player(P2).score_pile


def test_an_empty_hidden_zone_performs_nothing_and_raises_nothing() -> None:
    state = _state(())
    result = start_effect(state, "hidden-choice-v1", _context(state), PROGRAMS, REGISTRY)
    assert result.status is EffectStatus.COMPLETE
    assert result.qualifying_changes == 0


def test_swapping_two_equal_value_hidden_cards_yields_the_same_first_stage() -> None:
    """The executor's public choice must not depend on hidden identities."""

    first = _state(("canal-building", "construction"))
    second = _state(("canal-building", "currency"))
    started_first = start_effect(first, "hidden-choice-v1", _context(first), PROGRAMS, REGISTRY)
    started_second = start_effect(second, "hidden-choice-v1", _context(second), PROGRAMS, REGISTRY)
    assert started_first.decision is not None and started_second.decision is not None
    assert started_first.decision.legal_actions == started_second.decision.legal_actions, (
        "a hidden swap changed the executor's public options"
    )


def test_the_two_stage_choice_round_trips_between_stages() -> None:
    state = _state(("canal-building", "construction"))
    started = start_effect(state, "hidden-choice-v1", _context(state), PROGRAMS, REGISTRY)
    assert started.decision is not None
    restored = loads_state(dumps_state(started.state), REGISTRY)
    assert state_hash(restored) == state_hash(started.state)
    assert current_effect_decision(restored, PROGRAMS, REGISTRY) == started.decision

    value = next(
        action for action in started.decision.legal_actions if isinstance(action, ChooseValueAction)
    )
    direct = submit_effect_action(started.state, value, PROGRAMS, REGISTRY)
    resumed = submit_effect_action(restored, value, PROGRAMS, REGISTRY)
    assert state_hash(direct.state) == state_hash(resumed.state)
    mid = loads_state(dumps_state(direct.state), REGISTRY)
    assert state_hash(mid) == state_hash(direct.state)
    assert current_effect_decision(mid, PROGRAMS, REGISTRY) == direct.decision


def test_an_owner_who_is_also_the_executor_gets_one_direct_choice() -> None:
    """Decision 13: a demand victim chooses exact cards from their own zone directly."""

    own_hand = EffectProgram(
        "own-hand-choice-v1",
        CARD,
        (ProgramEffect(DogmaEffectId(CARD, 1), False, "sequence"),),
        (
            SequenceNode("sequence", ("choose", "transfer")),
            ChoiceNode(
                "choose",
                ChoiceKind.HIDDEN_CARD,
                "target",
                chooser=EXECUTOR,
                cards=CardSelector.hand(
                    EXECUTOR, extreme=Extreme.HIGHEST, extreme_scope=ExtremeScope.ONE_TIED
                ),
                owner=EXECUTOR,
            ),
            MoveNode(
                "transfer",
                MovementKind.TRANSFER,
                CardSelector.from_variable("target"),
                destination_player=ACTIVATOR,
                destination_zone=ZoneKind.HAND,
            ),
        ),
    )
    programs = EffectProgramRegistry((own_hand,))
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (
                P1,
                ExplicitPlayerPosition(
                    board=((Color.BLUE, (CardId("rocketry"),)),),
                    hand=(CardId("canal-building"), CardId("construction")),
                ),
            ),
            (P2, ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),))),
        ),
    )
    started = start_effect(state, "own-hand-choice-v1", _context(state), programs, REGISTRY)
    assert started.decision is not None
    # One stage only: the owner may inspect the zone, so no public projection is needed.
    assert all(isinstance(action, ChooseCardAction) for action in started.decision.legal_actions)
    assert len(started.decision.legal_actions) == 2


def test_a_hidden_choice_without_an_explicit_owner_uses_the_selectors_player() -> None:
    programs = EffectProgramRegistry((_program(owner_disambiguates=False),))
    state = _state(("canal-building", "construction"))
    started = start_effect(state, "hidden-choice-v1", _context(state), programs, REGISTRY)
    assert started.decision is not None
    assert all(isinstance(action, ChooseValueAction) for action in started.decision.legal_actions)
    value = started.decision.legal_actions[0]
    second = submit_effect_action(started.state, value, programs, REGISTRY)
    assert second.decision is not None
    assert second.decision.chooser is P2


def test_a_hidden_choice_collapses_to_direct_ids_when_the_chooser_can_see_them() -> None:
    visible_hidden_choice = EffectProgram(
        "visible-hidden-choice-v1",
        CARD,
        (ProgramEffect(DogmaEffectId(CARD, 1), False, "sequence"),),
        (
            SequenceNode("sequence", ("reveal", "choose")),
            RevealNode("reveal", CardSelector.hand(OPPONENT)),
            ChoiceNode(
                "choose",
                ChoiceKind.HIDDEN_CARD,
                "target",
                chooser=EXECUTOR,
                cards=CardSelector.hand(OPPONENT),
                owner=OPPONENT,
            ),
        ),
    )
    programs = EffectProgramRegistry((visible_hidden_choice,))
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (P1, ExplicitPlayerPosition(board=((Color.BLUE, (CARD,)),))),
            (
                P2,
                ExplicitPlayerPosition(
                    board=((Color.BLUE, (CardId("pottery"),)),),
                    hand=(CardId("canal-building"), CardId("construction")),
                ),
            ),
        ),
    )

    started = start_effect(
        state, visible_hidden_choice.program_id, _context(state), programs, REGISTRY
    )
    assert started.decision is not None
    assert started.decision.chooser is P1
    assert all(isinstance(action, ChooseCardAction) for action in started.decision.legal_actions)


def test_an_exact_card_choice_cannot_disclose_an_unrevealed_opponent_zone() -> None:
    leaking = EffectProgram(
        "leaking-choice-v1",
        CARD,
        (ProgramEffect(DogmaEffectId(CARD, 1), False, "choose"),),
        (
            ChoiceNode(
                "choose",
                ChoiceKind.CARD,
                "target",
                chooser=EXECUTOR,
                cards=CardSelector.score(OPPONENT),
            ),
        ),
    )
    programs = EffectProgramRegistry((leaking,))
    state = _state(("canal-building", "construction"))

    with pytest.raises(EffectInvariantError, match="hidden card identities"):
        start_effect(state, leaking.program_id, _context(state), programs, REGISTRY)


def test_an_exact_choice_may_use_cards_revealed_from_an_opponents_hand() -> None:
    revealed_choice = EffectProgram(
        "revealed-choice-v1",
        CARD,
        (ProgramEffect(DogmaEffectId(CARD, 1), False, "sequence"),),
        (
            SequenceNode("sequence", ("reveal", "choose")),
            RevealNode("reveal", CardSelector.hand(OPPONENT)),
            ChoiceNode(
                "choose",
                ChoiceKind.CARD,
                "target",
                chooser=EXECUTOR,
                cards=CardSelector.hand(OPPONENT),
            ),
        ),
    )
    programs = EffectProgramRegistry((revealed_choice,))
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (P1, ExplicitPlayerPosition(board=((Color.BLUE, (CARD,)),))),
            (
                P2,
                ExplicitPlayerPosition(
                    board=((Color.BLUE, (CardId("pottery"),)),),
                    hand=(CardId("canal-building"), CardId("construction")),
                ),
            ),
        ),
    )

    started = start_effect(state, revealed_choice.program_id, _context(state), programs, REGISTRY)
    assert started.decision is not None
    assert started.decision.chooser is P1
    assert {
        action.card_id
        for action in started.decision.legal_actions
        if isinstance(action, ChooseCardAction)
    } == {CardId("canal-building"), CardId("construction")}


def test_only_a_newly_public_reveal_qualifies_for_sharing() -> None:
    repeated_reveal = EffectProgram(
        "repeated-reveal-v1",
        CARD,
        (ProgramEffect(DogmaEffectId(CARD, 1), False, "sequence"),),
        (
            SequenceNode("sequence", ("first", "second")),
            RevealNode("first", CardSelector.hand(OPPONENT)),
            RevealNode("second", CardSelector.hand(OPPONENT)),
        ),
    )
    programs = EffectProgramRegistry((repeated_reveal,))
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (P1, ExplicitPlayerPosition(board=((Color.BLUE, (CARD,)),))),
            (
                P2,
                ExplicitPlayerPosition(
                    board=((Color.BLUE, (CardId("pottery"),)),),
                    hand=(CardId("tools"),),
                ),
            ),
        ),
    )

    result = start_effect(state, repeated_reveal.program_id, _context(state), programs, REGISTRY)
    reveals = tuple(event for event in result.events if event.kind.value == "reveal")
    assert len(reveals) == 1
    assert reveals[0].changed
    assert result.qualifying_changes == 1
    assert result.state.revealed == ()


def test_revealing_an_already_public_top_card_gives_no_credit() -> None:
    public_reveal = EffectProgram(
        "public-reveal-v1",
        CARD,
        (ProgramEffect(DogmaEffectId(CARD, 1), False, "reveal"),),
        (RevealNode("reveal", CardSelector.top_cards(OPPONENT)),),
    )
    programs = EffectProgramRegistry((public_reveal,))
    state = _state(())
    result = start_effect(state, public_reveal.program_id, _context(state), programs, REGISTRY)
    assert result.status is EffectStatus.COMPLETE
    assert result.events == ()
    assert result.qualifying_changes == 0
    assert result.state.revealed == ()


def test_a_hidden_choice_cannot_be_answered_with_the_wrong_stage_action() -> None:
    from innovation_ai.innovation.effects import IllegalEffectAction

    state = _state(("canal-building", "construction"))
    started = start_effect(state, "hidden-choice-v1", _context(state), PROGRAMS, REGISTRY)
    assert started.decision is not None
    with pytest.raises(IllegalEffectAction):
        submit_effect_action(
            started.state,
            ChooseCardAction(started.decision.decision_id, CardId("construction")),
            PROGRAMS,
            REGISTRY,
        )
