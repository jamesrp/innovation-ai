from __future__ import annotations

import random
from dataclasses import replace

import pytest

from innovation_ai.agents import (
    Agent,
    RandomAgent,
    ScriptedAgent,
    ScriptedAgentError,
    ScriptedIllegalActionError,
    ScriptExhaustedError,
    SimpleHeuristicAgent,
)
from innovation_ai.agents.descriptors import (
    SAMPLED_MINIMAX_AGENT_DESCRIPTOR,
    SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
)
from innovation_ai.innovation.actions import (
    AchieveAction,
    ChooseBranchAction,
    ChooseStartingMeldAction,
    Decision,
    DecisionKind,
    DeclineAction,
    DogmaAction,
    DrawAction,
    MeldAction,
)
from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.protocol import apply_action, current_decision
from innovation_ai.innovation.state import build_setup_state
from innovation_ai.innovation.types import CardId, NormalAchievementId
from innovation_ai.search.contracts import PRODUCTION_SEARCH_DESCRIPTOR


def test_sampled_minimax_descriptor_references_production_search_identity() -> None:
    assert SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.parameters == ()
    assert SAMPLED_MINIMAX_AGENT_DESCRIPTOR.parameters == (
        ("search_descriptor_id", PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id),
    )
    assert tuple(key for key, _ in SAMPLED_MINIMAX_AGENT_DESCRIPTOR.parameters) == tuple(
        sorted(key for key, _ in SAMPLED_MINIMAX_AGENT_DESCRIPTOR.parameters)
    )


def _first_turn_decision(seed: int = 801) -> Decision:
    state = build_setup_state(seed)
    for _ in range(2):
        decision = current_decision(state)
        assert decision is not None
        state = apply_action(state, decision.legal_actions[0]).state
    decision = current_decision(state)
    assert decision is not None
    return decision


def test_random_agent_is_seeded_isolated_and_satisfies_protocol() -> None:
    decision = _first_turn_decision()
    first = RandomAgent(19)
    second = RandomAgent(19)
    assert isinstance(first, Agent)

    random.seed(444)
    expected_global = random.random()
    random.seed(444)
    chosen_first = tuple(first.choose_action(decision) for _ in range(8))
    assert random.random() == expected_global
    chosen_second = tuple(second.choose_action(decision) for _ in range(8))

    assert chosen_first == chosen_second
    assert all(action in decision.legal_actions for action in chosen_first)
    assert first.seed == 19


def test_scripted_agent_consumes_exact_actions_and_selectors() -> None:
    decision = _first_turn_decision(802)
    exact = decision.legal_actions[0]
    scripted = ScriptedAgent((exact, lambda pending: pending.legal_actions[-1]))

    assert scripted.remaining == 2
    assert scripted.choose_action(decision) == exact
    assert scripted.choose_action(decision) == decision.legal_actions[-1]
    assert scripted.remaining == 0
    scripted.assert_consumed()
    with pytest.raises(ScriptExhaustedError, match="exhausted"):
        scripted.choose_action(decision)


def test_scripted_agent_rejects_illegal_and_unconsumed_fixture_steps() -> None:
    decision = _first_turn_decision(803)
    illegal = DrawAction(decision.decision_id + 100)
    scripted = ScriptedAgent((illegal, decision.legal_actions[0]))
    with pytest.raises(ScriptedIllegalActionError, match="illegal"):
        scripted.choose_action(decision)
    with pytest.raises(ScriptedAgentError, match="unconsumed"):
        scripted.assert_consumed()


def test_simple_heuristic_setup_effect_and_paid_action_priorities() -> None:
    agent = SimpleHeuristicAgent()
    setup = current_decision(build_setup_state(804))
    assert setup is not None
    setup_choices = tuple(
        action for action in setup.legal_actions if isinstance(action, ChooseStartingMeldAction)
    )
    expected_setup = min(setup_choices, key=lambda action: str(action.card_id))
    assert agent.choose_action(setup) == expected_setup

    paid = _first_turn_decision(805)
    achieve = AchieveAction(paid.decision_id, NormalAchievementId.AGE_1)
    with_achievement = replace(paid, legal_actions=(*paid.legal_actions, achieve))
    assert agent.choose_action(with_achievement) == achieve

    meld_low = MeldAction(paid.decision_id, CardId("writing"))
    meld_high = MeldAction(paid.decision_id, CardId.from_name("A.I."))
    meld_only = replace(paid, legal_actions=(DrawAction(paid.decision_id), meld_low, meld_high))
    assert agent.choose_action(meld_only) == meld_high

    effect = replace(
        paid,
        kind=DecisionKind.EFFECT_CHOICE,
        legal_actions=(
            DeclineAction(paid.decision_id),
            ChooseBranchAction(paid.decision_id, "accept"),
        ),
    )
    assert agent.choose_action(effect) == effect.legal_actions[1]
    decline_only = replace(
        effect,
        legal_actions=(DeclineAction(paid.decision_id),),
    )
    assert agent.choose_action(decline_only) == decline_only.legal_actions[0]


def test_simple_heuristic_uses_visible_icon_advantage_for_dogma() -> None:
    registry = load_card_registry()
    agent = SimpleHeuristicAgent(registry)
    paid = _first_turn_decision(806)
    # The paid decision only offers Dogma for cards whose effects are registered, so the
    # heuristic's icon comparison is exercised with an explicitly named implemented card.
    dogma = DogmaAction(paid.decision_id, CardId("the-wheel"))
    source = registry.card(dogma.card_id)

    opponent = next(
        player for player in paid.observation.players if player.player_id is not paid.chooser
    )
    empty_board = tuple(
        replace(stack, top_card_id=None, covered_cards=(), covered_count=0)
        for stack in opponent.board
    )
    weak_opponent = replace(opponent, board=empty_board)
    observed_players = paid.observation.players
    favorable_observation = replace(
        paid.observation,
        players=(
            weak_opponent
            if observed_players[0].player_id is opponent.player_id
            else observed_players[0],
            weak_opponent
            if observed_players[1].player_id is opponent.player_id
            else observed_players[1],
        ),
    )
    favorable = replace(
        paid,
        observation=favorable_observation,
        legal_actions=(dogma, DrawAction(paid.decision_id)),
    )
    assert agent.choose_action(favorable) == dogma

    booster = max(
        registry.cards,
        key=lambda card: card.functional_icons.count(source.featured_icon),
    )
    strong_board = tuple(
        replace(stack, top_card_id=booster.id, covered_cards=(), covered_count=0)
        for stack in opponent.board
    )
    strong_opponent = replace(opponent, board=strong_board)
    unfavorable_observation = replace(
        paid.observation,
        players=(
            strong_opponent
            if observed_players[0].player_id is opponent.player_id
            else observed_players[0],
            strong_opponent
            if observed_players[1].player_id is opponent.player_id
            else observed_players[1],
        ),
    )
    unfavorable = replace(
        paid,
        observation=unfavorable_observation,
        legal_actions=(dogma, DrawAction(paid.decision_id)),
    )
    assert agent.choose_action(unfavorable) == unfavorable.legal_actions[1]
    dogma_only = replace(unfavorable, legal_actions=(dogma,))
    assert agent.choose_action(dogma_only) == dogma

    branch_only = replace(
        paid,
        legal_actions=(ChooseBranchAction(paid.decision_id, "fallback"),),
    )
    assert agent.choose_action(branch_only) == branch_only.legal_actions[0]
