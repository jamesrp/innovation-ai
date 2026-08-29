from __future__ import annotations

import pytest

from innovation_ai.harness.afterstates import CandidateExpansion, TerminalCandidate
from innovation_ai.harness.policy import CandidateRoute
from innovation_ai.innovation.actions import DrawAction, MeldAction
from innovation_ai.innovation.types import CardId, PlayerId
from innovation_ai.training.selection import (
    ActionValue,
    PolicyRngFactory,
    SelectionDomain,
    aggregate_candidate_values,
    choose_temperature_action,
    select_expansion_action,
)


def test_aggregate_groups_exact_semantic_actions_and_averages_terminal_samples() -> None:
    draw = DrawAction(8)
    meld = MeldAction(8, CardId.from_name("Agriculture"))
    routes = (
        CandidateRoute("game", 8, draw, 0, "value"),
        CandidateRoute("game", 8, meld, 0, "value"),
    )
    terminal_route = CandidateRoute("game", 8, draw, 1, "value")

    # The missing second Meld sample is rejected rather than silently averaging
    # by flattened position index.
    with pytest.raises(ValueError, match="identical sampled-index"):
        aggregate_candidate_values(
            (draw, meld), routes, (0.2, 0.6), terminal_values=((terminal_route, 1.0),)
        )

    complete = aggregate_candidate_values(
        (draw, meld),
        (*routes, CandidateRoute("game", 8, meld, 1, "value")),
        (0.2, 0.6, 0.8),
        terminal_values=((terminal_route, 1.0),),
    )
    assert tuple(item.action for item in complete) == (draw, meld)
    assert tuple(item.mean_value for item in complete) == pytest.approx((0.6, 0.7))


def test_temperature_zero_is_legal_order_argmax_and_terminal_needs_no_evaluator() -> None:
    draw = DrawAction(9)
    meld = MeldAction(9, CardId.from_name("Agriculture"))
    terminal = TerminalCandidate(CandidateRoute("game", 9, draw, 0, "value"), 1.0)
    expansion = CandidateExpansion((), (), (terminal,))

    selection = select_expansion_action(
        policy_id="policy",
        game_id="game",
        decision_id=9,
        legal_actions=(draw,),
        expansion=expansion,
        evaluated_values=(),
        temperature=0.0,
    )
    assert selection.action == draw
    assert selection.mean_value == 1.0

    tied = (ActionValue(draw, (0.5,), 0.5), ActionValue(meld, (0.5,), 0.5))
    assert choose_temperature_action(tied, 0.0).action == draw


def test_policy_rng_is_per_decision_and_rebatch_invariant() -> None:
    factory = PolicyRngFactory(run_seed=404, generation=3)
    first = factory.for_decision(
        game_id="same-game",
        chooser=PlayerId.PLAYER_1,
        decision_id=12,
        domain=SelectionDomain.TEMPERATURE,
        policy_id="policy",
    )
    # Deriving another game's stream between repeated derivations cannot advance
    # the first stream or change its categorical random source.
    factory.for_decision(
        game_id="other-game",
        chooser=PlayerId.PLAYER_2,
        decision_id=55,
        domain=SelectionDomain.DETERMINIZATION,
        policy_id="other-policy",
    )
    repeated = factory.for_decision(
        game_id="same-game",
        chooser=PlayerId.PLAYER_1,
        decision_id=12,
        domain=SelectionDomain.TEMPERATURE,
        policy_id="policy",
    )
    assert first.unit_interval() == repeated.unit_interval()
