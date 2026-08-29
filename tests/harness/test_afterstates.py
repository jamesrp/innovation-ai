from __future__ import annotations

from support.scenario import scenario

from innovation_ai.harness.afterstates import TrustedCandidateExpander, terminal_utility
from innovation_ai.innovation.actions import DrawAction
from innovation_ai.innovation.protocol import apply_action
from innovation_ai.innovation.types import PlayerId
from innovation_ai.training.determinizations import InformationSetSampler, InformationSetSpecBuilder


def test_terminal_candidate_bypasses_positions_and_uses_exact_original_viewer_utility() -> None:
    state = scenario().active(PlayerId.PLAYER_1, paid_actions=1).exhaust_supply().build()
    spec = InformationSetSpecBuilder().build(state)
    sampled = InformationSetSampler(seed=901).sample(spec)
    assert sampled is not None
    draw = next(action for action in spec.legal_actions if isinstance(action, DrawAction))
    exact = apply_action(sampled, draw)
    assert exact.terminal is not None

    expansion = TrustedCandidateExpander().expand(
        spec,
        (sampled,),
        game_id="terminal-game",
        evaluator_key="frozen-terminal",
    )

    terminal = next(item for item in expansion.terminal_candidates if item.route.action == draw)
    assert terminal.utility == terminal_utility(exact.terminal, spec.chooser)
    assert all(route.action != draw for route in expansion.routes)
