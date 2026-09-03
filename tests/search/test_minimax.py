from __future__ import annotations

from dataclasses import dataclass, replace

from innovation_ai.innovation.actions import (
    ChooseStartingMeldAction,
    Decision,
    DecisionKind,
    DeclineAction,
    DrawAction,
    SemanticAction,
)
from innovation_ai.innovation.protocol import apply_action, current_decisions
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    TerminalReason,
    TerminalResult,
    build_setup_state,
)
from innovation_ai.innovation.types import CardId, PlayerId
from innovation_ai.search import (
    DeterministicSampledMinimax,
    InformationSetSampler,
    InformationSetSpec,
    InformationSetSpecBuilder,
    MinimaxSelection,
    SearchDescriptor,
    SearchHooks,
    action_key,
)


@dataclass
class _Graph:
    spec: InformationSetSpec
    decisions: dict[int, Decision]
    edges: dict[tuple[int, str], tuple[int, int]]
    values: dict[int, float]
    terminals: dict[int, tuple[PlayerId, ...]]
    digest_aliases: dict[int, str] | None = None

    def pending(self, state: GameState) -> tuple[Decision, ...]:
        decision = self.decisions.get(state.next_event_id)
        return () if decision is None else (decision,)

    def apply(self, state: GameState, action: SemanticAction) -> GameState:
        target, completed = self.edges[(state.next_event_id, action_key(action))]
        if target in self.terminals:
            return replace(
                state,
                next_event_id=target,
                phase=GamePhase.TERMINAL,
                terminal_result=TerminalResult(TerminalReason.CARD_EFFECT, self.terminals[target]),
            )
        return replace(
            state,
            next_event_id=target,
            turn_number=state.turn_number + completed,
        )

    def evaluate(self, state: GameState, root: PlayerId) -> float:
        del root
        return self.values.get(state.next_event_id, 0.0)

    def digest(self, state: GameState) -> str:
        if self.digest_aliases is not None and state.next_event_id in self.digest_aliases:
            return self.digest_aliases[state.next_event_id]
        return f"node:{state.next_event_id}"

    def hooks(self) -> SearchHooks:
        return SearchHooks(self.pending, self.apply, self.digest)


def _starting_fixture(seed: int = 1201) -> tuple[InformationSetSpec, GameState]:
    live = build_setup_state(seed)
    decision = current_decisions(live)[0]
    spec = InformationSetSpecBuilder().build(live, decision)
    sampled = InformationSetSampler(seed=seed + 1).sample(spec)
    assert sampled is not None
    return spec, sampled


def _play_fixture(seed: int = 1301) -> tuple[InformationSetSpec, GameState]:
    live = build_setup_state(seed)
    first = current_decisions(live)[0]
    live = apply_action(live, first.legal_actions[0]).state
    second = current_decisions(live)[0]
    live = apply_action(live, second.legal_actions[0]).state
    decision = current_decisions(live)[0]
    spec = InformationSetSpecBuilder().build(live, decision)
    sampled = InformationSetSampler(seed=seed + 1).sample(spec)
    assert sampled is not None
    return spec, sampled


def _decision(
    template: Decision,
    decision_id: int,
    chooser: PlayerId,
    actions: tuple[SemanticAction, ...],
) -> Decision:
    return Decision(
        decision_id,
        DecisionKind.EFFECT_CHOICE,
        chooser,
        chooser,
        template.observation,
        actions,
    )


def _search(
    graph: _Graph,
    sample: GameState,
    *,
    budget: int = 100,
    root_depth: int = 1,
    opponent_depth: int = 1,
    starting_depth: int = 1,
    samples: tuple[GameState, ...] | None = None,
) -> MinimaxSelection:
    chosen_samples = samples or (sample,)
    descriptor = SearchDescriptor(
        determinization_count=len(chosen_samples),
        route_transition_budget=budget,
        root_turn_horizon=root_depth,
        opponent_turn_horizon=opponent_depth,
        starting_meld_horizon=starting_depth,
    )
    return DeterministicSampledMinimax(
        descriptor, evaluator=graph.evaluate, hooks=graph.hooks()
    ).select(graph.spec, chosen_samples)


def test_action_key_ignores_only_decision_identity() -> None:
    assert action_key(DrawAction(1)) == action_key(DrawAction(99))
    assert action_key(ChooseStartingMeldAction(1, CardId("pottery"))) != action_key(
        ChooseStartingMeldAction(2, CardId("tools"))
    )


def test_horizon_counts_current_partial_turn_and_completed_player_turns() -> None:
    spec, sample = _play_fixture()
    root = replace(sample, next_event_id=100)
    root_decision = replace(current_decisions(sample)[0], legal_actions=spec.legal_actions)
    graph = _Graph(spec, {100: root_decision}, {}, {104: 0.9}, {})
    for index, root_action in enumerate(spec.legal_actions):
        child = 200 + index * 10
        graph.edges[(100, action_key(root_action))] = (child, 0)
        action = DrawAction(800 + index)
        graph.decisions[child] = _decision(
            root_decision, action.decision_id, spec.chooser, (action,)
        )
        # Only the first route needs a four-turn chain; every transition completes exactly one turn.
        for turn in range(4):
            node = child + turn
            action = DrawAction(800 + index + turn)
            graph.decisions[node] = _decision(
                root_decision, action.decision_id, spec.chooser, (action,)
            )
            graph.edges[(node, action_key(action))] = (node + 1, 1)
            graph.values[node + 1] = float(turn + 1) / 10
    result = _search(graph, root, root_depth=4, budget=100)

    assert all(route.completed_turn_depth == 4 for route in result.telemetry.routes)
    assert result.telemetry.routes[0].value == 0.4


def test_minimax_uses_each_decision_chooser_for_nested_max_and_min() -> None:
    spec, sample = _play_fixture(1302)
    root = replace(sample, next_event_id=300)
    root_decision = replace(current_decisions(sample)[0], legal_actions=spec.legal_actions)
    graph = _Graph(spec, {300: root_decision}, {}, {}, {})
    expected = []
    for index, root_action in enumerate(spec.legal_actions):
        opponent_node = 310 + index * 100
        graph.edges[(300, action_key(root_action))] = (opponent_node, 0)
        opponent_actions: tuple[SemanticAction, ...] = (
            DrawAction(900 + index),
            DeclineAction(900 + index),
        )
        opponent = PlayerId.PLAYER_2 if spec.chooser is PlayerId.PLAYER_1 else PlayerId.PLAYER_1
        graph.decisions[opponent_node] = _decision(
            root_decision,
            opponent_actions[0].decision_id,
            opponent,
            opponent_actions,
        )
        branch_maxima = []
        for opponent_index, opponent_action in enumerate(opponent_actions):
            root_node = opponent_node + 10 + opponent_index
            graph.edges[(opponent_node, action_key(opponent_action))] = (root_node, 0)
            root_actions: tuple[SemanticAction, ...] = (
                DrawAction(950 + root_node),
                DeclineAction(950 + root_node),
            )
            graph.decisions[root_node] = _decision(
                root_decision, root_actions[0].decision_id, spec.chooser, root_actions
            )
            leaf_values = (
                0.1 + index * 0.05 + opponent_index * 0.2,
                0.8 - index * 0.1 - opponent_index * 0.1,
            )
            branch_maxima.append(max(leaf_values))
            for leaf_index, nested_action in enumerate(root_actions):
                terminal = root_node + 20 + opponent_index * 10 + leaf_index
                graph.edges[(root_node, action_key(nested_action))] = (terminal, 1)
                graph.terminals[terminal] = ()
                graph.values[terminal] = leaf_values[leaf_index]
        expected.append(min(branch_maxima))
    result = _search(graph, root)

    assert result.telemetry.action_mean_values == tuple(expected)


def test_common_determinizations_use_exact_mean_and_first_tie() -> None:
    spec, sample = _play_fixture(1306)
    samples = (replace(sample, next_event_id=3000), replace(sample, next_event_id=4000))
    root_decision = replace(current_decisions(sample)[0], legal_actions=spec.legal_actions)
    graph = _Graph(spec, {3000: root_decision, 4000: root_decision}, {}, {}, {})
    for sample_index, root in enumerate(samples):
        for action_index, root_action in enumerate(spec.legal_actions):
            terminal = root.next_event_id + 100 + action_index
            graph.edges[(root.next_event_id, action_key(root_action))] = (terminal, 0)
            graph.terminals[terminal] = ()
            # The first two actions have exact means 0.5 in opposite sample order.
            if action_index == 0:
                value = (1.0, 0.0)[sample_index]
            elif action_index == 1:
                value = (0.0, 1.0)[sample_index]
            else:
                value = -1.0
            graph.values[terminal] = value
    result = _search(graph, samples[0], samples=samples)

    assert result.telemetry.action_mean_values[:2] == (0.5, 0.5)
    assert result.telemetry.tied_action_indices == (0, 1)
    assert result.action == spec.legal_actions[0]


def test_equal_route_budgets_discard_an_incomplete_first_iteration() -> None:
    spec, sample = _play_fixture(1303)
    root = replace(sample, next_event_id=500)
    root_decision = replace(current_decisions(sample)[0], legal_actions=spec.legal_actions)
    graph = _Graph(spec, {500: root_decision}, {}, {}, {})
    for index, root_action in enumerate(spec.legal_actions):
        node = 510 + index * 100
        graph.edges[(500, action_key(root_action))] = (node, 0)
        for step in range(4):
            action = DrawAction(1000 + index * 10 + step)
            graph.decisions[node + step] = _decision(
                root_decision, action.decision_id, spec.chooser, (action,)
            )
            graph.edges[(node + step, action_key(action))] = (node + step + 1, 0)
    result = _search(graph, root, budget=2)

    assert {route.engine_transitions for route in result.telemetry.routes} == {2}
    assert all(route.completed_turn_depth == 0 for route in result.telemetry.routes)
    assert all(route.immediate_leaf_fallback for route in result.telemetry.routes)


def test_path_repetition_cuts_to_evaluator_instead_of_a_draw() -> None:
    spec, sample = _play_fixture(1304)
    root = replace(sample, next_event_id=700)
    root_decision = replace(current_decisions(sample)[0], legal_actions=spec.legal_actions)
    graph = _Graph(spec, {700: root_decision}, {}, {702: 0.37}, {}, {701: "cycle", 702: "cycle"})
    assert graph.digest_aliases is not None
    for index, root_action in enumerate(spec.legal_actions):
        first = 701 + index * 10
        second = 702 + index * 10
        action = DrawAction(1100 + index)
        graph.edges[(700, action_key(root_action))] = (first, 0)
        graph.decisions[first] = _decision(
            root_decision, action.decision_id, spec.chooser, (action,)
        )
        graph.edges[(first, action_key(action))] = (second, 0)
        graph.digest_aliases[first] = f"cycle-{index}"
        graph.digest_aliases[second] = f"cycle-{index}"
        graph.values[second] = 0.37 - index
    result = _search(graph, root)

    assert result.telemetry.routes[0].value == 0.37
    assert all(route.repeated_position_cutoffs == 1 for route in result.telemetry.routes)


def test_starting_meld_is_max_over_root_and_min_over_sampled_opponent_hand() -> None:
    spec, sample = _starting_fixture()
    root = replace(sample, next_event_id=1200)
    root_decision = replace(current_decisions(sample)[0], legal_actions=spec.legal_actions)
    opponent = PlayerId.PLAYER_2 if spec.chooser is PlayerId.PLAYER_1 else PlayerId.PLAYER_1
    graph = _Graph(spec, {1200: root_decision}, {}, {}, {})
    matrix = ((0.8, -0.2), (0.1, 0.2))
    for root_index, root_action in enumerate(spec.legal_actions):
        pending = 1210 + root_index
        graph.edges[(1200, action_key(root_action))] = (pending, 0)
        opponent_cards = sample.player(opponent).hand
        responses = tuple(
            ChooseStartingMeldAction(1300 + root_index, card) for card in opponent_cards
        )
        graph.decisions[pending] = Decision(
            1300 + root_index,
            DecisionKind.STARTING_MELD,
            opponent,
            opponent,
            root_decision.observation,
            responses,
        )
        for response_index, response in enumerate(responses):
            terminal = 1400 + root_index * 10 + response_index
            graph.edges[(pending, action_key(response))] = (terminal, 0)
            graph.terminals[terminal] = ()
            graph.values[terminal] = matrix[root_index][response_index]
    result = _search(graph, root, starting_depth=4)

    assert result.action == spec.legal_actions[1]
    assert result.telemetry.action_mean_values == (-0.2, 0.1)
    assert result.statistics.mandatory_setup_transitions == 4
    assert all(route.engine_transitions == 0 for route in result.telemetry.routes)


def test_terminal_value_is_taken_immediately_and_dominates_nonterminal_leaf() -> None:
    spec, sample = _play_fixture(1305)
    root = replace(sample, next_event_id=1500)
    root_decision = replace(current_decisions(sample)[0], legal_actions=spec.legal_actions)
    graph = _Graph(spec, {1500: root_decision}, {}, {}, {})
    for index, root_action in enumerate(spec.legal_actions):
        target = 1510 + index
        graph.edges[(1500, action_key(root_action))] = (target, 0)
        if index == 0:
            graph.terminals[target] = (spec.chooser,)
            graph.values[target] = 2.0
        else:
            graph.values[target] = 1.0
            action = DrawAction(1600 + index)
            graph.decisions[target] = _decision(
                root_decision, action.decision_id, spec.chooser, (action,)
            )
            graph.edges[(target, action_key(action))] = (target + 100, 1)
            graph.values[target + 100] = 1.0
    result = _search(graph, root)

    assert result.action == spec.legal_actions[0]
    assert result.telemetry.action_mean_values[0] == 2.0
