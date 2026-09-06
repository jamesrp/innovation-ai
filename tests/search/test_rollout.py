from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from innovation_ai.agents.descriptors import AgentDescriptor
from innovation_ai.innovation import Color, ExplicitPlayerPosition, build_explicit_state
from innovation_ai.innovation.actions import (
    ChooseStartingMeldAction,
    Decision,
    DecisionKind,
    DeclineAction,
    DogmaAction,
    DrawAction,
    SemanticAction,
)
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.protocol import apply_action, current_decision, current_decisions
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    SupplyState,
    TerminalReason,
    TerminalResult,
    build_setup_state,
)
from innovation_ai.innovation.types import CardId, PlayerId
from innovation_ai.search import (
    DeterministicSampledRollout,
    InformationSetSampler,
    InformationSetSpec,
    InformationSetSpecBuilder,
    RolloutHooks,
    RolloutSafetyCeilingError,
    RolloutSearchDescriptor,
    RolloutSelection,
    SearchInputError,
    SearchInvariantError,
    action_key,
)

_TEST_IMPLEMENTATION_VERSION = "test-graph-rollout-hooks-v1"
_TEST_EVALUATOR_VERSION = "test-graph-evaluator-v1"
_TEST_CONTINUATION_DESCRIPTOR = AgentDescriptor(
    "test-rollout-continuation",
    "first-or-configured-action-v1",
)


@dataclass(frozen=True)
class _Edge:
    target: int
    turn_delta: int = 0
    phase: GamePhase | None = None
    terminal: bool = False
    winners: tuple[PlayerId, ...] = ()


@dataclass
class _Graph:
    decisions: dict[int, tuple[Decision, ...]]
    edges: dict[tuple[int, str], _Edge]
    values: dict[int, float]

    def __post_init__(self) -> None:
        self.decision_calls: list[int] = []
        self.transition_calls: list[tuple[int, str]] = []
        self.evaluation_calls: list[int] = []

    def pending(self, state: GameState) -> tuple[Decision, ...]:
        self.decision_calls.append(state.next_event_id)
        return self.decisions.get(state.next_event_id, ())

    def transition(self, state: GameState, action: SemanticAction) -> GameState:
        key = action_key(action)
        self.transition_calls.append((state.next_event_id, key))
        edge = self.edges[(state.next_event_id, key)]
        if edge.terminal:
            return replace(
                state,
                next_event_id=edge.target,
                phase=GamePhase.TERMINAL,
                terminal_result=TerminalResult(TerminalReason.CARD_EFFECT, edge.winners),
                turn_number=state.turn_number + edge.turn_delta,
            )
        return replace(
            state,
            next_event_id=edge.target,
            phase=state.phase if edge.phase is None else edge.phase,
            turn_number=state.turn_number + edge.turn_delta,
        )

    def evaluate(self, state: GameState, root_player: PlayerId) -> float:
        del root_player
        self.evaluation_calls.append(state.next_event_id)
        return self.values[state.next_event_id]

    @property
    def version(self) -> str:
        return _TEST_EVALUATOR_VERSION

    def __call__(self, state: GameState, root_player: PlayerId) -> float:
        return self.evaluate(state, root_player)

    def hooks(self, policy: _RecordingPolicy | _IllegalPolicy | None = None) -> RolloutHooks:
        continuation = policy or _RecordingPolicy()
        return RolloutHooks(
            self.pending,
            self.transition,
            _TEST_IMPLEMENTATION_VERSION,
            _TEST_EVALUATOR_VERSION,
            _TEST_CONTINUATION_DESCRIPTOR.descriptor_id,
            choose_continuation=continuation.choose_action,
        )


class _RecordingPolicy:
    def __init__(self, *, last: bool = False) -> None:
        self.last = last
        self.decisions: list[Decision] = []

    def choose_action(self, decision: Decision, /) -> SemanticAction:
        self.decisions.append(decision)
        return decision.legal_actions[-1 if self.last else 0]


class _IllegalPolicy:
    def choose_action(self, decision: Decision, /) -> SemanticAction:
        return DrawAction(decision.decision_id + 10_000)


def _starting_fixture(seed: int = 2201) -> tuple[InformationSetSpec, GameState]:
    live = build_setup_state(seed)
    decision = current_decisions(live)[0]
    spec = InformationSetSpecBuilder().build(live, decision)
    sampled = InformationSetSampler(seed=seed + 1).sample(spec)
    assert sampled is not None
    return spec, sampled


def _play_fixture(seed: int = 2301) -> tuple[InformationSetSpec, GameState, GameState]:
    live = build_setup_state(seed)
    live = apply_action(live, current_decisions(live)[0].legal_actions[0]).state
    live = apply_action(live, current_decisions(live)[0].legal_actions[0]).state
    decision = current_decisions(live)[0]
    spec = InformationSetSpecBuilder().build(live, decision)
    sampled = InformationSetSampler(seed=seed + 1).sample(spec)
    assert sampled is not None
    return spec, sampled, live


def _decision(
    state: GameState,
    decision_id: int,
    chooser: PlayerId,
    actions: tuple[SemanticAction, ...],
    *,
    kind: DecisionKind = DecisionKind.EFFECT_CHOICE,
) -> Decision:
    return Decision(
        decision_id,
        kind,
        chooser,
        chooser,
        observe(state, chooser),
        actions,
    )


def _search(
    graph: _Graph,
    spec: InformationSetSpec,
    samples: tuple[GameState, ...],
    *,
    ceiling: int = 1024,
    policy: _RecordingPolicy | _IllegalPolicy | None = None,
) -> RolloutSelection:
    descriptor = RolloutSearchDescriptor(
        determinization_count=len(samples),
        route_transition_ceiling=ceiling,
        implementation_version=_TEST_IMPLEMENTATION_VERSION,
        evaluator_version=_TEST_EVALUATOR_VERSION,
        continuation_agent_descriptor_id=_TEST_CONTINUATION_DESCRIPTOR.descriptor_id,
        continuation_agent_descriptor_json=_TEST_CONTINUATION_DESCRIPTOR.canonical_json(),
    )
    return DeterministicSampledRollout(
        descriptor,
        evaluator=graph,
        hooks=graph.hooks(policy),
    ).select(spec, samples, sampler_seed_digest="sha256:" + "a" * 64)


def test_noncanonical_hooks_and_evaluator_cannot_publish_canonical_descriptor_identity() -> None:
    graph = _Graph({}, {}, {})
    canonical = RolloutSearchDescriptor(determinization_count=1)

    with pytest.raises(SearchInputError, match="runtime implementation version"):
        DeterministicSampledRollout(
            canonical,
            evaluator=graph,
            hooks=graph.hooks(),
        )
    with pytest.raises(SearchInputError, match=r"must declare.*version identity"):
        DeterministicSampledRollout(canonical, evaluator=graph.evaluate)


def test_every_root_uses_every_common_sample_and_math_fsum_in_stable_order() -> None:
    spec, sampled, _ = _play_fixture()
    samples = tuple(replace(sampled, next_event_id=node) for node in (1000, 2000, 3000))
    root_decision = replace(current_decisions(sampled)[0], legal_actions=spec.legal_actions)
    graph = _Graph(
        decisions={sample.next_event_id: (root_decision,) for sample in samples},
        edges={},
        values={},
    )
    values = (1e16, 1.0, -1e16)
    for sample_index, sample in enumerate(samples):
        for action_index, root_action in enumerate(spec.legal_actions):
            leaf = sample.next_event_id + 100 + action_index
            graph.edges[(sample.next_event_id, action_key(root_action))] = _Edge(leaf, turn_delta=1)
            graph.values[leaf] = (
                values[sample_index] if action_index == 0 else 0.2 if action_index == 1 else -1.0
            )

    result = _search(graph, spec, samples)

    assert len(result.telemetry.routes) == len(spec.legal_actions) * len(samples)
    assert tuple(
        (route.root_action_index, route.determinization_index) for route in result.telemetry.routes
    ) == tuple(
        (action_index, sample_index)
        for action_index in range(len(spec.legal_actions))
        for sample_index in range(len(samples))
    )
    assert result.telemetry.action_sample_values[0] == values
    assert result.telemetry.action_mean_values[0] == pytest.approx(1.0 / 3.0)
    assert result.action == spec.legal_actions[0]
    assert result.statistics.root_engine_transitions == len(result.telemetry.routes)


def test_exact_root_value_ties_choose_first_stable_legal_action() -> None:
    spec, sampled, _ = _play_fixture(2306)
    root = replace(sampled, next_event_id=3500)
    root_decision = replace(current_decisions(sampled)[0], legal_actions=spec.legal_actions)
    graph = _Graph({3500: (root_decision,)}, {}, {})
    for action_index, root_action in enumerate(spec.legal_actions):
        leaf = 3600 + action_index
        graph.edges[(3500, action_key(root_action))] = _Edge(leaf, turn_delta=1)
        graph.values[leaf] = 0.125

    result = _search(graph, spec, (root,))

    assert result.telemetry.tied_action_indices == tuple(range(len(spec.legal_actions)))
    assert result.telemetry.selected_action_index == 0
    assert result.action == spec.legal_actions[0]


def test_continuation_selects_one_action_per_decision_and_never_enters_next_turn() -> None:
    spec, sampled, _ = _play_fixture(2302)
    root = replace(sampled, next_event_id=4000)
    root_decision = replace(current_decisions(sampled)[0], legal_actions=spec.legal_actions)
    graph = _Graph({4000: (root_decision,)}, {}, {})
    policy = _RecordingPolicy(last=True)

    for action_index, root_action in enumerate(spec.legal_actions):
        if action_index != 0:
            leaf = 4900 + action_index
            graph.edges[(4000, action_key(root_action))] = _Edge(leaf, turn_delta=1)
            graph.values[leaf] = -float(action_index)
            continue
        graph.edges[(4000, action_key(root_action))] = _Edge(4100)
        choosers = (
            spec.chooser,
            spec.chooser,
            PlayerId.PLAYER_2 if spec.chooser is PlayerId.PLAYER_1 else PlayerId.PLAYER_1,
        )
        for offset, chooser in enumerate(choosers):
            node = 4100 + offset
            actions = (DrawAction(5000 + offset), DeclineAction(5000 + offset))
            graph.decisions[node] = (_decision(sampled, actions[0].decision_id, chooser, actions),)
            graph.edges[(node, action_key(actions[-1]))] = _Edge(
                node + 1,
                turn_delta=1 if offset == len(choosers) - 1 else 0,
            )
        graph.values[4103] = 0.75
        following = DrawAction(6000)
        graph.decisions[4103] = (
            _decision(
                sampled,
                following.decision_id,
                spec.chooser,
                (following,),
                kind=DecisionKind.TURN_ACTION,
            ),
        )

    result = _search(graph, spec, (root,), policy=policy)
    route = result.telemetry.routes[0]

    assert route.continuation_engine_transitions == 3
    assert tuple(step.chooser for step in route.rollout_path) == tuple(
        chooser.value
        for chooser in (
            spec.chooser,
            spec.chooser,
            PlayerId.PLAYER_2 if spec.chooser is PlayerId.PLAYER_1 else PlayerId.PLAYER_1,
        )
    )
    assert route.completed_turn_count == 1
    assert 4103 not in graph.decision_calls
    assert len(policy.decisions) == 3
    assert all(item.observation.viewer is item.chooser for item in policy.decisions)


def test_terminal_root_and_continuation_stop_immediately() -> None:
    spec, sampled, _ = _play_fixture(2303)
    root = replace(sampled, next_event_id=7000)
    root_decision = replace(current_decisions(sampled)[0], legal_actions=spec.legal_actions)
    graph = _Graph({7000: (root_decision,)}, {}, {})
    policy = _RecordingPolicy()
    for action_index, root_action in enumerate(spec.legal_actions):
        if action_index == 0:
            graph.edges[(7000, action_key(root_action))] = _Edge(
                7100, terminal=True, winners=(spec.chooser,)
            )
            graph.values[7100] = 2.0
        elif action_index == 1:
            graph.edges[(7000, action_key(root_action))] = _Edge(7200)
            action = DrawAction(7201)
            graph.decisions[7200] = (
                _decision(sampled, action.decision_id, spec.chooser, (action,)),
            )
            graph.edges[(7200, action_key(action))] = _Edge(7202, terminal=True, winners=())
            graph.values[7202] = 0.0
        else:
            leaf = 7300 + action_index
            graph.edges[(7000, action_key(root_action))] = _Edge(leaf, turn_delta=1)
            graph.values[leaf] = -1.0

    result = _search(graph, spec, (root,), policy=policy)

    root_terminal = result.telemetry.routes[0]
    continuation_terminal = result.telemetry.routes[1]
    assert root_terminal.continuation_engine_transitions == 0
    assert root_terminal.completed_turn_count == 0
    assert root_terminal.leaf is not None
    assert root_terminal.leaf.terminal_reason == TerminalReason.CARD_EFFECT.value
    assert continuation_terminal.continuation_engine_transitions == 1
    assert continuation_terminal.completed_turn_count == 0
    assert result.action == spec.legal_actions[0]


def test_starting_response_is_preselected_once_and_rebound_across_root_candidates() -> None:
    spec, sampled = _starting_fixture()
    root = replace(sampled, next_event_id=8000)
    initial = current_decisions(sampled)
    root_decision = next(item for item in initial if item.decision_id == spec.target_decision_id)
    opponent_decision = next(item for item in initial if item.chooser is not spec.chooser)
    graph = _Graph({8000: (root_decision, opponent_decision)}, {}, {})
    policy = _RecordingPolicy()
    preselected_key = action_key(opponent_decision.legal_actions[0])

    for root_index, root_action in enumerate(spec.legal_actions):
        pending = 8100 + root_index
        graph.edges[(8000, action_key(root_action))] = _Edge(pending)
        response_id = 8200 + root_index
        responses = tuple(
            ChooseStartingMeldAction(response_id, action.card_id)
            for action in opponent_decision.legal_actions
            if isinstance(action, ChooseStartingMeldAction)
        )
        rebound = _decision(
            sampled,
            response_id,
            opponent_decision.chooser,
            responses,
            kind=DecisionKind.STARTING_MELD,
        )
        graph.decisions[pending] = (rebound,)
        selected = next(action for action in responses if action_key(action) == preselected_key)
        play_node = 8300 + root_index
        graph.edges[(pending, action_key(selected))] = _Edge(
            play_node, turn_delta=1, phase=GamePhase.PLAY
        )
        continuation = DrawAction(8400 + root_index)
        graph.decisions[play_node] = (
            _decision(
                sampled,
                continuation.decision_id,
                spec.chooser,
                (continuation,),
                kind=DecisionKind.TURN_ACTION,
            ),
        )
        leaf = 8500 + root_index
        graph.edges[(play_node, action_key(continuation))] = _Edge(leaf, turn_delta=1)
        graph.values[leaf] = float(root_index)

    result = _search(graph, spec, (root,), policy=policy)

    assert len(policy.decisions) == 1 + len(spec.legal_actions)
    assert policy.decisions[0].chooser is opponent_decision.chooser
    assert {route.setup_response_action_key for route in result.telemetry.routes} == {
        preselected_key
    }
    assert all(
        route.setup_response_engine_transitions == 1 and route.completed_turn_count == 1
        for route in result.telemetry.routes
    )


def test_one_latent_starting_choice_is_not_reopened_or_counted_as_a_response() -> None:
    live = build_setup_state(2401)
    first = current_decisions(live)[0]
    after_first = apply_action(live, first.legal_actions[0])
    assert after_first.decision is not None
    spec = InformationSetSpecBuilder().build(after_first.state, after_first.decision)
    sample = InformationSetSampler(seed=2402).sample(spec)
    assert sample is not None
    assert sum(choice is not None for choice in sample.starting_meld_choices) == 1

    result = DeterministicSampledRollout(RolloutSearchDescriptor(determinization_count=1)).select(
        spec, (sample,)
    )

    assert all(route.setup_response_engine_transitions == 0 for route in result.telemetry.routes)
    assert all(route.setup_response_action_key is None for route in result.telemetry.routes)
    assert all(route.completed_turn_count == 1 for route in result.telemetry.routes)


def test_first_and_second_paid_action_roots_stop_after_the_same_anchored_turn() -> None:
    live = build_setup_state(2451)
    live = apply_action(live, current_decisions(live)[0].legal_actions[0]).state
    live = apply_action(live, current_decisions(live)[0].legal_actions[0]).state
    # The first game turn has one paid action. Complete it to reach a normal two-action turn.
    first_turn = current_decision(live)
    assert first_turn is not None
    live = apply_action(
        live,
        next(action for action in first_turn.legal_actions if isinstance(action, DrawAction)),
    ).state
    assert live.paid_actions_remaining == 2

    first_paid = current_decision(live)
    assert first_paid is not None
    first_spec = InformationSetSpecBuilder().build(live, first_paid)
    first_sample = InformationSetSampler(seed=2452).sample(first_spec)
    assert first_sample is not None
    first_result = DeterministicSampledRollout(
        RolloutSearchDescriptor(determinization_count=1)
    ).select(first_spec, (first_sample,))
    first_draw_index = next(
        index
        for index, action in enumerate(first_spec.legal_actions)
        if isinstance(action, DrawAction)
    )
    first_draw_route = first_result.telemetry.routes[first_draw_index]
    assert first_draw_route.completed_turn_count == 1
    assert first_draw_route.rollout_path[0].decision_kind == DecisionKind.TURN_ACTION.value

    after_first_paid = apply_action(
        live,
        next(action for action in first_paid.legal_actions if isinstance(action, DrawAction)),
    ).state
    assert after_first_paid.turn_number == live.turn_number
    assert after_first_paid.paid_actions_remaining == 1
    second_paid = current_decision(after_first_paid)
    assert second_paid is not None
    second_spec = InformationSetSpecBuilder().build(after_first_paid, second_paid)
    second_sample = InformationSetSampler(seed=2453).sample(second_spec)
    assert second_sample is not None
    second_result = DeterministicSampledRollout(
        RolloutSearchDescriptor(determinization_count=1)
    ).select(second_spec, (second_sample,))
    second_draw_index = next(
        index
        for index, action in enumerate(second_spec.legal_actions)
        if isinstance(action, DrawAction)
    )
    second_draw_route = second_result.telemetry.routes[second_draw_index]
    assert second_draw_route.completed_turn_count == 1
    assert second_draw_route.rollout_path == ()


@pytest.mark.parametrize("pending_owner", ["active", "opponent"])
def test_own_and_opponent_pending_effect_roots_finish_the_active_turn(
    pending_owner: str,
) -> None:
    if pending_owner == "active":
        live = build_explicit_state(
            positions=(
                (
                    PlayerId.PLAYER_1,
                    ExplicitPlayerPosition(
                        hand=(CardId("city-states"), CardId("sailing")),
                        board=((Color.YELLOW, (CardId("agriculture"),)),),
                    ),
                ),
                (
                    PlayerId.PLAYER_2,
                    ExplicitPlayerPosition(hand=(CardId("archery"),)),
                ),
            ),
            active_player=PlayerId.PLAYER_1,
        )
    else:
        live = build_explicit_state(
            positions=(
                (
                    PlayerId.PLAYER_1,
                    ExplicitPlayerPosition(
                        hand=(CardId("tools"),),
                        board=((Color.RED, (CardId("archery"),)),),
                    ),
                ),
                (
                    PlayerId.PLAYER_2,
                    ExplicitPlayerPosition(hand=(CardId("pottery"), CardId("clothing"))),
                ),
            ),
            active_player=PlayerId.PLAYER_1,
        )
    turn = current_decision(live)
    assert turn is not None
    dogma = next(action for action in turn.legal_actions if isinstance(action, DogmaAction))
    started = apply_action(live, dogma)
    assert started.decision is not None
    root = started.decision
    assert root.kind is DecisionKind.EFFECT_CHOICE
    if pending_owner == "active":
        assert root.chooser is started.state.active_player
    else:
        assert root.chooser is not started.state.active_player

    spec = InformationSetSpecBuilder().build(started.state, root)
    sample = InformationSetSampler(seed=f"pending-{pending_owner}").sample(spec)
    assert sample is not None
    result = DeterministicSampledRollout(RolloutSearchDescriptor(determinization_count=1)).select(
        spec, (sample,)
    )

    assert all(route.completed_turn_count == 1 for route in result.telemetry.routes)
    if pending_owner == "opponent":
        assert any(
            step.chooser == PlayerId.PLAYER_1.value
            for route in result.telemetry.routes
            for step in route.rollout_path
        )


def test_strict_all_in_ceiling_fails_before_next_transition_with_private_trace() -> None:
    spec, sampled, _ = _play_fixture(2304)
    root = replace(sampled, next_event_id=9000)
    root_decision = replace(current_decisions(sampled)[0], legal_actions=spec.legal_actions)
    first = DrawAction(9101)
    second = DrawAction(9102)
    graph = _Graph(
        {
            9000: (root_decision,),
            9100: (_decision(sampled, first.decision_id, spec.chooser, (first,)),),
            9101: (_decision(sampled, second.decision_id, spec.chooser, (second,)),),
        },
        {
            (9000, action_key(spec.legal_actions[0])): _Edge(9100),
            (9100, action_key(first)): _Edge(9101),
            (9101, action_key(second)): _Edge(9102, turn_delta=1),
        },
        {9102: 1.0},
    )

    with pytest.raises(RolloutSafetyCeilingError) as raised:
        _search(graph, spec, (root,), ceiling=2, policy=_RecordingPolicy())

    trace = raised.value.trace
    assert trace.route.safety_ceiling_hit
    assert trace.route.total_engine_transitions == 2
    assert trace.route.root_engine_transitions == 1
    assert trace.route.continuation_engine_transitions == 1
    assert len(trace.route.rollout_path) == 1
    assert graph.evaluation_calls == []
    assert (9101, action_key(second)) not in graph.transition_calls
    assert trace.dumps() == trace.dumps()


def test_rollout_rejects_live_state_illegal_policy_missing_decision_and_turn_skip() -> None:
    spec, sampled, live = _play_fixture(2305)
    descriptor = RolloutSearchDescriptor(determinization_count=1)
    with pytest.raises(SearchInputError, match="only synthetic"):
        DeterministicSampledRollout(descriptor).select(spec, (live,))

    root = replace(sampled, next_event_id=10_000)
    root_decision = replace(current_decisions(sampled)[0], legal_actions=spec.legal_actions)
    continuation = DrawAction(10_101)
    illegal_graph = _Graph(
        {
            10_000: (root_decision,),
            10_100: (_decision(sampled, continuation.decision_id, spec.chooser, (continuation,)),),
        },
        {(10_000, action_key(action)): _Edge(10_100) for action in spec.legal_actions},
        {},
    )
    with pytest.raises(SearchInvariantError, match="illegal action"):
        _search(illegal_graph, spec, (root,), policy=_IllegalPolicy())

    missing_graph = _Graph(
        {10_000: (root_decision,)},
        {(10_000, action_key(action)): _Edge(10_200) for action in spec.legal_actions},
        {},
    )
    with pytest.raises(SearchInvariantError, match="exactly one decision"):
        _search(missing_graph, spec, (root,), policy=_RecordingPolicy())

    skipping_graph = _Graph(
        {10_000: (root_decision,)},
        {
            (10_000, action_key(action)): _Edge(10_300, turn_delta=2)
            for action in spec.legal_actions
        },
        {10_300: 0.0},
    )
    with pytest.raises(SearchInvariantError, match="skipped or reversed"):
        _search(skipping_graph, spec, (root,), policy=_RecordingPolicy())


def test_hidden_equivalent_roots_produce_identical_samples_and_rollout_selection() -> None:
    first = build_explicit_state(
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(hand=(CardId("city-states"),)),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(
                    hand=(CardId("archery"),),
                    score_pile=(CardId("metalworking"),),
                    board=((Color.BLUE, (CardId("pottery"), CardId("tools"))),),
                ),
            ),
        ),
        active_player=PlayerId.PLAYER_1,
        next_decision_id=41,
        next_event_id=19,
        next_dogma_action_id=13,
    )
    hidden = first.player(PlayerId.PLAYER_2).hand[0]
    replacement = next(card for card in first.supply.pile(1) if card != hidden)
    opponent = replace(first.player(PlayerId.PLAYER_2), hand=(replacement,))
    pile = tuple(hidden if card == replacement else card for card in first.supply.pile(1))
    second = replace(
        first.replace_player(opponent),
        supply=SupplyState((pile, *first.supply.piles[1:])),
    )
    first_decision = current_decision(first)
    second_decision = current_decision(second)
    assert first_decision is not None and second_decision is not None
    builder = InformationSetSpecBuilder()
    first_spec = builder.build(first, first_decision)
    second_spec = builder.build(second, second_decision)
    assert first_spec == second_spec
    first_sample = InformationSetSampler().sample(first_spec, "rollout-hidden-equivalent")
    second_sample = InformationSetSampler().sample(second_spec, "rollout-hidden-equivalent")
    assert first_sample is not None and first_sample == second_sample

    searcher = DeterministicSampledRollout(RolloutSearchDescriptor(determinization_count=1))
    first_result = searcher.select(first_spec, (first_sample,))
    second_result = searcher.select(second_spec, (second_sample,))

    assert first_result.action == second_result.action
    assert first_result.telemetry.dumps() == second_result.telemetry.dumps()
