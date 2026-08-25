"""Explicit mid-game position builder and scripted dogma driver."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from innovation_ai.innovation.actions import (
    ChooseBranchAction,
    ChooseCardAction,
    ChooseColorAction,
    ChoosePlayerAction,
    ChooseSplayAction,
    ChooseValueAction,
    Decision,
    DeclineAction,
    FinishSelectionAction,
    SemanticAction,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects.dogma import start_dogma
from innovation_ai.innovation.effects.engine import (
    resume_effect,
    submit_effect_action,
)
from innovation_ai.innovation.effects.model import (
    EffectEvent,
    EffectResolution,
    EffectStatus,
)
from innovation_ai.innovation.effects.program import EffectProgramRegistry
from innovation_ai.innovation.effects.registry import load_effect_programs
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GamePhase,
    GameState,
    PlayerTurnCounters,
    TerminalResult,
    TurnCounters,
    build_explicit_state,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)

ChoicePicker = Callable[[Decision], SemanticAction]


def _card_ids(names: Sequence[str | CardId]) -> tuple[CardId, ...]:
    return tuple(name if isinstance(name, CardId) else CardId(name) for name in names)


@dataclass(frozen=True, slots=True)
class _Position:
    hand: tuple[CardId, ...] = ()
    score_pile: tuple[CardId, ...] = ()
    board: tuple[tuple[Color, tuple[CardId, ...]], ...] = ()
    splays: tuple[tuple[Color, SplayDirection], ...] = ()
    normal_achievements: tuple[NormalAchievementId, ...] = ()
    special_achievements: tuple[SpecialAchievementId, ...] = ()


class ScenarioBuilder:
    """Fluent builder for an arbitrary validated mid-game position.

    Every board colour is given bottom-to-top, matching WP2's stack ordering. The result is
    validated by ``assert_state_invariants``, so an impossible fixture fails in the builder rather
    than deep inside a card program.
    """

    def __init__(self, registry: CardRegistry | None = None) -> None:
        self._registry = registry or load_card_registry()
        self._positions: dict[PlayerId, _Position] = {}
        self._supply_tops: dict[int, tuple[CardId, ...]] = {}
        self._removed: tuple[CardId, ...] = ()
        self._phase = GamePhase.PLAY
        self._active: PlayerId | None = PlayerId.PLAYER_1
        self._turn = 3
        self._paid_actions = 2
        self._counters: TurnCounters | None = None

    def _position(self, player_id: PlayerId) -> _Position:
        return self._positions.get(player_id, _Position())

    def hand(self, player_id: PlayerId, cards: Sequence[str | CardId]) -> ScenarioBuilder:
        """Set one player's complete hand."""

        self._positions[player_id] = replace(self._position(player_id), hand=_card_ids(cards))
        return self

    def score(self, player_id: PlayerId, cards: Sequence[str | CardId]) -> ScenarioBuilder:
        """Set one player's complete score pile."""

        self._positions[player_id] = replace(self._position(player_id), score_pile=_card_ids(cards))
        return self

    def board(
        self,
        player_id: PlayerId,
        color: Color,
        cards: Sequence[str | CardId],
        *,
        splay: SplayDirection | None = None,
    ) -> ScenarioBuilder:
        """Set one colour stack bottom-to-top, optionally splayed."""

        position = self._position(player_id)
        stacks = tuple(
            (existing_color, existing_cards)
            for existing_color, existing_cards in position.board
            if existing_color is not color
        )
        splays = tuple(
            (existing_color, direction)
            for existing_color, direction in position.splays
            if existing_color is not color
        )
        if splay is not None:
            splays = (*splays, (color, splay))
        self._positions[player_id] = replace(
            position,
            board=(*stacks, (color, _card_ids(cards))),
            splays=splays,
        )
        return self

    def achievements(
        self,
        player_id: PlayerId,
        *,
        normal: Sequence[NormalAchievementId] = (),
        special: Sequence[SpecialAchievementId] = (),
    ) -> ScenarioBuilder:
        """Grant already-claimed achievements."""

        self._positions[player_id] = replace(
            self._position(player_id),
            normal_achievements=tuple(normal),
            special_achievements=tuple(special),
        )
        return self

    def supply(self, age: int, top: Sequence[str | CardId]) -> ScenarioBuilder:
        """Pin named cards to the top of one age pile so a draw is predictable."""

        self._supply_tops[age] = _card_ids(top)
        return self

    def removed(self, cards: Sequence[str | CardId]) -> ScenarioBuilder:
        """Set aside cards outside the game."""

        self._removed = _card_ids(cards)
        return self

    def exhaust_supply(self, *, into: PlayerId = PlayerId.PLAYER_1) -> ScenarioBuilder:
        """Move every card not already placed into one score pile, emptying all ten supplies.

        This is how a draw-above-age-ten test is written: no supply is left, so the next draw
        cannot be satisfied and the game must end.
        """

        used = set(self._removed)
        for position in self._positions.values():
            used.update(position.hand)
            used.update(position.score_pile)
            for _, cards in position.board:
                used.update(cards)
        for cards in self._supply_tops.values():
            used.update(cards)
        # Ages 1-9 each keep one hidden normal achievement, chosen from what is left.
        for age in range(1, 10):
            candidates = sorted(
                (
                    card.id
                    for card in self._registry.cards
                    if card.age == age and card.id not in used
                ),
                key=str,
            )
            if candidates:
                used.add(candidates[0])
        target = self._position(into)
        remaining = tuple(
            sorted((card.id for card in self._registry.cards if card.id not in used), key=str)
        )
        self._positions[into] = replace(target, score_pile=(*target.score_pile, *remaining))
        return self

    def active(self, player_id: PlayerId, *, paid_actions: int = 2) -> ScenarioBuilder:
        """Set the active player and their remaining paid actions."""

        self._active = player_id
        self._paid_actions = paid_actions
        return self

    def turn(self, number: int) -> ScenarioBuilder:
        """Set the turn number."""

        self._turn = number
        return self

    def counters(self, player_id: PlayerId, *, tucked: int = 0, scored: int = 0) -> ScenarioBuilder:
        """Set one player's per-turn Monument counters."""

        base = self._counters or TurnCounters.empty()
        players = tuple(
            PlayerTurnCounters(counter.player_id, tucked, scored)
            if counter.player_id is player_id
            else counter
            for counter in base.players
        )
        self._counters = TurnCounters((players[0], players[1]))
        return self

    def terminal(self, result: TerminalResult) -> ScenarioBuilder:
        """Build an already-terminal position."""

        self._phase = GamePhase.TERMINAL
        self._terminal = result
        return self

    def build(self) -> GameState:
        """Return the validated authoritative state."""

        positions = tuple(
            (
                player_id,
                ExplicitPlayerPosition(
                    hand=position.hand,
                    score_pile=position.score_pile,
                    board=position.board,
                    splays=position.splays,
                    normal_achievements=position.normal_achievements,
                    special_achievements=position.special_achievements,
                ),
            )
            for player_id, position in sorted(
                self._positions.items(), key=lambda item: item[0].value
            )
        )
        return build_explicit_state(
            self._registry,
            positions=positions,
            supply_tops=tuple(sorted(self._supply_tops.items())),
            removed_cards=self._removed,
            phase=self._phase,
            active_player=self._active,
            turn_number=self._turn,
            paid_actions_remaining=self._paid_actions,
            turn_counters=self._counters,
        )


def scenario(registry: CardRegistry | None = None) -> ScenarioBuilder:
    """Start building an explicit mid-game position."""

    return ScenarioBuilder(registry)


def choose_card(card: str | CardId) -> ChoicePicker:
    """Pick the ``ChooseCardAction`` naming ``card``."""

    target = card if isinstance(card, CardId) else CardId(card)

    def picker(decision: Decision) -> SemanticAction:
        for action in decision.legal_actions:
            if isinstance(action, ChooseCardAction) and action.card_id == target:
                return action
        raise AssertionError(
            f"decision {decision.decision_id} cannot choose {target}; "
            f"legal: {[getattr(a, 'card_id', a.kind) for a in decision.legal_actions]}"
        )

    return picker


def choose_color(color: Color) -> ChoicePicker:
    """Pick the ``ChooseColorAction`` naming ``color``."""

    def picker(decision: Decision) -> SemanticAction:
        for action in decision.legal_actions:
            if isinstance(action, ChooseColorAction) and action.color is color:
                return action
        raise AssertionError(f"decision {decision.decision_id} cannot choose {color}")

    return picker


def choose_value(value: int) -> ChoicePicker:
    """Pick the ``ChooseValueAction`` naming ``value``."""

    def picker(decision: Decision) -> SemanticAction:
        for action in decision.legal_actions:
            if isinstance(action, ChooseValueAction) and action.value == value:
                return action
        raise AssertionError(f"decision {decision.decision_id} cannot choose value {value}")

    return picker


def choose_player(player_id: PlayerId) -> ChoicePicker:
    """Pick the ``ChoosePlayerAction`` naming ``player_id``."""

    def picker(decision: Decision) -> SemanticAction:
        for action in decision.legal_actions:
            if isinstance(action, ChoosePlayerAction) and action.player_id is player_id:
                return action
        raise AssertionError(f"decision {decision.decision_id} cannot choose {player_id}")

    return picker


def choose_splay(direction: SplayDirection) -> ChoicePicker:
    """Pick the ``ChooseSplayAction`` naming ``direction``."""

    def picker(decision: Decision) -> SemanticAction:
        for action in decision.legal_actions:
            if isinstance(action, ChooseSplayAction) and action.direction is direction:
                return action
        raise AssertionError(f"decision {decision.decision_id} cannot splay {direction}")

    return picker


def choose_branch(branch_id: str) -> ChoicePicker:
    """Pick the ``ChooseBranchAction`` naming ``branch_id``."""

    def picker(decision: Decision) -> SemanticAction:
        for action in decision.legal_actions:
            if isinstance(action, ChooseBranchAction) and action.branch_id == branch_id:
                return action
        raise AssertionError(f"decision {decision.decision_id} cannot branch to {branch_id!r}")

    return picker


def decline() -> ChoicePicker:
    """Pick the ``DeclineAction``."""

    def picker(decision: Decision) -> SemanticAction:
        for action in decision.legal_actions:
            if isinstance(action, DeclineAction):
                return action
        raise AssertionError(f"decision {decision.decision_id} is not declinable")

    return picker


def finish() -> ChoicePicker:
    """Pick the ``FinishSelectionAction``."""

    def picker(decision: Decision) -> SemanticAction:
        for action in decision.legal_actions:
            if isinstance(action, FinishSelectionAction):
                return action
        raise AssertionError(f"decision {decision.decision_id} cannot finish selection")

    return picker


@dataclass(frozen=True, slots=True)
class DogmaResult:
    """Everything a card test needs about one completed dogma action."""

    state: GameState
    status: EffectStatus
    events: tuple[EffectEvent, ...]
    qualifying_changes: int
    decisions: tuple[Decision, ...]

    @property
    def terminal(self) -> TerminalResult | None:
        """Return the terminal result, if the dogma action ended the game."""

        return self.state.terminal_result

    def changed_cards(self) -> tuple[CardId, ...]:
        """Return every card moved by a qualifying change, in event order."""

        return tuple(
            card_id for event in self.events if event.changed for card_id in event.card_ids
        )


def resolve_dogma(
    state: GameState,
    card_id: str | CardId,
    *choices: ChoicePicker,
    activator: PlayerId | None = None,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
    verify_resume: bool = True,
) -> DogmaResult:
    """Run one whole dogma action, answering each decision from ``choices`` in order.

    With ``verify_resume`` the state is serialized and restored at every decision boundary and the
    restored state must produce an identical hash, which is the acceptance criterion every card
    wave has to meet.
    """

    from .assertions import round_trip_state

    registry = registry or load_card_registry()
    resolved_programs = programs or load_effect_programs()
    target = card_id if isinstance(card_id, CardId) else CardId(card_id)
    chosen_activator = activator or state.active_player
    if chosen_activator is None:
        raise AssertionError("resolve_dogma needs an activator or an active player")

    resolution: EffectResolution = start_dogma(
        state, target, chosen_activator, resolved_programs, registry
    )
    events = list(resolution.events)
    decisions: list[Decision] = []
    pending = list(choices)
    while resolution.status is EffectStatus.AWAIT_DECISION:
        decision = resolution.decision
        assert decision is not None
        decisions.append(decision)
        working = resolution.state
        if verify_resume:
            working = round_trip_state(working, registry)
        if not pending:
            raise AssertionError(
                f"dogma on {target} needs another choice at decision {decision.decision_id}: "
                f"{[type(a).__name__ for a in decision.legal_actions]}"
            )
        action = pending.pop(0)(decision)
        resolution = submit_effect_action(working, action, resolved_programs, registry)
        events.extend(resolution.events)
    if pending:
        raise AssertionError(f"dogma on {target} left {len(pending)} unused scripted choices")
    return DogmaResult(
        resolution.state,
        resolution.status,
        tuple(events),
        resolution.qualifying_changes,
        tuple(decisions),
    )


def step_to_boundary(
    state: GameState,
    registry: CardRegistry | None = None,
    programs: EffectProgramRegistry | None = None,
) -> EffectResolution:
    """Resume a paused effect stack to its next boundary."""

    registry = registry or load_card_registry()
    return resume_effect(state, programs or load_effect_programs(), registry)
