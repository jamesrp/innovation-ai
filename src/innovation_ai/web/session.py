"""Thread-safe in-memory hot-seat game session for the QA web UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import cast
from uuid import uuid4

from innovation_ai.innovation.actions import (
    AchieveAction,
    ChooseBranchAction,
    ChooseCardAction,
    ChooseCardsAction,
    ChooseColorAction,
    ChoosePlayerAction,
    ChooseSplayAction,
    ChooseStartingMeldAction,
    ChooseValueAction,
    Decision,
    DeclineAction,
    DogmaAction,
    DrawAction,
    FinishSelectionAction,
    MeldAction,
    OrderCardsAction,
    SemanticAction,
    action_payload,
    decision_payload,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.replay import GameLogRecorder
from innovation_ai.innovation.serialization import (
    JsonObject,
    JsonValue,
    action_from_payload,
    observation_payload,
    terminal_payload,
)
from innovation_ai.innovation.state import build_setup_state
from innovation_ai.innovation.types import CardId, PlayerId


def _title(value: str) -> str:
    return value.replace("-", " ").title()


class SessionConflict(RuntimeError):
    """A browser request targets a stale game generation or revision."""


@dataclass(slots=True)
class WebGameSession:
    """One process-local game with replay-backed undo and downloadable logs."""

    seed: int = 0
    registry: CardRegistry = field(default_factory=load_card_registry)
    _recorder: GameLogRecorder = field(init=False, repr=False)
    _actions: list[SemanticAction] = field(default_factory=list, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _game_id: str = field(default_factory=lambda: uuid4().hex, init=False, repr=False)
    _revision: int = field(default=0, init=False, repr=False)
    _last_viewer: PlayerId = field(default=PlayerId.PLAYER_1, init=False, repr=False)

    def __post_init__(self) -> None:
        self._recorder = GameLogRecorder(build_setup_state(self.seed, self.registry), self.registry)

    def new_game(self, seed: int) -> None:
        """Replace the current game with a fresh deterministic setup."""

        with self._lock:
            self.seed = seed
            self._actions.clear()
            self._game_id = uuid4().hex
            self._revision = 0
            self._last_viewer = PlayerId.PLAYER_1
            self._recorder = GameLogRecorder(build_setup_state(seed, self.registry), self.registry)

    def submit_payload(self, payload: object) -> None:
        """Decode and submit one exact semantic action from the current decision."""

        action = action_from_payload(payload)
        with self._lock:
            decision = next(
                (
                    item
                    for item in self._recorder.decisions()
                    if item.decision_id == action.decision_id
                ),
                None,
            )
            if decision is None:
                raise SessionConflict("the submitted decision is no longer pending")
            self._last_viewer = decision.chooser
            self._recorder.submit(action)
            self._actions.append(action)
            self._revision += 1

    def undo(self) -> bool:
        """Replay all but the most recent action, returning whether anything changed."""

        with self._lock:
            if not self._actions:
                return False
            retained = self._actions[:-1]
            recorder = GameLogRecorder(build_setup_state(self.seed, self.registry), self.registry)
            for action in retained:
                recorder.submit(action)
            self._actions = retained
            self._recorder = recorder
            self._revision += 1
            decisions = recorder.decisions()
            self._last_viewer = decisions[0].chooser if decisions else PlayerId.PLAYER_1
            return True

    def new_game_and_snapshot(self, seed: int) -> JsonObject:
        """Reset and return its first boundary under one lock acquisition."""

        with self._lock:
            self.new_game(seed)
            return self.snapshot()

    def submit_and_snapshot(self, payload: object, *, game_id: str, revision: int) -> JsonObject:
        """Apply an action only to the exact browser revision and return the next boundary."""

        with self._lock:
            self._require_revision(game_id, revision)
            self.submit_payload(payload)
            return self.snapshot()

    def undo_and_snapshot(self, *, game_id: str, revision: int) -> JsonObject:
        """Undo only the exact browser revision and return the restored boundary."""

        with self._lock:
            self._require_revision(game_id, revision)
            self.undo()
            return self.snapshot()

    def _require_revision(self, game_id: str, revision: int) -> None:
        if game_id != self._game_id or revision != self._revision:
            raise SessionConflict("the game changed in another request; refresh and try again")

    def game_log_json(self) -> str:
        """Return the current complete-or-in-progress game log."""

        from innovation_ai.innovation.logs import dumps_game_log

        with self._lock:
            return dumps_game_log(self._recorder.game_log())

    def snapshot(self) -> JsonObject:
        """Build the browser-facing QA view model from player-safe observations."""

        with self._lock:
            state = self._recorder.state
            decisions = self._recorder.decisions()
            selected = decisions[0] if decisions else None
            if selected is not None:
                observation = cast(JsonValue, observation_payload(selected.observation))
            else:
                observation = cast(
                    JsonValue,
                    observation_payload(observe(state, self._last_viewer, self.registry)),
                )
            log = self._recorder.game_log()
            return {
                "seed": self.seed,
                "game_id": self._game_id,
                "revision": self._revision,
                "transition_count": len(self._actions),
                "can_undo": bool(self._actions),
                "phase": state.phase.value,
                "state_hash": log.final_state_hash,
                "pending_decision_count": len(decisions),
                "decision": None if selected is None else self._decision_view(selected),
                "observation": observation,
                "terminal_result": (
                    None
                    if state.terminal_result is None
                    else cast(JsonValue, terminal_payload(state.terminal_result))
                ),
                "cards": cast(JsonValue, self._card_catalog()),
                "special_achievements": cast(JsonValue, self._special_achievements()),
                "history": cast(JsonValue, self._history(state.phase.value)),
            }

    def _decision_view(self, decision: Decision) -> JsonObject:
        payload = decision_payload(decision)
        payload["legal_actions"] = [
            {
                "payload": cast(JsonValue, action_payload(action)),
                "label": self._action_label(action),
                "card_id": self._action_card_id(action),
            }
            for action in decision.legal_actions
        ]
        return cast(JsonObject, payload)

    def _action_card_id(self, action: SemanticAction) -> str | None:
        card_id = getattr(action, "card_id", None)
        return card_id.value if isinstance(card_id, CardId) else None

    def _card_name(self, card_id: CardId) -> str:
        return self.registry.card(card_id).name.title()

    def _action_label(self, action: SemanticAction) -> str:
        if isinstance(action, ChooseStartingMeldAction):
            return f"Start with {self._card_name(action.card_id)}"
        if isinstance(action, DrawAction):
            return "Draw"
        if isinstance(action, MeldAction):
            return f"Meld {self._card_name(action.card_id)}"
        if isinstance(action, DogmaAction):
            return f"Dogma {self._card_name(action.card_id)}"
        if isinstance(action, AchieveAction):
            return f"Achieve age {action.achievement_id.value.rsplit('-', 1)[-1]}"
        if isinstance(action, ChooseCardAction):
            return f"Choose {self._card_name(action.card_id)}"
        if isinstance(action, ChooseCardsAction):
            names = ", ".join(self._card_name(card_id) for card_id in action.card_ids)
            return f"Choose {names}" if names else "Choose no cards"
        if isinstance(action, ChooseColorAction):
            return f"Choose {action.color.value}"
        if isinstance(action, ChoosePlayerAction):
            return f"Choose {_title(action.player_id.value)}"
        if isinstance(action, ChooseValueAction):
            return f"Choose {action.value}"
        if isinstance(action, ChooseSplayAction):
            return f"Splay {action.direction.value}"
        if isinstance(action, ChooseBranchAction):
            return _title(action.branch_id)
        if isinstance(action, OrderCardsAction):
            names = " → ".join(self._card_name(card_id) for card_id in action.card_ids)
            return f"Order {names}"
        if isinstance(action, DeclineAction):
            return "Decline"
        if isinstance(action, FinishSelectionAction):
            return "Finish selection"
        raise TypeError(f"unsupported action type: {type(action).__name__}")

    def _card_catalog(self) -> JsonObject:
        return {
            card.id.value: {
                "name": card.name.title(),
                "age": card.age,
                "color": card.color.value,
                "featured_icon": card.featured_icon.value,
                "icons": [icon.value for icon in card.functional_icons],
                "dogma": [effect.text for effect in card.dogma_effects],
            }
            for card in self.registry.cards
        }

    def _special_achievements(self) -> JsonObject:
        return {
            achievement.id.value: {
                "name": achievement.name,
                "condition": achievement.source_condition,
            }
            for achievement in self.registry.special_achievements.values()
        }

    def _history(self, phase: str) -> list[JsonObject]:
        return [
            {
                "number": number,
                "label": (
                    "Secret starting meld selected"
                    if phase == "starting-melds" and isinstance(action, ChooseStartingMeldAction)
                    else self._action_label(action)
                ),
                "kind": action.kind.value,
            }
            for number, action in enumerate(self._actions, start=1)
        ]
