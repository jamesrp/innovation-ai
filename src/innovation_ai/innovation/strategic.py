"""Strategic-position payloads and digests for cycle/transposition detection.

Unlike the authoritative replay hash, this key intentionally ignores setup provenance, absolute
turn number, and monotonic allocator IDs.  It retains all card locations and ordering, current-turn
state, effect progress, hidden state, and terminal state, so equality remains conservative enough
for route-local search cutoffs.
"""

from __future__ import annotations

import hashlib
import json
from typing import cast

from innovation_ai.innovation.state import GameState, state_payload

STRATEGIC_STATE_DIGEST_VERSION = "strategic-state-v1"

_TOP_LEVEL_NON_STRATEGIC_FIELDS = frozenset(
    {
        "turn_number",
        "starting_meld_decision_ids",
        "next_decision_id",
        "next_event_id",
        "next_dogma_action_id",
        "setup",
    }
)
_FRAME_TURN_ID_FIELDS = frozenset({"turn_id"})
_FRAME_DOGMA_ID_FIELDS = frozenset({"dogma_action_id", "dogma_action"})


def _normalize_effect_runtime(payload: dict[str, object]) -> None:
    """Normalize only effect-frame fields known to copy monotonic state IDs.

    Semantic IDs such as card, program, and node IDs are retained.  Progress counters, nested-scope
    ordinals, and arbitrary card-program variables are also retained.  Dogma-action aliases are
    mapped by first appearance so equality relationships survive normalization rather than every
    integer-like value being collapsed indiscriminately.
    """

    raw_frames = payload.get("pending_effects")
    if not isinstance(raw_frames, list):  # pragma: no cover - produced by trusted state_payload
        raise TypeError("strategic state pending_effects is not an array")

    dogma_ids: dict[int, str] = {}
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):  # pragma: no cover - produced by trusted state_payload
            raise TypeError("strategic state effect frame is not an object")
        raw_variables = raw_frame.get("variables")
        if not isinstance(raw_variables, list):  # pragma: no cover - trusted payload
            raise TypeError("strategic state frame variables is not an array")
        for raw_variable in raw_variables:
            if not isinstance(raw_variable, dict):  # pragma: no cover - trusted payload
                raise TypeError("strategic state frame variable is not an object")
            name = raw_variable.get("name")
            value = raw_variable.get("value")
            if (
                name in _FRAME_TURN_ID_FIELDS
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                raw_variable["value"] = "<turn-id>"
            elif (
                name in _FRAME_DOGMA_ID_FIELDS
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                token = dogma_ids.setdefault(value, f"<dogma-action-{len(dogma_ids)}>")
                raw_variable["value"] = token


def strategic_state_payload(state: GameState) -> dict[str, object]:
    """Return the canonical gameplay payload used for strategic equality.

    Supplies and all private identities are deliberately included: this is an authoritative,
    route-local search/debug key, not a player-safe observation or publishable trace field.
    """

    payload = state_payload(state)
    for name in _TOP_LEVEL_NON_STRATEGIC_FIELDS:
        payload.pop(name, None)
    payload["strategic_digest_version"] = STRATEGIC_STATE_DIGEST_VERSION
    _normalize_effect_runtime(payload)
    return payload


def strategic_state_json(state: GameState) -> str:
    """Serialize :func:`strategic_state_payload` with deterministic formatting."""

    return json.dumps(
        strategic_state_payload(state),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def strategic_state_digest(state: GameState) -> str:
    """Return a tagged SHA-256 digest of the strategic gameplay position."""

    encoded = strategic_state_json(state).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


# Transposition and diagnostic call sites use both terms; they are intentionally identical.
strategic_state_hash = strategic_state_digest
strategic_position_digest = strategic_state_digest


def is_strategically_equal(first: GameState, second: GameState) -> bool:
    """Compare payloads directly, avoiding even the theoretical risk of a digest collision."""

    return cast(object, strategic_state_payload(first)) == strategic_state_payload(second)
