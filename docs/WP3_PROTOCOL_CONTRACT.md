# WP3 Action, Decision, Observation, and Turn Contract

WP3 freezes the player-facing protocol used by agents, runners, replay, and later effect work.
All contracts are frozen, slotted dataclasses with stable semantic identifiers and canonical
JSON-compatible payload helpers.

## Decisions and actions

`current_decisions(state)` returns every pending decision in deterministic order. Setup exposes
one decision per player simultaneously; ordinary play exposes one decision for the active player.
`current_decision(state)` is the single-decision convenience API.

Every action carries the exact decision ID it answers. Paid actions are `DrawAction`,
`MeldAction(card_id)`, `DogmaAction(card_id)`, and `AchieveAction(achievement_id)`. Setup and
future effect choices use dedicated semantic action classes for cards, subsets, colors, players,
values, splay directions, branches, card order, decline, and bounded-selection finish. Display
strings and legal-list indices are never action identity.

Legal actions are ordered by rule action type, then canonical zone or enum order. Submission is
validated by exact membership in the decision's legal-action tuple. Agent mistakes raise
`IllegalAction`; invalid engine protocol states raise `EngineInvariantError`.

## Setup and turns

The two starting-meld decisions have stable IDs 1 and 2 and may arrive in either order. A submitted
choice remains authoritative but hidden; neither board changes until both choices exist. The two
cards then meld atomically, and printed card-title order selects the first player. That player's
first turn has one paid action. The opponent's first turn and every later turn have two.

A paid Dogma selection reserves one paid action and installs a serializable `dogma-action` handoff
frame. WP4 owns resolving that frame. The turn cannot rotate while any effect frame remains;
`finish_effect_resolution` advances only after WP4 has cleared all frames. This preserves honest
Dogma legality without treating unimplemented effects as no-ops.

## Observations

`observe(state, viewer, policy=...)` constructs a detached immutable projection and never embeds
or filters authoritative state. Under the default `rulebook-private-covered-v1` policy:

- supplies expose age and count, never order or identities;
- normal-achievement card identities never appear;
- both players' hand and score values are public, but only the owner sees card IDs;
- all own board identities are visible;
- opponent top-card IDs and splay geometry are public;
- opponent covered IDs are hidden, and an unsplayed stack's covered count is `None`;
- visible icon fragments of opponent splayed covered cards remain observable.

`public-covered-v1` is also supported for groups choosing the rulebook's open covered-card option.
Leak tests compare positions differing only by hidden identities and require equal observations
(and stable setup decisions) under the private policy.

## Terminal and serialization contracts

`TerminalResult` uses `TerminalReason` and canonical winner tuples; no winners means a draw.
Actions, decisions, observations, terminal results, and authoritative state all carry schema or
rules/policy versions. `action_payload` and `decision_payload` produce deterministic semantic
payloads suitable for logs and process boundaries.
