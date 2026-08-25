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

Effect choices use the same boundary. While a dogma or nested effect is waiting on a player,
`current_decisions(state)` returns exactly one `EFFECT_CHOICE` decision and `apply_action()` routes
its semantic action back into the resumable VM. A `DecisionContext` records demand/shared/nested
flags, frozen dogma icon counts, selection bounds, and incremental progress. Deterministic
low-level checkpoints created by `step_effect()` are resumed explicitly with
`resume_pending_effects()`.

## Observations

`observe(state, viewer, policy=...)` constructs a detached immutable projection and never embeds
or filters authoritative state. Under the default `rulebook-private-covered-v1` policy:

- supplies expose age and count, never order or identities;
- normal-achievement card identities never appear;
- both players' hand and score values are public, but exact IDs are visible only to the owner or
  while the cards carry authoritative face-up reveal markers;
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
rules/policy versions. State schema 2 includes multi-scope physical reveal markers; decision
schema 3 includes effect context plus explicit incremental-selection purpose. `action_payload` and
`decision_payload` produce deterministic semantic payloads suitable for logs and process
boundaries.
