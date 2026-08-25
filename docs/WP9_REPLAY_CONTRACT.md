# WP9 Serialization, Log, and Replay Contract

WP9 provides the strict JSON, game-log, and replay boundary used by the Freeze-B engine. The
implementation is standard-library-only and lives in `innovation/serialization.py`, `logs.py`, and
`replay.py`.

## Deterministic schemas

Authoritative states, semantic actions, decisions (including observations and effect context), and
terminal results round-trip through strict decoders. Unknown versions, missing fields, unexpected
fields, malformed effect frames, stale reveal/runtime relationships, and terminal states retaining
transient runtime are rejected.

State schema 2 includes serializable program/node/dogma frames, scoped effect variables, and
multi-scope physical reveal markers. Observation schema 3 exposes currently revealed identities
and public revealed card attributes such as Classification's announced color. Decision schema 3
carries demand/shared/nested flags, frozen dogma icon counts, selection bounds,
incremental progress, and an explicit incremental-selection purpose that distinguishes bounded
subsets from anti-factorial choose-next card ordering. Equal contracts produce byte-identical
compact, key-sorted JSON.

Loading checks rules and information-policy versions, the packaged card-data fingerprint, card
conservation, locations, geometry, achievement ownership, and effect-runtime structure. A paused
player decision or low-level deterministic effect checkpoint can therefore be saved and restored;
`resume_pending_effects()` advances the latter to the next public boundary.

## Game logs

Game-log schema 2 records:

- format, package-engine, rules, information-policy, and every component schema version;
- card-data and effect-program fingerprints;
- setup seed, RNG convention, explicit shuffled piles, and deal order;
- the full versioned decision and submitted semantic action for every transition;
- a SHA-256 authoritative state hash after every submitted action;
- transition count and contiguous sequence numbers;
- initial/final hashes, final decision-or-terminal boundary kind, and terminal result when present.

There is no `effect-resolution-pending` replay outcome: public `apply_action()` always resolves
Dogma deterministically to a player decision, abort completion, normal completion, or terminal
result. Logs may end at a decision or terminal boundary, and the final marker must match.

Explicit shuffled piles, rather than the seed alone, are authoritative during replay. Edited or
truncated decisions/actions, stale hashes, unsupported versions, changed fingerprints, malformed
runtime, or mismatched terminal markers fail loudly. Hashes detect accidental or unsophisticated
editing; logs are not signed and do not claim adversarial tamper proofing.

## Replay and effects

`DefaultReplayAdapter` reconstructs explicit setup, gets every setup/paid/effect decision from
`current_decisions()`, and submits actions through `apply_action()`. Mid-dogma choices therefore
replay through the same public transition contract as live agents and runners.

The `effects_fingerprint` is a SHA-256 digest over canonical declarative programs and validated
named-helper implementations. Any implemented card behavior change invalidates older logs rather
than allowing a silent state-hash divergence. All 105 catalog cards are registered; an absent
program remains a typed error, never a no-op. Behavior-affecting interpreter or primitive changes
also require an explicit package engine-version bump because those implementations are outside the
declarative effects fingerprint.

The recorder begins from explicit setup. Future arbitrary checkpoint logs should add a separately
versioned initial-state/checkpoint record instead of overloading setup replay.

## CLI

`innovation-ai doctor` is retained. `innovation-ai play --seed N --log FILE` runs the deterministic
baseline and writes a terminal log. `innovation-ai replay FILE` performs all compatibility,
fingerprint, decision, and hash checks and returns nonzero on failure.
