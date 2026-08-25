# WP9 Serialization, Log, and Replay Contract

WP9 freezes the Milestone 1 version-1 JSON boundary available at Freeze A. The implementation is
standard-library-only and lives in `innovation/serialization.py`, `logs.py`, and `replay.py`.

## Deterministic schemas

Authoritative states, semantic actions, decisions (including observations), and terminal results
round-trip through strict decoders. Every top-level contract carries its existing schema version;
unknown versions, missing fields, and unexpected fields are rejected. JSON output is UTF-8-safe,
ASCII-escaped, key-sorted, and compact, so equal contracts produce byte-identical output.

State loading reconstructs every current nested value, including setup provenance, counters,
effect variables, and the Freeze-A placeholder `EffectFrameState`. By default it checks the rules
version, information-policy version, packaged card-data fingerprint, card conservation, unique
locations, and other state invariants. This means a paused Dogma handoff state can be saved and
loaded now; WP4's richer frame variants must be added to this versioned decoder when merged.

## Game logs

A version-1 game log records:

- format, package-engine, rules, information-policy, and every component schema version;
- the packaged card-data fingerprint;
- setup seed, setup RNG convention, explicit shuffled piles, and deal order;
- the full versioned decision and submitted semantic action for every transition;
- a SHA-256 authoritative state hash after every submitted action;
- transition count and contiguous sequence numbers;
- initial/final state hashes, final boundary kind, and terminal result when present.

Explicit shuffled piles, rather than the seed alone, are authoritative during replay. The seed and
RNG label remain provenance. Logs may intentionally end at a decision or serializable pending-
effect boundary, but their final marker must match. A missing tail, edited action/decision, stale
hash, unsupported version, changed fingerprint, or mismatched terminal marker fails loudly.
Hashes detect accidental or unsophisticated editing; logs are not signed and do not claim
adversarial tamper proofing.

## Replay and effect integration hook

`ReplayAdapter` isolates log/replay from the concrete transition executor. The Freeze-A
`DefaultReplayAdapter` reconstructs explicit setup, obtains decisions from `current_decisions`, and
applies actions through `apply_action`. It preserves the current Dogma handoff frame and classifies
that state as `effect-resolution-pending`.

When WP4 lands, its integration owner should either extend `DefaultReplayAdapter.apply` or provide
a replacement adapter that resolves serializable frames to the next decision/terminal boundary.
The adapter must return the authoritative state whose hash belongs to that semantic action. Add
strict decoding for every new frame kind/field without relying on callbacks, generators, or Python
class names. No log schema change is needed merely to add frame behavior already represented
inside the versioned state schema; incompatible frame shape changes require a state-schema bump.

Card/effect integration must additionally verify that replay from a mid-effect checkpoint resumes
with the same subsequent decision and hashes. The current recorder deliberately requires the
initial explicit setup boundary; future checkpoint logs should add a separately versioned initial-
state/checkpoint record rather than overloading setup replay.

## CLI

`innovation-ai doctor` is retained. `innovation-ai play --seed N --log FILE` currently runs a
complete deterministic first-legal-action baseline (starting choices followed by Draw actions) and
writes a terminal log. It is intentionally a minimal Freeze-A adapter, not the WP8 human/agent
runner. `innovation-ai replay FILE` performs all compatibility and hash checks and returns a
nonzero status on failure.
