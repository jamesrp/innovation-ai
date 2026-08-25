# WP2 State and Geometry Contract

WP2 stores the complete authoritative position in immutable, slotted dataclasses. Public zone
primitives return a new `GameState` plus a semantic `ChangeRecord`; the input state is never
mutated.

## Ordering

- Supply piles are ordered **top to bottom**.
- Board stacks are ordered **bottom to top**.
- Setup provenance preserves each complete shuffled pile before achievements and deals.
- Hands, score piles, removed cards, and claimed-achievement tuples are rule-unordered and are
  canonicalized by stable semantic ID for hashing and tied-choice enumeration.
- Boards, players, colors, supply ages, and normal achievements use their enum order.

## Setup

Seeded setup starts each age from card-ID order and uses the versioned
`python-mt19937-shuffle-v1` algorithm. It removes the top card of ages 1-9 as the hidden normal
achievements, then deals one age-1 card to each player in player order for two rounds. The full
shuffled order and deal sequence are recorded. `build_setup_state_from_piles` reconstructs setup
from explicit pile order so replay need not depend on an RNG implementation.

## Geometry and movement

Top cards expose all three functional icons. Covered cards expose bottom-right when splayed left,
top-left and bottom-left when splayed right, and all three bottom positions when splayed up. A
stack reduced to zero or one card becomes unsplayed immediately; movement records include that
geometry change. Melding adds to the top and tucking adds to the bottom while preserving a valid
existing splay.

Returns append to the bottom of the matching age supply. Draws search upward only, with requested
value zero or lower beginning at age 1. Removed cards cannot return to play. Exchanges are atomic,
permit either or both selected sides to be empty, and require distinct non-achievement,
non-removed locations.

## Invariants and hashing

Every primitive validates registry fingerprint compatibility, conservation of all 105 cards,
unique card location, supply ages, normal-achievement ages, board colors, splay collapse, and
unique achievement ownership. Terminal states reject further mutation. The canonical full-state
JSON payload is hashed with SHA-256; setup provenance, pending frames, counters, IDs, versions,
and terminal data are included.

WP4 will extend change records with causal dogma provenance and compose these atomic zone
operations into effect primitives. WP3 owns semantic actions, decisions, observations, and legal
transition enforcement above this layer.
