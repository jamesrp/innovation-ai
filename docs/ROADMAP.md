# Project Roadmap

## Milestone 1

Complete the deterministic two-player Innovation base-game engine, all 105 cards, agents,
runners, replay, and invariant/fuzz coverage as specified in `docs/MILESTONE_1_PLAN.md`.

## Future information-memory policy

Milestone 1 observations expose current physical information only: cards and attributes that are
literally face-up, public board information, and identities in zones the viewer may inspect.

A later milestone may add a versioned information-memory policy that tracks **definite identity
knowledge** across hidden-zone movement. For example, if a player returns Road Building to a known
supply position, follows that card through deterministic draws into an opponent's hand, and no
ambiguating operation occurs, their observation could continue identifying its location. Once an
operation makes the identity's location uncertain—such as the opponent scoring one of several
same-valued hand cards—the engine should stop asserting a definite location.

This future policy should initially track only facts of the form “the viewer definitely knows this
card is at this location.” It should not attempt arbitrary information-set inference such as
probability distributions or disjunctions over several possible hands/score piles. Human agents
may keep their own notes, and learned agents may encode strategic memory independently.

Requirements when implemented:

- separate authoritative location from each viewer's knowledge state;
- update knowledge deterministically after every movement and hidden choice;
- forget facts conservatively whenever identity-location certainty is lost;
- serialize and version knowledge state and record its policy in logs;
- add observation non-leak and replay tests for remembered and forgotten identities.
