# Project Roadmap

## Long-term aim

Build a strong, reproducible, inspectable Innovation opponent and self-play research platform in
the spirit of Keldon Jones's 2009 Race for the Galaxy AI project. The project should produce a
useful playable AI, not only an engine or isolated model experiment. See `PROJECT_GOAL.md`.

## Milestone 1 — complete

Complete the deterministic two-player Innovation base-game engine, all 105 cards, agents,
runners, replay, and invariant/fuzz coverage as specified in `docs/MILESTONE_1_PLAN.md`.

## Milestone 2 — complete

Delivered the first end-to-end learned baseline: audited random/heuristic agents, a flat
viewpoint-relative encoder, the `D -> 128 tanh -> 1 sigmoid` PyTorch value network,
terminal-outcome training from compact replay data, information-safe batched one-ply afterstate
selection, frozen-checkpoint self-play, paired arena evaluation, and CPU profiling. See
`docs/MILESTONE_2_PLAN.md` and `docs/MILESTONE_2_REPORT.md`.

## Milestone 3 — complete: training prototype

The frozen Milestone 2 pipeline completed its first measured medium pilot, deterministic arena
preflights, cycle diagnosis, and a controlled selector experiment. The August 29–30, 2026 pilot had
healthy replay/data integrity and useful held-out learning. Its original temperature-softmax policy
reached a 10,000-action cycle against the deliberately weak heuristic at seed 50000; the versioned
repetition-aware variant completed the subsequent preflight but did not establish the original
cycle's root cause. See `docs/MILESTONE_3_TRAINING_PROTOTYPE.md` and
`docs/MILESTONE_3_REPORT.md`.

## Milestone 4 — current: player-safe search heuristic and retraining

Replace the simple printed-card heuristic with a deterministic sampled minimax policy that searches
approximately two complete rounds through the real engine from player-safe synthetic states. Use it
as both the primary heuristic baseline and the learned policy's setup/effect-choice fallback.

Milestone 4 also:

- adopts `public-covered-v1` so every ordered board-card identity is public even when unsplayed,
  while splay geometry still determines functional icon counts;
- restores `temperature-softmax-v1` as the primary learned selector and retains
  `recent-paid-action-penalty-v1` only as an experimental comparator;
- adds complete compressed traces and selection/search telemetry for the original seed-50000
  reproduction and later pathological games;
- validates that the new heuristic is stronger and computationally usable before generation;
- performs a fresh training run under the changed information and fallback policies; and
- reruns fixed seed-50000 diagnostics and paired preflights against random, the old heuristic, and
  the new search heuristic.

See `docs/MILESTONE_4_PLAN.md`.

## Future information-memory policy

Milestone 4 makes every ordered board-card identity public under `public-covered-v1`, including
cards in unsplayed stacks. Functional icon counts still follow splay geometry. The remaining future
information-memory problem therefore concerns hidden hands, score piles, supplies, achievements,
and secret choices rather than covered-board identities.

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
