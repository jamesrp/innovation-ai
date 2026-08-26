# Project Goal

Build a strong, reproducible, and inspectable AI for the two-player base game of **Innovation**—not
just a rules simulator. The broader aim is a complete playable opponent and research platform in
the spirit of [Keldon Jones's 2009 Race for the Galaxy AI project](https://www.keldon.net/rftg/):
combine a faithful game implementation with self-play, learned evaluation, rigorous comparison,
and artifacts that let results be reproduced and improved over time. The project is inspired by
that end-to-end ambition rather than committed to reproducing Keldon's exact methods.

The deterministic engine foundation is complete for all 105 Third Edition base-game cards in the
two-player game. Setup randomness is explicit; every setup, paid-action, and nested dogma choice
is a first-class decision; transitions are deterministic and replayable; and authoritative state
is strictly separated from player-visible observations. Random, scripted, and simple heuristic
agents plus single-game and pull-based multi-game runners establish non-ML baselines and the batch
orchestration seam.

The next goal is the first learned baseline: a viewpoint-relative flat observation encoder, a small
PyTorch value network, terminal-outcome training from compact replay data, information-safe
one-ply afterstate selection, frozen-checkpoint iterative self-play, and statistically grounded
arena evaluation. It must run usefully on a CPU-only exe.dev development box while retaining clean
paths to parallel actors and GPU-backed training or inference.

Preserve the engine as an explicit state machine. Every player choice remains a semantic
`Decision`, and applying one semantic `Action` advances to the next decision or terminal result.
The engine never calls agents, parses card prose at runtime, or exposes model tensors. Learned
components consume only versioned player-safe observations and public decision context; trusted
orchestration may operate the engine but must not allow candidate evaluation to exploit hidden
deck order, achievements, hands, covered cards, or secret setup choices.

Keep rules execution, observation and public-context construction, encoding, model definition,
training, inference, self-play orchestration, replay storage, evaluation, and profiling loosely
coupled. Actions remain stable structured data rather than display strings or legal-action
indices. Every dataset, checkpoint, run, and arena report records enough schema versions,
fingerprints, seeds, and configuration to reject incompatible inputs and reproduce its result.

The long-term measure of success is an Innovation AI that becomes meaningfully stronger through
iterative self-play while remaining fair, testable, auditable, and practical to run—not merely a
neural-network demo attached to the rules engine.
