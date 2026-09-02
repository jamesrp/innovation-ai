# AGENTS.md

## Project purpose and current milestone

This repository implements **Innovation (base game, Third Edition), two players**, and is building
toward a strong self-play-trained Innovation AI. **Milestones 1–3 are complete:** the deterministic
engine covers all 105 supplied cards, the first learned-value pipeline includes encoder v1, compact
replay training, information-safe afterstates, frozen self-play, paired arenas, and CPU profiling,
and the first measured training pilot exposed and diagnosed a deterministic policy cycle.
**Milestone 4 is current:** build a player-safe sampled minimax heuristic, adopt public board-card
information, harden pathological-game observability, restore temperature softmax as the primary
learned selector, retrain, and reevaluate.

Read these before changing engine or ML behavior:

1. `PROJECT_GOAL.md` — binding architecture and milestone goals.
2. `game-rules-plaintext/innovation_2p_base_rules.md` — authoritative supplied rules.
3. `game-rules-plaintext/cards.csv` — authoritative supplied card list and dogma text.
4. `game-rules-plaintext/special_achievements.csv` — achievement index; where abbreviated
   text differs, the full rules and card text govern.
5. `docs/MILESTONE_1_PLAN.md` — completed engine implementation contracts and history.
6. `docs/MILESTONE_2_PLAN.md` — completed first learned-value milestone contracts and gates.
7. `docs/MILESTONE_2_REPORT.md` — implementation evidence, commands, profiling, and limitations.
8. `docs/MILESTONE_3_TRAINING_PROTOTYPE.md` — completed measured-pilot experiment sequence.
9. `docs/MILESTONE_3_REPORT.md` — pilot, cycle, selector experiment, profiling, and limitations.
10. `docs/MILESTONE_4_PLAN.md` — current sampled-search heuristic, observability, retraining, and
    evaluation plan.

Do not silently resolve a rules ambiguity. Record the interpretation and a focused test; if
it materially changes gameplay or observations, request a project decision first.

## Environment

- Python 3.12+
- `uv` owns dependencies, the lockfile, and command execution.
- Core engine code uses the standard library and must not import NumPy, PyTorch, or agent code.
- NumPy and CPU-only PyTorch live in the optional `ai` dependency group for ML work.
- The virtual environment is `.venv/` and is never committed.
- Run commands from the repository root.

## Setup, build, run, and test

```bash
make install       # core package and development dependencies
make install-ai    # also install NumPy and CPU-only PyTorch
make run           # run the environment doctor CLI
make test          # pytest with branch coverage
make lint          # ruff checks and formatting check
make format        # ruff autofixes and formatter
make typecheck     # strict mypy
make check         # lint + typecheck + test; required before handoff
```

Equivalent direct commands:

```bash
uv sync
uv sync --extra ai
uv run innovation-ai doctor
uv run pytest --cov=innovation_ai --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

There is no separate compilation step. Keep `uv.lock` committed whenever dependencies change.

## Intended repository layout

Create paths only when their work package begins; avoid empty scaffolding.

- `src/innovation_ai/innovation/`: Innovation rules, state, card registry, and effects
- `src/innovation_ai/agents/`: random, scripted, heuristic, and later learned policies
- `src/innovation_ai/harness/`: game runners, batching, logs, replay, and evaluation
- `src/innovation_ai/training/`: ML encoding, models, compact replay, training, inference, and
  self-play code added by Milestone 2 work packages
- `tests/`: mirrors package layout; card tests are partitioned to avoid agent conflicts
- `game-rules-plaintext/`: immutable human-readable source material supplied by the user
- `docs/`: architecture contracts, rules decisions, and implementation plans
- `artifacts/`: generated checkpoints/metrics; ignored except deliberately committed fixtures

## Engine contract

- Model play as an explicit, resumable state machine.
- Every player choice—including setup and choices nested in dogma effects—must yield a
  first-class `Decision` with acting player, player-safe observation, and legal actions.
- Applying one semantic `Action` advances deterministically to the next `Decision` or a
  terminal result. The engine never calls an agent and card effects never use callbacks into
  agents or UIs.
- Actions use stable semantic identifiers such as card IDs, player IDs, colors, and operation
  kinds. Never encode choices as display strings or transient legal-action list indices.
- The public transition boundary must be observably pure: input state is not changed. Internal
  implementation may use controlled mutation or structural sharing, but partially resolved
  state must remain serializable and reproducible.
- Randomness is explicit and consumed during setup to create ordered supplies. In-play state
  transitions depend only on state plus action.
- Illegal agent choices raise a typed recoverable error; violated engine invariants raise a
  distinct engine-bug error.

## State, effects, and observations

- Keep authoritative state separate from every player observation. An observation must never
  retain a reference to authoritative state.
- Hidden deck order, normal-achievement identities, and opponent private card identities must
  not leak. Follow the configured information policy for public values/counts and covered board
  information, and record that policy in logs.
- Preserve ordered effect resolution, frozen dogma icon counts, opponent-first sharing,
  demand immunity, partial execution, sharing bonuses, and immediate termination exactly as
  specified by the rules.
- Use shared movement/effect primitives with provenance rather than unrelated card-specific
  mutation code. Provenance must distinguish score/tuck operations from transfers/exchanges,
  track demand-caused changes, and support Monument and sharing-bonus rules.
- Natural-language card text is reference data, not a runtime programming language. Implement
  structured, reviewable effects keyed by canonical card ID; do not attempt to infer full card
  semantics by parsing prose at runtime.
- Check invariants and immediate achievements at defined atomic-operation boundaries. Operations
  intended to be simultaneous must not expose invalid intermediate states.

## Serialization, replay, and batching

- State, pending effect frames, decisions, actions, terminal results, and logs must use versioned
  schemas with deterministic ordering.
- Logs record seed/setup, rules and information-policy versions, semantic actions, card-data
  fingerprint, and per-step state hashes. Replay must verify those hashes.
- Keep engine execution, observation construction, agent selection, logging, and evaluation
  loosely coupled.
- The runner pulls pending decisions and submits actions. Its protocol must allow decisions from
  many independent games to be collected and sent to a future external batch policy.

## Python style

- Use type hints on public APIs and maintain strict `mypy` compliance.
- Prefer frozen, slotted dataclasses/enums for IDs, actions, decisions, observations, and other
  leaf values. Do not force deep immutable-copy boilerplate where it obscures rule correctness.
- Use `pathlib.Path` and `importlib.resources`; never depend on the current directory to load
  packaged runtime data.
- Prefer small explicit functions and composition over deep inheritance.
- Avoid mutable defaults, wildcard imports, import-time side effects, and nondeterministic set or
  dict iteration in serialized or legal-action output.
- Libraries use structured logging, not `print`; the CLI may render output.
- Public modules, types, contracts, and non-obvious rule interpretations need concise docstrings.

## Testing requirements

- Every rule, card effect, ambiguity decision, and bug fix needs a focused test.
- Use table-driven tests for rule matrices and card branches.
- Test every decision's legal-action set, including decline/stop, partial execution, tied
  highest/lowest choices, ordering choices, and hidden-information boundaries.
- Maintain invariants including card conservation, unique card location, score consistency,
  visible-icon geometry, valid turns, achievement uniqueness, and no mutation after terminal.
- Add deterministic replay tests and property/fuzz tests. Normal tests must remain fast and must
  not run long training jobs.
- Each card needs no-effect/minimal, ordinary, branch/choice, sharing or demand, and relevant
  termination/achievement coverage.
- Before handing off work, run `make check` and report any unrun slow/fuzz suite separately.

## Agent work and dependency policy

- Follow work-package dependencies and file ownership in `docs/MILESTONE_1_PLAN.md`.
- Freeze shared contracts before parallel card implementation. Card agents must not independently
  invent new action schemas, effect primitives, or state fields.
- Keep commits focused and do not mix broad refactors with card behavior.
- Add runtime dependencies only when the standard library is inadequate. Milestone 1 engine code
  must remain independent of ML frameworks.
- Use `uv add`, `uv add --dev`, or `uv add --optional ai` when changing dependencies; commit
  `pyproject.toml` and `uv.lock` together.
- Never commit secrets, `.env` files, virtual environments, generated datasets, or model
  checkpoints.
- Update `AGENTS.md`, `README.md`, and the relevant contract/decision document when commands,
  architecture, or interpretations change.
- Push completed commits to `origin` before handoff unless the user asks otherwise or the remote
  is unavailable; report any push failure explicitly.
