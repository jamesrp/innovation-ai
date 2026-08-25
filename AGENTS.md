# AGENTS.md

## Project purpose

Build a correct, deterministic card-game rules engine first, then add small PyTorch models
that learn from or act through that engine. Development and training must work on CPU and
within modest memory. Correctness, reproducibility, and clear state transitions matter more
than early performance optimization.

## Environment

- Python 3.12+
- `uv` owns Python dependencies, the lockfile, and command execution.
- PyTorch comes from the official CPU wheel index; do not introduce CUDA dependencies.
- The virtual environment is `.venv/` and is never committed.
- Run commands from the repository root.

## Setup, build, run, and test

```bash
make install       # uv sync: core package + development dependencies
make install-ai    # uv sync --extra ai: includes CPU-only PyTorch
make run           # run the environment doctor CLI
make test          # pytest with branch coverage
make lint          # ruff check
make format        # ruff formatter and safe autofixes
make typecheck     # strict mypy
make check         # lint + typecheck + test
```

Equivalent direct commands:

```bash
uv sync --extra ai
uv run card-game-ai doctor
uv run pytest --cov=card_game_ai --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

There is no separate compilation step. `uv sync` creates the environment and installs the
package in editable form. Keep `uv.lock` committed and update it whenever dependencies change.

## Repository layout

- `src/card_game_ai/`: application package
- `src/card_game_ai/engine/`: game-neutral and game-specific rules/state transitions
- `src/card_game_ai/agents/`: baseline, search, and learned policies
- `src/card_game_ai/training/`: data generation, self-play, training, and evaluation
- `tests/`: mirrors the package layout
- `artifacts/`: generated checkpoints/metrics; ignored by Git except small fixtures

Create subpackages only when implementing them; avoid empty architecture scaffolding.

## Rules-engine conventions

- Keep state transitions pure: `next_state = apply_action(state, action)`.
- Prefer frozen dataclasses, enums, tuples, and other immutable values for game state.
- Validate actions at the engine boundary and distinguish illegal actions from engine bugs.
- Make randomness explicit by passing a `random.Random`, NumPy generator, or seed. Never rely
  on hidden module-global RNG state.
- A state plus RNG seed must reproduce a simulation exactly.
- Keep rules independent from CLI, rendering, persistence, networking, PyTorch, and training.
- Represent player-visible observations separately from omniscient engine state to prevent
  accidental information leakage.
- Define stable action ordering and state encoding before training models against them.

## AI and training conventions

- Start with simple baselines (random/legal-action policy and heuristic policy) before neural
  models; learned agents must be compared against them.
- Mask illegal actions before action selection and test the mask independently.
- Keep tensor conversion at the boundary between engine and agent code.
- Seed Python, NumPy, and PyTorch in training entry points. Record seed and configuration with
  every result.
- Default to tiny networks, short runs, and CPU. A normal test suite must never launch a long
  training job.
- Save generated checkpoints and large datasets under `artifacts/`, not in Git. Use Git LFS
  only when a small, essential binary fixture genuinely belongs in the repository.
- Report evaluation over multiple seeds/games; do not treat training loss alone as game skill.

## Python style

- Use type hints on public APIs and maintain strict `mypy` compliance.
- Use `pathlib.Path`, not string-based path manipulation.
- Prefer small explicit functions and composition over deep class hierarchies.
- Avoid mutable default arguments, wildcard imports, and import-time side effects.
- Public modules, classes, and non-obvious algorithms need concise docstrings.
- Use structured logging for training and diagnostics; libraries must not call `print`.
- Keep functions deterministic unless mutation or I/O is central to their documented purpose.

## Testing

- Every rule and bug fix needs a focused test.
- Table-driven tests are preferred for rule matrices and edge cases.
- Add invariant/property tests when useful: card conservation, valid turn progression, score
  bounds, legal-action completeness, and terminal-state behavior.
- Tests must be deterministic and fast. Mark any future integration/slow tests explicitly.
- Test both valid play and rejection of illegal actions.
- Before finishing a change, run `make check`.

## Dependency and change policy

- Add runtime packages only when the standard library or current dependencies are inadequate.
- Use `uv add <package>` or `uv add --dev <package>` rather than hand-editing versions when
  practical, then commit `pyproject.toml` and `uv.lock` together.
- Keep commits focused. Do not mix broad refactors with rule changes.
- Never commit secrets, `.env` files, virtual environments, generated datasets, or model
  checkpoints.
- Update this file and `README.md` when commands or architecture conventions change.
