# Innovation AI

A side project for building a deterministic card-game rules engine, followed by a small
PyTorch agent trained on CPU.

## Quick start

```bash
make install       # Core package and development tools
make install-ai    # Also install CPU-only PyTorch
make check         # Lint, type-check, and test
make run           # Verify the environment
```

Direct equivalents are available through `uv`, for example:

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
uv run card-game-ai doctor
```

See `AGENTS.md` for architecture and contribution conventions.
