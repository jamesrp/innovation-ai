# Innovation AI

A deterministic Python implementation of the two-player Innovation base game, designed first for
reliable play between basic agents and later for self-play and PyTorch research.

## Current milestone

**Milestone 1:** implement the complete rules engine for all 105 supplied cards. Every setup,
turn, and nested card-effect choice will be exposed as a serializable first-class decision, with
strict separation between authoritative state and player-visible observations.

The engine foundation now includes WP1's validated immutable card catalog and WP2's immutable
authoritative state, seeded setup construction, card-conserving zone primitives, board/splay
geometry, value and icon queries, cloning, and deterministic state hashing. WP3 will add semantic
actions, player-safe observations, and the setup/turn decision protocol.

- Project contract: `PROJECT_GOAL.md`
- Supplied rules/data: `game-rules-plaintext/`
- Implementation sequence: `docs/MILESTONE_1_PLAN.md`
- WP2 state/geometry contract: `docs/WP2_STATE_CONTRACT.md`
- Rules interpretations: `docs/RULES_DECISIONS.md`
- Agent conventions: `AGENTS.md`

## Development environment

```bash
make install       # Core package and development tools
make install-ai    # Also install NumPy and CPU-only PyTorch for later milestones
make check         # Lint, type-check, and test
make run           # Verify the environment
```

Direct equivalents:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src tests
uv run innovation-ai doctor
```
