# Innovation AI

A deterministic Python implementation of the two-player Innovation base game, designed first for
reliable play between basic agents and later for self-play and PyTorch research.

## Current milestone

**Milestone 1:** complete deterministic rules engine for all 105 supplied cards. Every setup,
turn, and nested card-effect choice is exposed as a serializable first-class decision, with strict
separation between authoritative state and player-visible observations.

The engine includes the validated catalog/state/protocol layers, resumable effects and public
Dogma integration, achievement/terminal handling, all **105/105** card programs and all **158/158**
printed effects, basic agents and pull runners, versioned serialization/log/replay, reusable
invariants, and deterministic full-Dogma fuzzing. Missing registrations fail loudly rather than
acting as no-ops.

- Project contract: `PROJECT_GOAL.md`
- Supplied rules/data: `game-rules-plaintext/`
- Implementation sequence: `docs/MILESTONE_1_PLAN.md`
- WP2 state/geometry contract: `docs/WP2_STATE_CONTRACT.md`
- WP3 protocol contract: `docs/WP3_PROTOCOL_CONTRACT.md`
- WP5 Freeze-B/effect contract: `docs/WP5_FREEZE_B_CONTRACT.md`
- WP6 achievement/terminal contract: `docs/WP6_ACHIEVEMENT_CONTRACT.md`
- WP9 serialization/replay contract: `docs/WP9_REPLAY_CONTRACT.md`
- Rules interpretations: `docs/RULES_DECISIONS.md`
- Future roadmap: `docs/ROADMAP.md`
- Agent conventions: `AGENTS.md`

## Development environment

```bash
make install       # Core package and development tools
make install-ai    # Also install NumPy and CPU-only PyTorch for later milestones
make check         # Lint, type-check, and full default test/coverage gate
INNOVATION_LARGE_FUZZ_SEEDS=100 uv run pytest -q -m fuzz tests/innovation/test_fuzz.py
make run           # Verify the environment
```

Direct equivalents:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src tests
uv run innovation-ai doctor
uv run innovation-ai play --seed 0 --log artifacts/game.json
uv run innovation-ai replay artifacts/game.json
```
