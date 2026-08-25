.PHONY: install install-ai run test lint format typecheck check clean

install:
	uv sync

install-ai:
	uv sync --extra ai

run:
	uv run card-game-ai doctor

test:
	uv run pytest --cov=card_game_ai --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src tests

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
