install:
	uv sync

run:
	uv run uvicorn apps.api.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run black .

typecheck:
	uv run mypy .

fix:
	uv run ruff check . --fix