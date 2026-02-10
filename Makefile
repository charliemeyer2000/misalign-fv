.PHONY: lint typecheck test test-integration format all

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v -m integration

format:
	uv run ruff format src/ tests/

all: lint typecheck test
