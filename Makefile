.PHONY: install dev test test-live build up down logs clean

install:
	uv venv --python 3.12 .venv
	uv pip install -r requirements-dev.txt --python .venv/bin/python

dev:
	.venv/bin/uvicorn app.main:app --reload --port $${PORT:-8000}

test:
	.venv/bin/python -m pytest -q

build:
	./run.sh build

up:
	./run.sh up

down:
	./run.sh down

logs:
	./run.sh logs

clean:
	rm -rf data .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
