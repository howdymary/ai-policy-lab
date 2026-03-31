# Contributing to AI Policy Research Lab

Thanks for helping improve `ai-policy-lab`. The codebase is intentionally small and opinionated so it stays auditable.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
cp .env.example .env
```

## Local Checks

```bash
poetry run pytest -v
poetry run mypy src/
poetry run ruff check .
poetry run bandit -r src/ -q
```

## How To Add An Agent

1. Extend `BaseResearchAgent` in `src/ai_policy_lab/agents/`.
2. Keep the agent focused on one responsibility.
3. Make the `run()` method return a patch that merges cleanly into `ResearchState`.
4. Register the agent in `src/ai_policy_lab/agents/__init__.py` and `src/ai_policy_lab/graph.py`.
5. Add or update tests in `tests/`.

## How To Add A Connector

1. Extend `BaseConnector` in `src/ai_policy_lab/connectors/base.py`.
2. Use `_get_json()`, `_get_text()`, or `_post_json()` so rate limiting, retries, and caching are handled consistently.
3. Validate the response structure in the connector method before returning it.
4. Add tests for happy path and error handling.

## Style

- Keep code formatted with Ruff defaults and the repo line length of 100
- Prefer explicit state patches over hidden mutation
- Do not hardcode secrets or live data values
- Treat user input as untrusted and sanitize it before embedding in prompts

## Pull Requests

- Keep changes small and focused
- Include tests for new behavior
- Update docs when a change affects runtime behavior or setup
- Do not revert user work unless explicitly asked

## Release Readiness

Before opening a release PR, verify:

```bash
poetry run pytest -v
poetry run mypy src/
poetry run ruff check .
poetry run bandit -r src/ -q
```

If a change adds a new public CLI flag, connector, or agent behavior, please update `README.md` and `ARCHITECTURE.md` as part of the same change.
