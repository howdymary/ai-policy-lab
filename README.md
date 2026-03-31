# AI Policy Research Lab

`ai-policy-lab` is the research engine behind `jobanxiety.ai`: a LangGraph-based, multi-agent policy research system for structured reports, source audits, and dataset manifests.

The current flagship focus is AI labor market intelligence. The main research question is:

> How is AI adoption reshaping the occupational structure of the U.S. labor market, and what are the distributional consequences across geographies, education levels, and demographic groups?

## What’s Included

- A typed shared state bus (`ResearchState`) with append-only reducers for auditability
- A LangGraph DAG with intake, discovery, midcourse refinement, analysis, quality gate, and synthesis phases
- Ten specialist agents, including adversarial review and the Research Director orchestrator
- Connectors for BLS, Census, FRED, O*NET, Federal Register, Crossref, Semantic Scholar, and a web-search placeholder
- A markdown report renderer and CLI for writing `report.md` and `state.json` artifacts to `runs/`
- Support for mock mode and live OpenAI-compatible LLM endpoints, including local Ollama
- Tests covering state initialization, graph flow, CLI behavior, config validation, sanitization, and core agent paths

## Current Scope

This repo is a working scaffold with real live discovery paths for the flagship labor-market track, not a finished public data product.

- The graph, CLI, report renderer, and state schema are implemented
- Live retrieval exists for several federal and academic sources
- Mock mode works for local development and test runs
- The `Great Reallocation` and `Upskilling Pathways` tracks are wired as specialized paths
- Some deeper public-release features, such as a persistent knowledge base and notebook generation, are still future work

## Quickstart

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
```

### 2. Configure Environment

```bash
cp .env.example .env
```

The project defaults to mock mode, so you can exercise the DAG without API keys.

## API Keys

The core government data APIs are free:

- BLS API: [data.bls.gov/registrationEngine](https://data.bls.gov/registrationEngine/)
- FRED API: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- Census API: [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)
- O*NET Web Services: [services.onetcenter.org/reference/register](https://services.onetcenter.org/reference/register)
- Semantic Scholar API: [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)

Optional local/OpenAI-compatible setup:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export APL_DEFAULT_MODEL=qwopus-q4km
export APL_USE_MOCK=false
```

## Run It

Run a mock job:

```bash
poetry run ai-policy-lab run \
  --question "How is AI adoption reshaping occupational mobility in U.S. metros?" \
  --constraint "United States" \
  --constraint "MSA-level where possible" \
  --quality-floor tier_2
```

Run the flagship Great Reallocation track:

```bash
poetry run ai-policy-lab run \
  --question "How is AI adoption reshaping the occupational structure of the U.S. labor market, and what are the distributional consequences across geographies, education levels, and demographic groups?" \
  --constraint "United States" \
  --constraint "2015-2025" \
  --constraint "MSA-level where possible" \
  --quality-floor tier_2
```

Outputs land in `runs/<timestamp>-<slug>/`.

## Testing

Run the full local check suite:

```bash
poetry run pytest -v
poetry run mypy src/
poetry run ruff check .
poetry run bandit -r src/ -q
```

To run coverage locally:

```bash
poetry run pytest --cov=ai_policy_lab --cov-fail-under=85
```

## Troubleshooting

- If you see `OPENAI_API_KEY must be set`, either enable mock mode with `APL_USE_MOCK=true` or set a non-empty key for your OpenAI-compatible endpoint.
- If FRED requests fail with a missing-key error, set `FRED_API_KEY`.
- If Semantic Scholar returns 429s, add `SEMANTIC_SCHOLAR_API_KEY`.
- If a run fails early, check `state.json` for `run_status` and `run_errors`.

## Docs

- [Architecture](./ARCHITECTURE.md)
- [Contributing](./CONTRIBUTING.md)

## Project Layout

```text
src/ai_policy_lab/
  agents/         agent prompts and node implementations
  connectors/     external data clients and shared HTTP utilities
  report/         final report renderer
  cli.py          Typer CLI
  config.py       environment-driven runtime settings
  graph.py        LangGraph DAG assembly
  llm.py          OpenAI-compatible LLM client
  state.py        typed state schema and helpers
tests/            CLI, graph, agent, config, runtime, and sanitization coverage
```

## Status

The repo is in active release-prep mode. The current codebase has passing tests and hardened connector/runtime behavior, and the remaining work is mostly about documentation polish and future feature depth.
