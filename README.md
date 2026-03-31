# AI Policy Research Lab

`ai-policy-lab` is the research engine behind `jobanxiety.ai`: a LangGraph-based, multi-agent policy research system for producing graduate-level reports, structured source audits, and normalized public dataset manifests.

The first target domain is AI labor market intelligence. The first flagship question is:

> How is AI adoption reshaping the occupational structure of the U.S. labor market, and what are the distributional consequences across geographies, education levels, and demographic groups?

## What this repo includes

- A typed shared state bus (`ResearchState`) modeled on the system spec
- A LangGraph DAG with discovery, analysis, audit, and synthesis phases
- Specialist agent modules with prompts derived from the design document
- Connector scaffolding for BLS, FRED, Census, O*NET, scholar search, and web search
- A report renderer that turns final state into a structured markdown report
- A CLI for running research jobs and writing outputs to `runs/`
- Tests for state initialization, graph flow, report rendering, and CLI behavior

## Current scope

This first commit is a serious scaffold, not a finished research product.

- The graph, agents, state schema, CLI, and output pipeline are implemented
- Connectors are present as real client stubs and extension points
- Mock mode is fully supported for local development and testing
- Live LLM mode is supported through any OpenAI-compatible endpoint, including Ollama
- Full live retrieval, structured source extraction, and empirical analysis still need to be filled in

## Quickstart

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
```

### 2. Configure environment

```bash
cp .env.example .env
```

By default, the project runs in mock mode so you can exercise the DAG without API keys.

### 3. Run a mock research job

```bash
poetry run ai-policy-lab run \
  --question "How is AI adoption reshaping occupational mobility in U.S. metros?" \
  --constraint "United States" \
  --constraint "MSA-level where possible" \
  --quality-floor tier_2
```

Outputs land in `runs/<timestamp>-<slug>/`.

## Using a local Ollama model

The CLI can talk to any OpenAI-compatible endpoint. To use the local model you just set up in Ollama:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export APL_DEFAULT_MODEL=qwopus-q4km
export APL_USE_MOCK=false
```

Then run:

```bash
poetry run ai-policy-lab run \
  --question "How is AI adoption reshaping occupational structure in the U.S. labor market?" \
  --constraint "United States" \
  --constraint "2015-2025" \
  --quality-floor tier_2
```

## Project layout

```text
src/ai_policy_lab/
  agents/         agent prompts and node implementations
  connectors/     external data client interfaces and stubs
  report/         final report renderer
  cli.py          Typer CLI
  config.py       environment-driven runtime settings
  graph.py        LangGraph DAG assembly
  llm.py          OpenAI-compatible LLM client
  state.py        typed state schema and helpers
tests/
  graph, report, and CLI smoke coverage
```

## Design notes

- Agents communicate through a shared typed state dictionary, not direct calls
- Append-only list fields use LangGraph reducers (`operator.add`) for auditability
- The quality gate is sequential by design: source audit, then methodology review, then synthesis
- The runtime is explicitly designed to support per-agent model overrides later

## Next milestones

1. Replace discovery stubs with live search + source extraction pipelines
2. Wire connector outputs into the Data Scout and Literature Review agents
3. Add structured source parsing and bibliography normalization
4. Implement real quantitative analysis and dataset normalization workflows
5. Add SSE/web delivery for `jobanxiety.ai`
