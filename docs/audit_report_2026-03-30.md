# AI Policy Research Lab — Audit Report
**Date:** March 30, 2026
**Auditor:** Codex (GPT-5)
**Spec Version:** 0.2.0

## Summary
- Total high-signal checks audited: 27
- Passed: 15
- Failed — BLOCKER: 9
- Failed — WARNING: 3
- Not Yet Implemented: 7 major subsystems or controls remain absent; these are called out below and overlap with some blocker items.

## Changes Landed During Audit
- Added an `AdversarialReviewerAgent` and inserted it into the Phase 3 quality gate after methodology review and before synthesis.
- Extended `ResearchState` and the markdown report to carry and render adversarial counterarguments.
- Tightened repo hygiene by updating `.gitignore`, expanding `.env.example`, and improving the README API-key setup section.

## Blocker Issues (Must Fix)
1. `pyproject.toml` is still below the required research stack. It lacks key dependencies from the spec, including `langchain-anthropic`, `langchain-openai`, `pandas`, `numpy`, `scipy`, `statsmodels`, `nbformat`, `nbconvert`, `pydantic`, and `pytest-asyncio`. Without these, the project cannot support the required notebook generation, statistical analysis, or provider matrix.
2. The connector layer does not implement the required `BaseConnector` contract from the spec. There is no `health_check()`, no unified `fetch()` returning `(data, ProvenanceRecord)`, no provenance hashing, no retry/backoff policy, and no rate limiter or cache abstraction in [base.py](/Users/maryliu/Projects/ai-policy-lab/src/ai_policy_lab/connectors/base.py).
3. FRED is not operational in this environment because `FRED_API_KEY` is unset. The current connector hard-fails with `ConnectorConfigurationError`, which means the MVP connector audit cannot pass end-to-end yet.
4. The human-in-the-loop Phase 1.5 checkpoint is not implemented. The graph runs straight from discovery into analysis without pausing for human approval or logging approval input.
5. The persistent knowledge base package does not exist. There is no SQLite store, no deduplication module, no schema/migration layer, and no cross-run dataset/source reuse.
6. The notebook-generation pipeline does not exist. There is no `nbformat`-based generator, no notebook templates, and no executable `.ipynb` output.
7. Runtime quality enforcement is still far below spec. There are no automated claim-citation checks, no provenance-chain validator, no effect-size/CI validator, no limitations-length gate, and no runtime pre-publication blocker system.
8. Provenance is incomplete across the system. Live data is fetched, but there is no auditable provenance record per fetch and no source-to-claim chain that a reviewer can verify in under 60 seconds.
9. The `congress.gov` connector and broader primary-source policy retrieval stack are absent, so the policy pipeline cannot yet meet the institutional-facts-only standard in the specification.

## Warning Issues (Should Fix)
1. `.env.example` is better than before but still does not document every model/provider path or registration workflow expected by the spec, especially around multi-provider LLM configuration and optional search backends.
2. `README.md` now includes key registration links, but it still lacks contribution guidelines, a full `0.2.0` architecture map, and an explicit explanation of the zero-hallucination policy and publication-quality audit bar.
3. The Semantic Scholar connector hit a live `429` without an API key during the audit. The connector works as a client stub, but the project has no built-in rate-limit strategy or authenticated fallback, so academic retrieval is still operationally fragile.

## Not Yet Implemented
- `src/ai_policy_lab/knowledge_base/`
- `src/ai_policy_lab/notebooks/`
- `src/ai_policy_lab/quality/`
- `src/ai_policy_lab/connectors/congress.py`
- `src/ai_policy_lab/graph/checkpoints.py` or equivalent pause/resume checkpoint logic
- `src/ai_policy_lab/llm/providers.py` and `src/ai_policy_lab/llm/config.py`
- Multi-format report outputs (`PDF`, `docx`) and committed docs such as `architecture.md`

## Connector Test Results
| Connector | Health Check | Fetch Test | Provenance | Rate Limiting | Overall |
|-----------|-------------|------------|------------|---------------|---------|
| BLS       | ❌ | ✅ | ❌ | ❌ | ❌ |
| FRED      | ❌ | ❌ | ❌ | ❌ | ❌ |
| Census    | ❌ | ✅ | ❌ | ❌ | ❌ |
| O*NET     | ❌ | ✅ | ❌ | ❌ | ❌ |
| Scholar   | ❌ | ❌ | ❌ | ❌ | ❌ |

Connector notes:
- BLS live endpoint test succeeded against `LNS14000000` and returned `REQUEST_SUCCEEDED` with a current value.
- Census live ACS request succeeded and returned 530 metro/micro observations for `C24010_001E`.
- O*NET live release detection succeeded and found public text release `db_30_2_text`.
- Semantic Scholar returned HTTP `429` unauthenticated in this environment.
- FRED could not be fetched because `FRED_API_KEY` is not configured.

## Quality Enforcement Test Results
| Test | Status | Notes |
|------|--------|-------|
| Every claim cited | ❌ | No automated claim extractor or citation gate exists. |
| No single-source findings | ❌ | No automated validator exists; some findings still rely on narrow source sets. |
| No hardcoded data | ⚠️ | No secrets were found, but there is no automated hardcoded-data scanner yet. |
| Effect sizes present | ❌ | Current live analysis is descriptive and does not emit effect sizes or confidence intervals. |
| Limitations substantive | ❌ | No automated word-count or substance gate exists. |
| Adversarial review addressed | ⚠️ | Adversarial review now exists, but there is no automated addressed-counterargument validator yet. |
| Provenance chain complete | ❌ | Source-to-fetch-to-claim provenance is not fully tracked. |
| Source tier distribution | ⚠️ | The live Great Reallocation run is manually acceptable, but there is no automated test yet. |

## Integration Test Results
- `pytest` passed locally: `9 passed`.
- `ruff` passed locally.
- The graph compiles and renders a Mermaid diagram with the expected Phase 3 sequence: `source_quality_auditor -> methodology_reviewer -> adversarial_reviewer -> research_director_synthesis`.
- Mock CLI run completed and wrote artifacts to `runs/audit-mock/`.
- The live Great Reallocation run completed and wrote artifacts to:
  - `/Users/maryliu/Projects/ai-policy-lab/runs/live-great-reallocation-analysis/report.md`
  - `/Users/maryliu/Projects/ai-policy-lab/runs/live-great-reallocation-analysis/state.json`
- The live report now contains a `Counterarguments and Rebuttals` section and the graph/state serialization still complete successfully after the adversarial-review change.
- The narrower end-to-end test from the specification (`SOC 15-1252`, 2019-2023, state variation, notebook reproduction) is not implemented yet and therefore cannot be marked passed.

## Security Audit Results
- ✅ No committed `.env` file, notebook, or SQLite knowledge-base database was found in tracked files.
- ✅ Grep did not surface live API secrets in source or tests.
- ✅ Remote data connectors use HTTPS; the only `http://` references are local Ollama endpoints on `localhost`.
- ✅ No `eval()` or `exec()` usage was found in the audited codebase.
- ⚠️ There is still no formal provenance or prompt-injection defense layer on retrieved web/search content.
- ⚠️ Cached raw-data policy and knowledge-base persistence controls are not implemented yet.

## Recommendations
1. Build the provenance-first connector layer next: `ProvenanceRecord`, retry/backoff, rate limiting, and cached raw responses should come before adding more analysis logic.
2. Implement the Phase 1.5 human checkpoint before deepening the research agents further. That pause is the highest-leverage quality intervention in the entire architecture.
3. Add the `knowledge_base` package and write every research run, source, dataset, and provenance record into SQLite so cross-run reuse and stale-data detection become possible.
4. Stand up the `notebooks` package only after provenance is in place; otherwise the notebook will reproduce outputs without reproducing trusted lineage.
5. Replace the current heuristic quality gate with executable tests for citations, provenance, source tiers, effect sizes, and counterargument handling.
6. Expand the provider layer into the spec’s `llm/config.py` and `llm/providers.py` shape so per-agent model selection is explicit rather than environment-variable-only.
7. Add a real `congress.gov` connector and primary-source policy retrieval before relying on policy outputs for anything beyond internal exploration.
