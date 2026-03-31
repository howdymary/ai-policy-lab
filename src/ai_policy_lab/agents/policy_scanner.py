from __future__ import annotations

from typing import Any, Protocol

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.connectors import FederalRegisterConnector
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import wrap_user_content
from ai_policy_lab.state import ResearchState, SourceRecord

SYSTEM_PROMPT = """You are a policy analyst focused on legislation, regulation, institutional initiatives,
and policy precedents. Report institutional facts, not opinions."""


class FederalRegisterLike(Protocol):
    def search_documents(
        self,
        *,
        term: str,
        agency_slug: str | None = None,
        per_page: int = 5,
    ) -> dict[str, Any]: ...


class PolicyScannerAgent(BaseResearchAgent):
    name = "policy_scanner"
    phase = "phase_1_discovery"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if not runtime.settings.use_mock:
            summary, sources, issues = _run_live_policy_scan(
                root_question=state["root_question"],
                federal_register=FederalRegisterConnector(runtime.settings),
            )
            return {
                "policy_landscape_summary": runtime.maybe_generate(
                    agent_name=self.name,
                    system_prompt=self.system_prompt,
                    user_prompt=(
                        f"{wrap_user_content('root_question', state['root_question'])}\n"
                        "Summarize the active policy landscape, key institutions, and historical precedents "
                        "using these primary-source retrieval notes:\n"
                        f"{wrap_user_content('retrieval_notes', summary)}"
                    ),
                    fallback=summary,
                ),
                "sources": sources,
                "flagged_issues": issues,
            }

        prompt = (
            f"{wrap_user_content('root_question', state['root_question'])}\n"
            "Summarize the active policy landscape, key institutions, and historical precedents."
        )
        fallback = (
            "Policy scanner is not connected to congress.gov, Federal Register, or agency sources yet. "
            "A full production run should extract active legislation, agency initiatives, evaluation evidence, "
            "and relevant historical precedents from primary institutional sources."
        )
        return {
            "policy_landscape_summary": runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                fallback=fallback,
            )
        }


def _run_live_policy_scan(
    *,
    root_question: str,
    federal_register: FederalRegisterLike,
) -> tuple[str, list[SourceRecord], list[str]]:
    scans = [
        {
            "term": "artificial intelligence",
            "agency_slug": "labor-department",
            "label": "Labor Department AI/labor rulemaking",
        },
        {
            "term": "workforce",
            "agency_slug": "education-department",
            "label": "Education Department workforce and training actions",
        },
        {
            "term": "chips workforce",
            "agency_slug": "commerce-department",
            "label": "Commerce Department workforce-capacity actions",
        },
    ]
    notes = [
        "Primary-source policy scan from the Federal Register for labor, education, and industrial-capacity institutions relevant to AI-era workforce transitions."
    ]
    sources: list[SourceRecord] = []
    issues: list[str] = [
        "NOTE: Policy scanning now uses live Federal Register primary sources, but congress.gov, GAO, CRS, and program-evaluation ingestion are still not wired."
    ]

    for scan in scans:
        try:
            result = federal_register.search_documents(
                term=scan["term"],
                agency_slug=scan["agency_slug"],
                per_page=3,
            )
            documents = result.get("results", [])
            top_documents = documents[:2]
            if not top_documents:
                notes.append(f"- {scan['label']}: no recent Federal Register documents were returned.")
                continue

            rendered = []
            for document in top_documents:
                agency = (document.get("agencies") or [{}])[0].get("name", "Unknown agency")
                title = str(document.get("title", "Untitled document")).strip()
                publication_date = str(document.get("publication_date", ""))
                html_url = str(document.get("html_url", ""))
                doc_number = str(document.get("document_number", title))
                rendered.append(f"{title} ({agency}, {publication_date})")
                sources.append(
                    {
                        "id": f"fr-{doc_number.lower()}",
                        "title": title,
                        "authors": [agency],
                        "publication": "Federal Register",
                        "date": publication_date or "2026-03-30",
                        "url": html_url,
                        "source_tier": "tier_1",
                        "source_type": "government_data",
                        "relevance_score": 0.75,
                        "conflict_of_interest": None,
                        "notes": (
                            f"Live Federal Register policy scan for '{root_question}' via term "
                            f"'{scan['term']}' and agency '{scan['agency_slug']}'."
                        ),
                    }
                )
            notes.append(f"- {scan['label']}: " + "; ".join(rendered) + ".")
        except Exception as exc:  # noqa: BLE001
            issues.append(
                f"WARNING: Federal Register retrieval failed for {scan['agency_slug']} / {scan['term']}: {exc}"
            )

    notes.append(
        "- Institutional picture so far: Labor, Education, and Commerce are the clearest currently retrievable federal nodes for wage protections, training infrastructure, and industrial-capacity workforce programs connected to AI-era adjustment."
    )
    notes.append(
        "- Remaining gap: this scan does not yet cover live congressional bills, CBO/GAO evaluations, WIOA evidence reviews, or state-level policy databases."
    )
    return ("\n".join(notes), _dedupe_sources(sources), issues)


def _dedupe_sources(sources: list[SourceRecord]) -> list[SourceRecord]:
    deduped: dict[str, SourceRecord] = {}
    for source in sources:
        deduped[source["id"]] = source
    return list(deduped.values())
