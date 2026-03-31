from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.research_tracks import (
    discover_great_reallocation_literature,
    discover_upskilling_pathways_literature,
    is_ai_labor_market_question,
    is_great_reallocation_question,
    is_upskilling_pathways_question,
)
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import wrap_user_content, wrap_user_list
from ai_policy_lab.state import ResearchState, SourceRecord

SYSTEM_PROMPT = """You are a graduate research assistant conducting a systematic literature review.
Identify what is known, debated, and unknown, prioritize Tier 1 and Tier 2 sources, and summarize by sub-question."""


class LiteratureReviewAgent(BaseResearchAgent):
    name = "literature_review"
    phase = "phase_1_discovery"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if is_upskilling_pathways_question(state["root_question"]):
            result = discover_upskilling_pathways_literature(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            issues = list(result.issues)
            issues.append(
                "NOTE: The literature review section is rendered directly from the retrieved source inventory to prevent unsupported citations."
            )
            return {
                "sources": result.sources,
                "existing_literature_summary": _compose_inventory_literature_review(
                    root_question=state["root_question"],
                    research_questions=[item["question"] for item in state["research_questions"]],
                    base_summary=result.summary,
                    sources=result.sources,
                ),
                "flagged_issues": issues,
            }

        if is_great_reallocation_question(state["root_question"]):
            result = discover_great_reallocation_literature(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            issues = list(result.issues)
            issues.append(
                "NOTE: The literature review section is rendered directly from the retrieved source inventory to prevent unsupported citations."
            )
            return {
                "sources": result.sources,
                "existing_literature_summary": _compose_inventory_literature_review(
                    root_question=state["root_question"],
                    research_questions=[item["question"] for item in state["research_questions"]],
                    base_summary=result.summary,
                    sources=result.sources,
                ),
                "flagged_issues": issues,
            }

        if is_ai_labor_market_question(state["root_question"], state["domain_constraints"]):
            result = discover_great_reallocation_literature(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            issues = list(result.issues)
            issues.append(
                "NOTE: Generic AI labor-market questions reuse the Great Reallocation literature spine until a broader live literature router is added."
            )
            issues.append(
                "NOTE: The literature review section is rendered directly from the retrieved source inventory to prevent unsupported citations."
            )
            return {
                "sources": result.sources,
                "existing_literature_summary": _compose_inventory_literature_review(
                    root_question=state["root_question"],
                    research_questions=[item["question"] for item in state["research_questions"]],
                    base_summary=result.summary,
                    sources=result.sources,
                ),
                "flagged_issues": issues,
            }

        prompt = (
            f"{wrap_user_content('root_question', state['root_question'])}\n"
            f"{wrap_user_list('sub_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
            f"<quality_floor>{state['quality_floor']}</quality_floor>\n"
            "Produce a literature review summary organized by sub-question."
        )
        fallback = (
            "Mock mode: literature review not executed against live academic sources yet. "
            "When discovery is fully wired, this node should search Google Scholar, NBER, Brookings, "
            "Federal Reserve research, and BLS Monthly Labor Review before synthesizing consensus, debate, and gaps."
        )
        return {
            "existing_literature_summary": runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                fallback=fallback,
            )
        }


def _compose_inventory_literature_review(
    *,
    root_question: str,
    research_questions: list[str],
    base_summary: str,
    sources: list[SourceRecord],
) -> str:
    question_lines = "".join(f"- {question}\n" for question in research_questions) or "- No sub-questions.\n"
    return (
        f"Structured literature review for: {root_question}\n\n"
        "Sub-questions in scope:\n"
        f"{question_lines}\n"
        f"{base_summary}\n\n"
        "Retrieved sources cited in this section:\n"
        f"{_render_source_inventory(sources)}"
    )


def _render_source_inventory(sources: list[SourceRecord]) -> str:
    if not sources:
        return "- No retrieved sources were available.\n"
    rendered = []
    for source in sources:
        rendered.append(
            f"- [{source['id']}] {source['title']} ({source['publication']}, {source['date'][:4]}) "
            f"[{source['source_tier']}]"
        )
    return "\n".join(rendered) + "\n"
