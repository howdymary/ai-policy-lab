from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.research_tracks import (
    discover_great_reallocation_literature,
    discover_upskilling_pathways_literature,
    is_great_reallocation_question,
    is_upskilling_pathways_question,
)
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import wrap_user_content, wrap_user_list
from ai_policy_lab.state import ResearchState

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
            summary = runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=(
                    f"{wrap_user_content('root_question', state['root_question'])}\n"
                    f"{wrap_user_list('sub_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
                    "Using the source inventory below, produce a literature review organized by "
                    "established consensus, active debates, and knowledge gaps:\n"
                    + wrap_user_content(
                        "source_inventory",
                        "\n".join(
                        f"- {source['title']} ({source['publication']}, {source['date'][:4]}, {source['source_tier']})"
                        for source in result.sources
                        ),
                    )
                ),
                fallback=result.summary,
            )
            return {
                "sources": result.sources,
                "existing_literature_summary": summary,
                "flagged_issues": result.issues,
            }

        if is_great_reallocation_question(state["root_question"]):
            result = discover_great_reallocation_literature(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            summary = runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=(
                    f"{wrap_user_content('root_question', state['root_question'])}\n"
                    f"{wrap_user_list('sub_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
                    "Using the source inventory below, produce a literature review organized by "
                    "established consensus, active debates, and knowledge gaps:\n"
                    + wrap_user_content(
                        "source_inventory",
                        "\n".join(
                        f"- {source['title']} ({source['publication']}, {source['date'][:4]}, {source['source_tier']})"
                        for source in result.sources
                        ),
                    )
                ),
                fallback=result.summary,
            )
            return {
                "sources": result.sources,
                "existing_literature_summary": summary,
                "flagged_issues": result.issues,
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
