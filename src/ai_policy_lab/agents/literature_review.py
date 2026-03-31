from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are a graduate research assistant conducting a systematic literature review.
Identify what is known, debated, and unknown, prioritize Tier 1 and Tier 2 sources, and summarize by sub-question."""


class LiteratureReviewAgent(BaseResearchAgent):
    name = "literature_review"
    phase = "phase_1_discovery"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        prompt = (
            f"Root question: {state['root_question']}\n"
            f"Sub-questions: {[item['question'] for item in state['research_questions']]}\n"
            f"Quality floor: {state['quality_floor']}\n"
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
