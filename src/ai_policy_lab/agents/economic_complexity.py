from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.research_tracks import (
    analyze_great_reallocation_exposure,
    is_great_reallocation_question,
)
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are an economic complexity researcher working with industry-space,
capability, and occupational transition methods."""


class EconomicComplexityAgent(BaseResearchAgent):
    name = "economic_complexity"
    phase = "phase_2_analysis"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if is_great_reallocation_question(state["root_question"]):
            result = analyze_great_reallocation_exposure(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            findings = [finding for finding in result.findings if finding["agent"] == self.name]
            if findings:
                return {"findings": findings}
            return {
                "flagged_issues": [
                    "Economic complexity network metrics are still pending; the first live analysis run did not yield a distinct capability-overlay finding."
                ],
            }

        _ = (state, runtime)
        return {
            "flagged_issues": [
                "Economic complexity metrics are not yet computed; capability mapping requires additional data engineering."
            ],
        }
