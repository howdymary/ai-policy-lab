from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are an economic complexity researcher working with industry-space,
capability, and occupational transition methods."""


class EconomicComplexityAgent(BaseResearchAgent):
    name = "economic_complexity"
    phase = "phase_2_analysis"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        _ = (state, runtime)
        return {
            "flagged_issues": [
                "Economic complexity metrics are not yet computed; capability mapping requires additional data engineering."
            ],
        }
