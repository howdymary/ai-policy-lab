from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are a political economist analyzing institutions, distributional consequences,
and political feasibility. Distinguish descriptive claims from normative claims."""


class PoliticalEconomyAgent(BaseResearchAgent):
    name = "political_economy"
    phase = "phase_2_analysis"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        _ = (state, runtime)
        return {
            "flagged_issues": [
                "Political economy interpretation has not been grounded in live evidence yet; treat current output as a framing layer."
            ],
        }
