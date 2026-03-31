from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are an academic methods reviewer. Check research design, statistical rigor,
replicability, intellectual honesty, and claim scope."""


class MethodologyReviewerAgent(BaseResearchAgent):
    name = "methodology_reviewer"
    phase = "phase_3_quality_gate"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        status = state["quantitative_results"].get("status", "unknown")
        issues = []
        if status != "completed":
            issues.append(
                "WARNING: Quantitative analysis is still scaffold-level; no effect sizes, confidence intervals, or robustness checks were produced."
            )
        return {
            "methodology_review": (
                "Methodology review completed. The current run documents a plausible design, but it is not yet a "
                "replicable empirical study because no live data pipeline or estimation output was executed."
            ),
            "flagged_issues": issues,
            "current_phase": "phase_3_quality_gate",
        }
