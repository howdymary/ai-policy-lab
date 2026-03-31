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
        analysis_type = state["quantitative_results"].get("analysis_type", "unknown")
        issues = []
        if status != "completed":
            issues.append(
                "WARNING: Quantitative analysis is still scaffold-level; no effect sizes, confidence intervals, or robustness checks were produced."
            )
            review = (
                "Methodology review completed. The current run documents a plausible design, but it is not yet a "
                "replicable empirical study because no live data pipeline or estimation output was executed."
            )
        elif analysis_type == "descriptive_index":
            issues.append(
                "NOTE: The current live analysis is reproducible and data-backed, but it remains descriptive; no causal identification, uncertainty intervals, or robustness sweeps were estimated yet."
            )
            review = (
                "Methodology review completed. This run executes a reproducible descriptive metro exposure index "
                "using O*NET, ACS, and CBP. The design is appropriate for comparative ranking and hypothesis "
                "generation, but conclusions should remain descriptive because the pipeline does not yet estimate "
                "causal effects, sampling uncertainty, or specification sensitivity."
            )
        else:
            review = (
                "Methodology review completed. A live quantitative path was executed, but its inferential scope "
                "should still be checked carefully against the claims made in the final report."
            )
        return {
            "methodology_review": review,
            "flagged_issues": issues,
            "current_phase": "phase_3_quality_gate",
        }
