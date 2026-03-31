from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import wrap_user_content, wrap_user_list
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
            fallback = (
                "Methodology review completed. The current run documents a plausible design, but it is not yet a "
                "replicable empirical study because no live data pipeline or estimation output was executed."
            )
        elif analysis_type == "descriptive_index":
            fallback = (
                "Methodology review completed. This run executes a reproducible descriptive index using the available "
                "source state. The design is appropriate for comparative ranking and hypothesis generation, but "
                "conclusions should remain descriptive because the pipeline does not yet estimate causal effects, "
                "sampling uncertainty, or specification sensitivity."
            )
        else:
            fallback = (
                "Methodology review completed. A live quantitative path was executed, but its inferential scope "
                "should still be checked carefully against the claims made in the final report."
            )

        review = runtime.maybe_generate(
            agent_name=self.name,
            system_prompt=self.system_prompt,
            user_prompt=(
                f"{wrap_user_content('root_question', state['root_question'])}\n"
                f"{wrap_user_list('research_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
                f"{wrap_user_content('methodology_description', state['methodology_description'])}\n"
                f"<quantitative_status>{status}</quantitative_status>\n"
                f"<analysis_type>{analysis_type}</analysis_type>\n"
                f"{wrap_user_list('findings', [item['claim'] for item in state['findings']], item_tag='finding')}\n"
                "Assess research design appropriateness, statistical rigor, replicability, intellectual honesty, and claim scope."
            ),
            fallback=fallback,
            temperature=0.1,
        )
        if status != "completed":
            issues.append(
                "WARNING: Quantitative analysis has not yet produced effect sizes, confidence intervals, or robustness checks for this run."
            )
        elif analysis_type == "descriptive_index":
            issues.append(
                "NOTE: The current live analysis is reproducible and data-backed, but it remains descriptive; no causal identification, uncertainty intervals, or robustness sweeps were estimated yet."
            )
        return {
            "methodology_review": review,
            "flagged_issues": issues,
            "current_phase": "phase_3_quality_gate",
        }
