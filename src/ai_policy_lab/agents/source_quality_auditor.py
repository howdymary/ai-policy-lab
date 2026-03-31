from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are the quality gate for citations, source tiers, provenance, recency,
and conflicts of interest. Flag blockers, warnings, and notes."""


class SourceQualityAuditorAgent(BaseResearchAgent):
    name = "source_quality_auditor"
    phase = "phase_3_quality_gate"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        source_count = len(state["sources"])
        report = (
            f"Source audit completed on {source_count} captured sources. "
            "In this scaffold build, the main blocker is that live literature and policy retrieval "
            "have not yet populated a real bibliography."
        )
        issues = []
        if source_count == 0:
            issues.append(
                "BLOCKER: No live sources were ingested, so claims cannot yet meet the Tier 1/Tier 2 citation standard."
            )
        return {
            "source_audit_report": report,
            "flagged_issues": issues,
            "current_phase": "phase_3_quality_gate",
        }
