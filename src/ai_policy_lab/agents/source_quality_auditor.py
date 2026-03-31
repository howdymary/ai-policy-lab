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
        issues = []
        tier_1_count = sum(1 for source in state["sources"] if source["source_tier"] == "tier_1")
        tier_2_count = sum(1 for source in state["sources"] if source["source_tier"] == "tier_2")
        report = (
            f"Source audit completed on {source_count} captured sources, including "
            f"{tier_1_count} Tier 1 and {tier_2_count} Tier 2 records. "
            "The bibliography is now populated for the Great Reallocation path, but policy-source ingestion "
            "and source-to-claim validation still need to be completed before publication."
        )
        if source_count == 0:
            issues.append(
                "BLOCKER: No live sources were ingested, so claims cannot yet meet the Tier 1/Tier 2 citation standard."
            )
        elif "policy scanner is not connected" in state["policy_landscape_summary"].lower():
            issues.append(
                "WARNING: Policy discovery is still stubbed, so legislative and regulatory claims are not yet covered by live primary sources."
            )
        return {
            "source_audit_report": report,
            "flagged_issues": issues,
            "current_phase": "phase_3_quality_gate",
        }
