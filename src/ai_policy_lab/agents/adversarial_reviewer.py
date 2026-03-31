from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import AdversarialReviewItem, ResearchState

SYSTEM_PROMPT = """You are the Devil's Advocate of this research lab.
For each finding, identify the strongest counterargument, rate the evidence strength of that
counterargument, and recommend STANDS, WEAKENED, or CHALLENGED."""


class AdversarialReviewerAgent(BaseResearchAgent):
    name = "adversarial_reviewer"
    phase = "phase_3_quality_gate"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        _ = runtime
        if not state["findings"]:
            return {
                "flagged_issues": [
                    "WARNING: Adversarial review could not run because no findings were available to challenge."
                ],
                "current_phase": "phase_3_quality_gate",
            }

        counter_sources = _counterargument_sources(state)
        review_items = [
            _build_review_item(
                finding=finding,
                analysis_type=str(state["quantitative_results"].get("analysis_type", "")),
                counter_sources=counter_sources,
            )
            for finding in state["findings"]
        ]
        return {
            "adversarial_review": review_items,
            "flagged_issues": [
                "NOTE: Adversarial review is heuristic until a dedicated literature-plus-policy rebuttal retrieval path is wired."
            ],
            "current_phase": "phase_3_quality_gate",
        }


def _counterargument_sources(state: ResearchState) -> list[str]:
    preferred = [
        "doi-10.1257/jep.33.2.3",
        "doi-10.1086/705716",
        "doi-10.1016/j.techfore.2016.08.019",
        "src-webb-ai-labor-market",
    ]
    available = {source["id"] for source in state["sources"]}
    selected = [source_id for source_id in preferred if source_id in available]
    if selected:
        return selected
    return [source["id"] for source in state["sources"][:3]]


def _build_review_item(
    *,
    finding: dict[str, object],
    analysis_type: str,
    counter_sources: list[str],
) -> AdversarialReviewItem:
    claim = str(finding["claim"])
    evidence_strength = str(finding["evidence_strength"])

    if analysis_type == "descriptive_index":
        return {
            "finding_claim": claim,
            "counterargument": (
                "This result is a descriptive exposure ranking, not direct evidence of realized AI adoption, "
                "worker displacement, or wage effects. A metro can rank high because of occupational mix while "
                "still experiencing complementarity or net job growth rather than disruption."
            ),
            "evidence_strength": "moderate",
            "recommendation": "WEAKENED",
            "supporting_sources": counter_sources,
        }

    if evidence_strength == "suggestive":
        return {
            "finding_claim": claim,
            "counterargument": (
                "The current evidence is suggestive enough that a reasonable critic could argue the same pattern "
                "might disappear under a different sample window, variable construction, or geography definition."
            ),
            "evidence_strength": "moderate",
            "recommendation": "CHALLENGED",
            "supporting_sources": counter_sources,
        }

    return {
        "finding_claim": claim,
        "counterargument": (
            "The finding is directionally plausible, but the current implementation still depends on untested "
            "assumptions about measurement, sample construction, and model scope that should be surfaced more explicitly."
        ),
        "evidence_strength": "suggestive",
        "recommendation": "WEAKENED",
        "supporting_sources": counter_sources or list(finding.get("supporting_sources", [])),
    }
