from __future__ import annotations

import json
from typing import cast

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import wrap_user_content, wrap_user_list
from ai_policy_lab.state import (
    AdversarialRecommendation,
    AdversarialReviewItem,
    EvidenceStrength,
    Finding,
    ResearchState,
)

SYSTEM_PROMPT = """You are the Devil's Advocate of this research lab.
For each finding, identify the strongest counterargument, rate the evidence strength of that
counterargument, and recommend STANDS, WEAKENED, or CHALLENGED."""


class AdversarialReviewerAgent(BaseResearchAgent):
    name = "adversarial_reviewer"
    phase = "phase_3_quality_gate"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if not state["findings"]:
            return {
                "flagged_issues": [
                    "WARNING: Adversarial review could not run because no findings were available to challenge."
                ],
                "current_phase": "phase_3_quality_gate",
            }

        counter_sources = _counterargument_sources(state)
        analysis_type = str(state["quantitative_results"].get("analysis_type", ""))
        review_items = [
            _build_review_item_with_llm(
                finding=finding,
                analysis_type=analysis_type,
                counter_sources=counter_sources,
                runtime=runtime,
                system_prompt=self.system_prompt,
                agent_name=self.name,
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
    finding: Finding,
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
        "supporting_sources": counter_sources or list(finding["supporting_sources"]),
    }


def _build_review_item_with_llm(
    *,
    finding: Finding,
    analysis_type: str,
    counter_sources: list[str],
    runtime: ResearchRuntime,
    system_prompt: str,
    agent_name: str,
) -> AdversarialReviewItem:
    heuristic = _build_review_item(
        finding=finding,
        analysis_type=analysis_type,
        counter_sources=counter_sources,
    )
    fallback = json.dumps(heuristic)
    response = runtime.maybe_generate(
        agent_name=agent_name,
        system_prompt=system_prompt,
        user_prompt=(
            "Return a JSON object with keys "
            '"finding_claim", "counterargument", "evidence_strength", "recommendation", '
            '"supporting_sources".\n'
            f"{wrap_user_content('finding_claim', str(finding['claim']))}\n"
            f"<finding_evidence_strength>{finding['evidence_strength']}</finding_evidence_strength>\n"
            f"{wrap_user_content('finding_methodology', str(finding['methodology']))}\n"
            f"{wrap_user_list('finding_limitations', [str(item) for item in finding['limitations']], item_tag='limitation')}\n"
            f"<analysis_type>{analysis_type}</analysis_type>\n"
            f"{wrap_user_list('candidate_counter_sources', counter_sources, item_tag='source_id')}"
        ),
        fallback=fallback,
        temperature=0.1,
    )
    return _parse_review_item(response, heuristic)


def _parse_review_item(response: str, fallback: AdversarialReviewItem) -> AdversarialReviewItem:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(parsed, dict):
        return fallback

    recommendation = str(parsed.get("recommendation", fallback["recommendation"])).upper()
    if recommendation not in {"STANDS", "WEAKENED", "CHALLENGED"}:
        recommendation = fallback["recommendation"]

    evidence_strength = str(parsed.get("evidence_strength", fallback["evidence_strength"])).lower()
    if evidence_strength not in {"strong", "moderate", "suggestive", "weak"}:
        evidence_strength = fallback["evidence_strength"]

    supporting_sources = parsed.get("supporting_sources", fallback["supporting_sources"])
    if not isinstance(supporting_sources, list) or not all(
        isinstance(source_id, str) for source_id in supporting_sources
    ):
        supporting_sources = fallback["supporting_sources"]

    counterargument = parsed.get("counterargument")
    if not isinstance(counterargument, str) or not counterargument.strip():
        counterargument = fallback["counterargument"]

    finding_claim = parsed.get("finding_claim")
    if not isinstance(finding_claim, str) or not finding_claim.strip():
        finding_claim = fallback["finding_claim"]

    return {
        "finding_claim": finding_claim.strip(),
        "counterargument": counterargument.strip(),
        "evidence_strength": cast(EvidenceStrength, evidence_strength),
        "recommendation": cast(AdversarialRecommendation, recommendation),
        "supporting_sources": supporting_sources,
    }
