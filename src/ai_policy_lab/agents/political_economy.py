from __future__ import annotations

import json
from typing import cast

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import wrap_user_content, wrap_user_list
from ai_policy_lab.state import EvidenceStrength, Finding, ResearchState

SYSTEM_PROMPT = """You are a political economist analyzing institutions, distributional consequences,
and political feasibility. Distinguish descriptive claims from normative claims."""


class PoliticalEconomyAgent(BaseResearchAgent):
    name = "political_economy"
    phase = "phase_2_analysis"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        root_question = state["root_question"]
        supporting_sources = _supporting_sources(state)
        fallback_finding: Finding = {
            "agent": self.name,
            "claim": (
                f"The current evidence base suggests that the costs and benefits tied to {root_question} "
                "are unlikely to be evenly distributed: actors with less institutional flexibility, weaker bargaining "
                "power, or thinner local support systems are more likely to absorb transition costs first."
            ),
            "evidence_strength": "suggestive",
            "supporting_sources": supporting_sources,
            "confidence": 0.48,
            "methodology": (
                "Political-economy synthesis of the discovery summaries, dataset inventory, policy scan, and any "
                "existing quantitative findings captured in the shared state."
            ),
            "limitations": [
                "Generic-path interpretation is based on the current state summaries, not subgroup microdata or causal identification.",
                "Institutional mechanisms still need to be validated against program evaluation or administrative evidence.",
            ],
            "contradicts": [],
        }

        response = runtime.maybe_generate(
            agent_name=self.name,
            system_prompt=self.system_prompt,
            user_prompt=(
                "Return a JSON object with keys "
                '"claim", "evidence_strength", "confidence", "methodology", "limitations".\n'
                f"{wrap_user_content('root_question', root_question)}\n"
                f"{wrap_user_list('constraints', state['domain_constraints'], item_tag='constraint')}\n"
                f"{wrap_user_list('research_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
                f"{wrap_user_content('literature_summary', state['existing_literature_summary'])}\n"
                f"{wrap_user_content('data_summary', state['data_availability_assessment'])}\n"
                f"{wrap_user_content('policy_summary', state['policy_landscape_summary'])}\n"
                f"{wrap_user_list('quantitative_findings', [item['claim'] for item in state['findings']], item_tag='finding')}\n"
                "Produce one careful political-economy finding that identifies who likely bears costs, which institutions matter most, "
                "and how confident the current evidence justifies being. Keep the claim aligned to the root question topic."
            ),
            fallback=json.dumps({key: value for key, value in fallback_finding.items() if key != "agent"}),
            temperature=0.1,
        )

        finding = _parse_finding_response(response, fallback_finding)
        return {
            "findings": [finding],
            "flagged_issues": [
                "NOTE: Generic political-economy analysis is now state-aware, but it still needs subgroup or program-evaluation evidence before it can support strong causal claims."
            ],
        }


def _supporting_sources(state: ResearchState, *, limit: int = 4) -> list[str]:
    return [source["id"] for source in state["sources"][:limit]]


def _parse_finding_response(response: str, fallback: Finding) -> Finding:
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

    claim = parsed.get("claim")
    methodology = parsed.get("methodology")
    limitations = parsed.get("limitations")
    evidence_strength = str(parsed.get("evidence_strength", fallback["evidence_strength"])).lower()
    confidence = parsed.get("confidence", fallback["confidence"])

    if not isinstance(claim, str) or not claim.strip():
        claim = fallback["claim"]
    if not isinstance(methodology, str) or not methodology.strip():
        methodology = fallback["methodology"]
    if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
        limitations = fallback["limitations"]
    if evidence_strength not in {"strong", "moderate", "suggestive", "weak"}:
        evidence_strength = fallback["evidence_strength"]
    if not isinstance(confidence, (int, float)):
        confidence = fallback["confidence"]

    return {
        "agent": fallback["agent"],
        "claim": claim.strip(),
        "evidence_strength": cast(EvidenceStrength, evidence_strength),
        "supporting_sources": fallback["supporting_sources"],
        "confidence": max(0.0, min(float(confidence), 1.0)),
        "methodology": methodology.strip(),
        "limitations": limitations,
        "contradicts": fallback["contradicts"],
    }
