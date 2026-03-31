from __future__ import annotations

import json
from typing import cast

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.research_tracks import (
    analyze_great_reallocation_exposure,
    is_great_reallocation_question,
    is_upskilling_pathways_question,
)
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import wrap_user_content, wrap_user_list
from ai_policy_lab.state import EvidenceStrength, Finding, ResearchState

SYSTEM_PROMPT = """You are an economic complexity researcher working with industry-space,
capability, and occupational transition methods."""


class EconomicComplexityAgent(BaseResearchAgent):
    name = "economic_complexity"
    phase = "phase_2_analysis"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if is_great_reallocation_question(state["root_question"]) or is_upskilling_pathways_question(
            state["root_question"]
        ):
            result = analyze_great_reallocation_exposure(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            findings = [finding for finding in result.findings if finding["agent"] == self.name]
            if findings:
                return {"findings": findings}
            return {
                "flagged_issues": [
                    "Economic complexity network metrics are still pending; the first live analysis run did not yield a distinct capability-overlay finding."
                ],
            }

        supporting_sources = [source["id"] for source in state["sources"][:4]]
        regional_datasets = [
            dataset["name"]
            for dataset in state["datasets"]
            if "state" in dataset["geographic_coverage"].lower()
            or "msa" in dataset["geographic_coverage"].lower()
            or "county" in dataset["geographic_coverage"].lower()
        ]
        fallback_finding: Finding = {
            "agent": self.name,
            "claim": (
                "The current question is likely to hinge on place-specific capability differences rather than national averages alone, "
                "which means the most valuable next evidence is regional data that can distinguish locations with adjacent capabilities "
                "from those facing harder infrastructure, institutional, or industry-composition gaps."
            ),
            "evidence_strength": "suggestive",
            "supporting_sources": supporting_sources,
            "confidence": 0.44,
            "methodology": (
                "Generic economic-complexity synthesis using the dataset inventory, research questions, and any existing findings to infer "
                "what place-based evidence would matter most for diversification or transition analysis."
            ),
            "limitations": [
                "No RCA, proximity, ECI, or transition-network metrics were computed in the generic path.",
                "If the dataset inventory lacks regional production data, this memo is directional rather than measurement-based.",
            ],
            "contradicts": [],
        }

        response = runtime.maybe_generate(
            agent_name=self.name,
            system_prompt=self.system_prompt,
            user_prompt=(
                "Return a JSON object with keys "
                '"claim", "evidence_strength", "confidence", "methodology", "limitations".\n'
                f"{wrap_user_content('root_question', state['root_question'])}\n"
                f"{wrap_user_list('constraints', state['domain_constraints'], item_tag='constraint')}\n"
                f"{wrap_user_list('research_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
                f"{wrap_user_list('regional_datasets', regional_datasets, item_tag='dataset')}\n"
                f"{wrap_user_list('all_datasets', [item['name'] for item in state['datasets']], item_tag='dataset')}\n"
                f"{wrap_user_list('quantitative_findings', [item['claim'] for item in state['findings']], item_tag='finding')}\n"
                "Produce one careful economic-complexity finding about capability gaps, adjacency, and the level of geography needed to answer the question well."
            ),
            fallback=json.dumps({key: value for key, value in fallback_finding.items() if key != "agent"}),
            temperature=0.1,
        )
        finding = _parse_finding_response(response, fallback_finding)
        issue = (
            "NOTE: Generic economic-complexity analysis now emits a state-aware capability memo, but full network metrics still require explicit place-industry data engineering."
        )
        if not regional_datasets:
            issue = (
                "WARNING: Generic economic-complexity analysis ran without regional datasets in state, so the capability memo is constrained by missing place-based inputs."
            )
        return {"findings": [finding], "flagged_issues": [issue]}


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
