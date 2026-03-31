from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from ai_policy_lab.sanitize import sanitize_user_input, sanitize_user_inputs
from ai_policy_lab.utils import utcnow_iso

SourceTier = Literal["tier_1", "tier_2", "tier_3", "excluded"]
SourceType = Literal[
    "peer_reviewed",
    "government_data",
    "think_tank",
    "news_primary",
    "news_secondary",
    "preprint",
    "industry_report",
    "blog",
    "social_media",
]
EvidenceStrength = Literal["strong", "moderate", "suggestive", "weak"]
AdversarialRecommendation = Literal["STANDS", "WEAKENED", "CHALLENGED"]
ResearchQuestionStatus = Literal["pending", "in_progress", "completed", "deferred"]
ResearchQuestionPriority = Literal["primary", "secondary", "exploratory"]
NormalizationStatus = Literal["raw", "cleaned", "normalized", "merged"]
QualityFloor = Literal["tier_1", "tier_2", "tier_3"]
RunStatus = Literal["running", "completed", "failed"]
PhaseName = Literal[
    "phase_0_intake",
    "phase_1_discovery",
    "phase_1_5_refinement",
    "phase_2_analysis",
    "phase_3_quality_gate",
    "phase_4_synthesis",
    "complete",
]


class MessageRecord(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class SourceRecord(TypedDict):
    """A single source with quality metadata."""

    id: str
    title: str
    authors: list[str]
    publication: str
    date: str
    url: str
    source_tier: SourceTier
    source_type: SourceType
    relevance_score: float
    conflict_of_interest: str | None
    notes: str


class DatasetRecord(TypedDict):
    """A discovered or created dataset."""

    id: str
    name: str
    source_agency: str
    url: str
    format: str
    temporal_coverage: str
    geographic_coverage: str
    key_variables: list[str]
    update_frequency: str
    access_method: str
    quality_notes: str
    normalization_status: NormalizationStatus


class Finding(TypedDict):
    """A discrete research finding from any analyst agent."""

    agent: str
    claim: str
    evidence_strength: EvidenceStrength
    supporting_sources: list[str]
    confidence: float
    methodology: str
    limitations: list[str]
    contradicts: list[str]


class ResearchQuestion(TypedDict):
    """A research question, either original or generated during the process."""

    id: str
    question: str
    parent_question: str | None
    status: ResearchQuestionStatus
    priority: ResearchQuestionPriority
    assigned_to: list[str]


class AdversarialReviewItem(TypedDict):
    """A structured counterargument emitted by the adversarial reviewer."""

    finding_claim: str
    counterargument: str
    evidence_strength: EvidenceStrength
    recommendation: AdversarialRecommendation
    supporting_sources: list[str]


class AgentLogEntry(TypedDict):
    timestamp: str
    agent: str
    phase: str
    summary: str


class ResearchState(TypedDict):
    """The complete shared state that flows through the research DAG."""

    root_question: str
    domain_constraints: list[str]
    quality_floor: QualityFloor
    sources: Annotated[list[SourceRecord], operator.add]
    datasets: Annotated[list[DatasetRecord], operator.add]
    research_questions: Annotated[list[ResearchQuestion], operator.add]
    existing_literature_summary: str
    data_availability_assessment: str
    policy_landscape_summary: str
    findings: Annotated[list[Finding], operator.add]
    quantitative_results: dict[str, object]
    methodology_description: str
    source_audit_report: str
    methodology_review: str
    adversarial_review: Annotated[list[AdversarialReviewItem], operator.add]
    flagged_issues: Annotated[list[str], operator.add]
    executive_summary: str
    full_report: str
    datasets_manifest: list[DatasetRecord]
    citation_list: list[SourceRecord]
    research_agenda: list[ResearchQuestion]
    messages: Annotated[list[MessageRecord], operator.add]
    current_phase: PhaseName
    agent_log: Annotated[list[AgentLogEntry], operator.add]
    run_status: RunStatus
    run_errors: Annotated[list[str], operator.add]


def make_initial_state(
    *,
    root_question: str,
    domain_constraints: list[str] | None = None,
    quality_floor: QualityFloor = "tier_2",
) -> ResearchState:
    sanitized_question = sanitize_user_input(root_question)
    sanitized_constraints = sanitize_user_inputs(domain_constraints or [])
    return {
        "root_question": sanitized_question,
        "domain_constraints": sanitized_constraints,
        "quality_floor": quality_floor,
        "sources": [],
        "datasets": [],
        "research_questions": [],
        "existing_literature_summary": "",
        "data_availability_assessment": "",
        "policy_landscape_summary": "",
        "findings": [],
        "quantitative_results": {},
        "methodology_description": "",
        "source_audit_report": "",
        "methodology_review": "",
        "adversarial_review": [],
        "flagged_issues": [],
        "executive_summary": "",
        "full_report": "",
        "datasets_manifest": [],
        "citation_list": [],
        "research_agenda": [],
        "messages": [{"role": "user", "content": sanitized_question}],
        "current_phase": "phase_0_intake",
        "run_status": "running",
        "run_errors": [],
        "agent_log": [
            {
                "timestamp": utcnow_iso(),
                "agent": "system",
                "phase": "phase_0_intake",
                "summary": "Initialized research state.",
            }
        ],
    }


def make_agent_log_entry(agent: str, phase: str, summary: str) -> AgentLogEntry:
    return {
        "timestamp": utcnow_iso(),
        "agent": agent,
        "phase": phase,
        "summary": summary,
    }
