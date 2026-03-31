from __future__ import annotations

import logging
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from ai_policy_lab.agents import (
    AdversarialReviewerAgent,
    DataScoutAgent,
    EconomicComplexityAgent,
    LiteratureReviewAgent,
    MethodologyReviewerAgent,
    PolicyScannerAgent,
    PoliticalEconomyAgent,
    QuantitativeAnalystAgent,
    ResearchDirectorAgent,
    SourceQualityAuditorAgent,
)
from ai_policy_lab.report import render_report
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import QualityFloor, ResearchState, make_initial_state

logger = logging.getLogger(__name__)


def build_graph(runtime: ResearchRuntime) -> Any:
    workflow = StateGraph(ResearchState)
    workflow_any = cast(Any, workflow)

    workflow_any.add_node("research_director_intake", ResearchDirectorAgent("intake").as_node(runtime))
    workflow_any.add_node("literature_review", LiteratureReviewAgent().as_node(runtime))
    workflow_any.add_node("data_scout", DataScoutAgent().as_node(runtime))
    workflow_any.add_node("policy_scanner", PolicyScannerAgent().as_node(runtime))
    workflow_any.add_node("research_director_midcourse", ResearchDirectorAgent("midcourse").as_node(runtime))
    workflow_any.add_node("quantitative_analyst", QuantitativeAnalystAgent().as_node(runtime))
    workflow_any.add_node("political_economy", PoliticalEconomyAgent().as_node(runtime))
    workflow_any.add_node("economic_complexity", EconomicComplexityAgent().as_node(runtime))
    workflow_any.add_node("source_quality_auditor", SourceQualityAuditorAgent().as_node(runtime))
    workflow_any.add_node("methodology_reviewer", MethodologyReviewerAgent().as_node(runtime))
    workflow_any.add_node("adversarial_reviewer", AdversarialReviewerAgent().as_node(runtime))
    workflow_any.add_node("research_director_synthesis", ResearchDirectorAgent("synthesis").as_node(runtime))

    workflow.add_edge(START, "research_director_intake")
    workflow.add_edge("research_director_intake", "literature_review")
    workflow.add_edge("research_director_intake", "data_scout")
    workflow.add_edge("research_director_intake", "policy_scanner")

    workflow.add_edge("literature_review", "research_director_midcourse")
    workflow.add_edge("data_scout", "research_director_midcourse")
    workflow.add_edge("policy_scanner", "research_director_midcourse")

    workflow.add_edge("research_director_midcourse", "quantitative_analyst")
    workflow.add_edge("research_director_midcourse", "political_economy")
    workflow.add_edge("research_director_midcourse", "economic_complexity")

    workflow.add_edge("quantitative_analyst", "source_quality_auditor")
    workflow.add_edge("political_economy", "source_quality_auditor")
    workflow.add_edge("economic_complexity", "source_quality_auditor")

    workflow.add_edge("source_quality_auditor", "methodology_reviewer")
    workflow.add_edge("methodology_reviewer", "adversarial_reviewer")
    workflow.add_edge("adversarial_reviewer", "research_director_synthesis")
    workflow.add_edge("research_director_synthesis", END)

    return workflow.compile()


def run_research(
    *,
    runtime: ResearchRuntime,
    root_question: str,
    domain_constraints: list[str] | None = None,
    quality_floor: QualityFloor = "tier_2",
) -> ResearchState:
    initial_state = make_initial_state(
        root_question=root_question,
        domain_constraints=domain_constraints,
        quality_floor=quality_floor,
        runtime_mode="mock" if runtime.settings.use_mock else "live",
    )
    graph = build_graph(runtime)
    last_state: dict[str, object] = dict(initial_state)
    try:
        for streamed_state in graph.stream(initial_state, stream_mode="values"):
            if isinstance(streamed_state, dict):
                last_state = dict(streamed_state)
        final_state = dict(last_state)
        final_state["run_status"] = "completed"
        final_state["run_errors"] = []
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research run failed before graph completion.")
        final_state = dict(last_state)
        final_state["run_status"] = "failed"
        final_state["run_errors"] = [*_string_list(final_state.get("run_errors")), str(exc)]
        final_state["flagged_issues"] = [
            *_string_list(final_state.get("flagged_issues")),
            f"BLOCKER: Research run failed before completion: {exc}",
        ]
        final_state["executive_summary"] = final_state.get(
            "executive_summary",
            "",
        ) or "This research run failed before the DAG completed. Review the run errors and partial outputs."
        final_state["full_report"] = final_state.get("full_report", "") or render_report(
            final_state  # type: ignore[arg-type]
        )
    return final_state  # type: ignore[return-value]


def _string_list(value: object) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []
