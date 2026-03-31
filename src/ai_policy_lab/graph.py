from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai_policy_lab.agents import (
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
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import QualityFloor, ResearchState, make_initial_state


def build_graph(runtime: ResearchRuntime):
    workflow = StateGraph(ResearchState)

    workflow.add_node("research_director_intake", ResearchDirectorAgent("intake").as_node(runtime))
    workflow.add_node("literature_review", LiteratureReviewAgent().as_node(runtime))
    workflow.add_node("data_scout", DataScoutAgent().as_node(runtime))
    workflow.add_node("policy_scanner", PolicyScannerAgent().as_node(runtime))
    workflow.add_node("research_director_midcourse", ResearchDirectorAgent("midcourse").as_node(runtime))
    workflow.add_node("quantitative_analyst", QuantitativeAnalystAgent().as_node(runtime))
    workflow.add_node("political_economy", PoliticalEconomyAgent().as_node(runtime))
    workflow.add_node("economic_complexity", EconomicComplexityAgent().as_node(runtime))
    workflow.add_node("source_quality_auditor", SourceQualityAuditorAgent().as_node(runtime))
    workflow.add_node("methodology_reviewer", MethodologyReviewerAgent().as_node(runtime))
    workflow.add_node("research_director_synthesis", ResearchDirectorAgent("synthesis").as_node(runtime))

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
    workflow.add_edge("methodology_reviewer", "research_director_synthesis")
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
    )
    graph = build_graph(runtime)
    final_state = graph.invoke(initial_state)
    return final_state
