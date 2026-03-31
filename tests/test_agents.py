from __future__ import annotations

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
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import make_initial_state


def _runtime(monkeypatch) -> ResearchRuntime:
    monkeypatch.setenv("APL_USE_MOCK", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    return ResearchRuntime.from_env()


def _generic_state():
    state = make_initial_state(
        root_question="How are local permitting reforms affecting housing supply?",
        domain_constraints=["United States"],
        quality_floor="tier_2",
    )
    state["research_questions"] = [
        {
            "id": "rq-1",
            "question": "How do permitting timelines affect project delivery?",
            "parent_question": None,
            "status": "pending",
            "priority": "primary",
            "assigned_to": ["literature_review"],
        }
    ]
    return state


def test_research_director_generic_intake(monkeypatch) -> None:
    runtime = _runtime(monkeypatch)
    state = make_initial_state(
        root_question="How are local permitting reforms affecting housing supply?",
        domain_constraints=["United States"],
        quality_floor="tier_2",
    )

    patch = ResearchDirectorAgent("intake").run(state, runtime)

    assert patch["current_phase"] == "phase_1_discovery"
    assert len(patch["research_questions"]) >= 5
    assert all("??" not in item["question"] for item in patch["research_questions"])


def test_research_director_specialized_intake(monkeypatch) -> None:
    runtime = _runtime(monkeypatch)
    state = make_initial_state(
        root_question="How is AI adoption reshaping the occupational structure of the U.S. labor market?",
        domain_constraints=["United States"],
        quality_floor="tier_2",
    )

    patch = ResearchDirectorAgent("intake").run(state, runtime)

    assert patch["current_phase"] == "phase_1_discovery"
    assert len(patch["research_questions"]) == 6


def test_generic_agents_return_expected_shapes(monkeypatch) -> None:
    runtime = _runtime(monkeypatch)
    state = _generic_state()
    state["findings"] = [
        {
            "agent": "quantitative_analyst",
            "claim": "Permitting delays are uneven across metros.",
            "evidence_strength": "suggestive",
            "supporting_sources": ["src-1"],
            "confidence": 0.5,
            "methodology": "Descriptive comparison",
            "limitations": ["No causal identification"],
            "contradicts": [],
        }
    ]
    state["sources"] = [
        {
            "id": "src-1",
            "title": "Example source",
            "authors": ["Example Agency"],
            "publication": "Example Publication",
            "date": "2025-01-01",
            "url": "https://example.com",
            "source_tier": "tier_2",
            "source_type": "think_tank",
            "relevance_score": 0.7,
            "conflict_of_interest": None,
            "notes": "",
        }
    ]

    patches = {
        "literature": LiteratureReviewAgent().run(state, runtime),
        "data": DataScoutAgent().run(state, runtime),
        "policy": PolicyScannerAgent().run(state, runtime),
        "quant": QuantitativeAnalystAgent().run(state, runtime),
        "political": PoliticalEconomyAgent().run(state, runtime),
        "complexity": EconomicComplexityAgent().run(state, runtime),
        "audit": SourceQualityAuditorAgent().run(state, runtime),
        "methods": MethodologyReviewerAgent().run(state, runtime),
        "adversarial": AdversarialReviewerAgent().run(state, runtime),
    }

    assert isinstance(patches["literature"]["existing_literature_summary"], str)
    assert isinstance(patches["data"]["datasets"], list)
    assert patches["data"]["datasets"] == []
    assert isinstance(patches["policy"]["policy_landscape_summary"], str)
    assert isinstance(patches["quant"]["quantitative_results"], dict)
    assert isinstance(patches["political"]["flagged_issues"], list)
    assert isinstance(patches["complexity"]["flagged_issues"], list)
    assert isinstance(patches["audit"]["source_audit_report"], str)
    assert isinstance(patches["methods"]["methodology_review"], str)
    assert isinstance(patches["adversarial"]["adversarial_review"], list)


def test_specialized_agents_return_expected_shapes(monkeypatch) -> None:
    runtime = _runtime(monkeypatch)
    state = make_initial_state(
        root_question="How is AI adoption reshaping the occupational structure of the U.S. labor market?",
        domain_constraints=["United States", "2018-2025"],
        quality_floor="tier_2",
    )
    state["research_questions"] = ResearchDirectorAgent("intake").run(state, runtime)["research_questions"]

    literature_patch = LiteratureReviewAgent().run(state, runtime)
    data_patch = DataScoutAgent().run(state, runtime)
    quant_patch = QuantitativeAnalystAgent().run(state, runtime)
    complexity_patch = EconomicComplexityAgent().run(state, runtime)

    assert literature_patch["sources"]
    assert data_patch["datasets"]
    assert isinstance(quant_patch["quantitative_results"], dict)
    assert "status" in quant_patch["quantitative_results"]
    assert "findings" in complexity_patch or "flagged_issues" in complexity_patch
