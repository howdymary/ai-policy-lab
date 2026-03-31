from ai_policy_lab.agents.adversarial_reviewer import AdversarialReviewerAgent
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import make_initial_state


def test_adversarial_reviewer_produces_items_for_existing_findings(monkeypatch) -> None:
    monkeypatch.setenv("APL_USE_MOCK", "true")
    runtime = ResearchRuntime.from_env()
    state = make_initial_state(root_question="How is AI affecting metro labor markets?")
    state["findings"] = [
        {
            "agent": "quantitative_analyst",
            "claim": "A descriptive index ranks Metro A above Metro B.",
            "evidence_strength": "moderate",
            "supporting_sources": ["src-census-acs"],
            "confidence": 0.7,
            "methodology": "Descriptive occupational-composition index.",
            "limitations": ["Not causal."],
            "contradicts": [],
        }
    ]
    state["quantitative_results"] = {"analysis_type": "descriptive_index", "status": "completed"}
    state["sources"] = [
        {
            "id": "doi-10.1257/jep.33.2.3",
            "title": "Automation and New Tasks",
            "authors": ["Daron Acemoglu", "Pascual Restrepo"],
            "publication": "Journal of Economic Perspectives",
            "date": "2019-05-01",
            "url": "https://doi.org/10.1257/jep.33.2.3",
            "source_tier": "tier_1",
            "source_type": "peer_reviewed",
            "relevance_score": 0.95,
            "conflict_of_interest": None,
            "notes": "Methodological benchmark.",
        }
    ]

    patch = AdversarialReviewerAgent().run(state, runtime)

    assert patch["adversarial_review"]
    assert patch["adversarial_review"][0]["recommendation"] == "WEAKENED"
