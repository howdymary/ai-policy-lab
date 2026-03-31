from ai_policy_lab.report import render_report
from ai_policy_lab.state import make_initial_state


def test_render_report_includes_core_sections() -> None:
    state = make_initial_state(root_question="What is happening to AI-exposed occupations?")
    state["runtime_mode"] = "mock"
    state["executive_summary"] = "A concise summary."
    state["citation_list"] = [
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
    state["findings"] = [
        {
            "agent": "quantitative_analyst",
            "claim": "AI exposure is uneven across metros.",
            "evidence_strength": "moderate",
            "supporting_sources": ["src-1"],
            "confidence": 0.66,
            "methodology": "Descriptive index.",
            "limitations": ["No causal identification."],
            "contradicts": [],
        }
    ]
    state["adversarial_review"] = [
        {
            "finding_claim": "AI exposure is uneven across metros.",
            "counterargument": "Exposure is only a proxy for realized adoption.",
            "evidence_strength": "suggestive",
            "recommendation": "WEAKENED",
            "supporting_sources": ["src-1"],
        }
    ]
    report = render_report(state)

    assert "# AI Policy Research Lab Report" in report
    assert "THIS REPORT WAS GENERATED IN EXPLICIT MOCK MODE" in report
    assert "## 1. Executive Summary" in report
    assert "## 4. Literature Review" in report
    assert "## 9. Counterarguments and Rebuttals" in report
    assert "## 11. Data Appendix" in report
    assert "### Retrieved Sources" in report
    assert "[src-1] Example source" in report
    assert "AI exposure is uneven across metros." in report
    assert "Exposure is only a proxy for realized adoption." in report


def test_render_report_handles_missing_optional_fields() -> None:
    state = make_initial_state(root_question="What is happening to AI-exposed occupations?")
    state["executive_summary"] = "Summary."
    state["findings"] = [{"agent": "quantitative_analyst"}]  # type: ignore[list-item]
    state["datasets"] = [{"id": "dataset-1"}]  # type: ignore[list-item]
    state["adversarial_review"] = [{"recommendation": "STANDS"}]

    report = render_report(state)

    assert "No claim" in report
    assert "Unnamed" in report
    assert "[STANDS]" in report
