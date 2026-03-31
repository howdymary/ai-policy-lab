from ai_policy_lab.report import render_report
from ai_policy_lab.state import make_initial_state


def test_render_report_includes_core_sections() -> None:
    state = make_initial_state(root_question="What is happening to AI-exposed occupations?")
    state["executive_summary"] = "A concise summary."
    report = render_report(state)

    assert "# AI Policy Research Lab Report" in report
    assert "## 1. Executive Summary" in report
    assert "## 4. Literature Review" in report
    assert "## 10. Data Appendix" in report
