from ai_policy_lab.graph import run_research
from ai_policy_lab.runtime import ResearchRuntime


def test_run_research_completes_mock_graph(monkeypatch) -> None:
    monkeypatch.setenv("APL_USE_MOCK", "true")
    runtime = ResearchRuntime.from_env()

    final_state = run_research(
        runtime=runtime,
        root_question="How is AI adoption reshaping occupational structure in the U.S. labor market?",
        domain_constraints=["United States", "2015-2025"],
        quality_floor="tier_2",
    )

    assert final_state["current_phase"] == "complete"
    assert final_state["run_status"] == "completed"
    assert final_state["run_errors"] == []
    assert len(final_state["research_questions"]) >= 5
    assert final_state["datasets_manifest"]
    assert "Executive Summary" in final_state["full_report"]
    assert any("BLOCKER" in issue or "WARNING" in issue for issue in final_state["flagged_issues"])


def test_run_research_returns_partial_state_on_graph_failure(monkeypatch) -> None:
    class BrokenGraph:
        def stream(self, initial_state, stream_mode="values"):
            _ = stream_mode
            yield {**initial_state, "existing_literature_summary": "partial"}
            raise RuntimeError("boom")

    monkeypatch.setenv("APL_USE_MOCK", "true")
    runtime = ResearchRuntime.from_env()
    monkeypatch.setattr("ai_policy_lab.graph.build_graph", lambda _runtime: BrokenGraph())

    final_state = run_research(
        runtime=runtime,
        root_question="How is AI adoption reshaping occupational structure in the U.S. labor market?",
        domain_constraints=["United States"],
        quality_floor="tier_2",
    )

    assert final_state["run_status"] == "failed"
    assert final_state["run_errors"] == ["boom"]
    assert final_state["existing_literature_summary"] == "partial"
    assert "boom" in final_state["full_report"]


def test_run_research_uses_specialized_great_reallocation_track(monkeypatch) -> None:
    monkeypatch.setenv("APL_USE_MOCK", "true")
    runtime = ResearchRuntime.from_env()

    final_state = run_research(
        runtime=runtime,
        root_question="How is AI-driven automation causing a great reallocation of jobs across US metropolitan labor markets?",
        domain_constraints=["United States"],
        quality_floor="tier_2",
    )

    assert len(final_state["research_questions"]) == 6
    assert any(
        "highest exposure to ai-driven task automation" in question["question"].lower()
        for question in final_state["research_questions"]
    )
    assert any(dataset["id"] == "bls-jolts" for dataset in final_state["datasets"])
    assert any(source["id"].startswith("doi-") for source in final_state["sources"])
