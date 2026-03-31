from ai_policy_lab.state import make_initial_state


def test_make_initial_state_sets_expected_defaults() -> None:
    state = make_initial_state(
        root_question="How is AI adoption affecting labor markets?",
        domain_constraints=["United States", "2015-2025"],
        quality_floor="tier_2",
    )

    assert state["root_question"] == "How is AI adoption affecting labor markets?"
    assert state["domain_constraints"] == ["United States", "2015-2025"]
    assert state["quality_floor"] == "tier_2"
    assert state["research_questions"] == []
    assert state["datasets"] == []
    assert state["adversarial_review"] == []
    assert state["current_phase"] == "phase_0_intake"
    assert state["runtime_mode"] == "live"
    assert state["run_status"] == "running"
    assert state["run_errors"] == []
    assert state["agent_log"]
