from pathlib import Path

from ai_policy_lab.agents import SwarmCoordinatorAgent
from ai_policy_lab.config import Settings
from ai_policy_lab.llm import OpenAICompatibleLLM
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import make_initial_state


def _runtime() -> ResearchRuntime:
    settings = Settings(
        use_mock=True,
        runs_dir=Path("runs"),
        cache_ttl_hours=24,
        default_model="mock-model",
        default_temperature=0.2,
        openai_base_url="http://localhost:11434/v1",
        openai_api_key="ollama",
        http_timeout_seconds=30.0,
        bls_api_key=None,
        fred_api_key=None,
        census_api_key=None,
        onet_username=None,
        onet_password=None,
        semantic_scholar_api_key=None,
        web_search_api_key=None,
        crossref_contact_email=None,
    )
    return ResearchRuntime(settings=settings, llm=OpenAICompatibleLLM(settings=settings))


def test_swarm_intake_plans_question_and_standing_quality_tasks() -> None:
    state = make_initial_state(root_question="How is AI adoption affecting local labor markets?")
    state["research_questions"] = [
        {
            "id": "rq-1",
            "question": "Which workers are exposed?",
            "parent_question": None,
            "status": "pending",
            "priority": "primary",
            "assigned_to": ["literature_review", "quantitative_analyst"],
        }
    ]

    patch = SwarmCoordinatorAgent("intake_plan").run(state, _runtime())
    tasks = patch["swarm_tasks"]

    assert {task["assigned_agent"] for task in tasks} >= {
        "literature_review",
        "quantitative_analyst",
        "source_quality_auditor",
        "methodology_reviewer",
        "adversarial_reviewer",
        "research_director",
    }
    assert any(task["output_contract"] for task in tasks)
    assert all(task["status"] == "queued" for task in tasks)


def test_swarm_midcourse_marks_discovery_tasks_from_shared_outputs() -> None:
    state = make_initial_state(root_question="How is AI adoption affecting local labor markets?")
    state["research_questions"] = [
        {
            "id": "rq-1",
            "question": "Which sources and datasets are available?",
            "parent_question": None,
            "status": "pending",
            "priority": "primary",
            "assigned_to": ["literature_review", "data_scout", "policy_scanner"],
        }
    ]
    state["swarm_tasks"] = SwarmCoordinatorAgent("intake_plan").run(state, _runtime())[
        "swarm_tasks"
    ]
    state["existing_literature_summary"] = "Retrieved source inventory."
    state["data_availability_assessment"] = "Cataloged datasets."
    state["policy_landscape_summary"] = "Federal policy scan."

    patch = SwarmCoordinatorAgent("midcourse_review").run(state, _runtime())
    latest = {task["id"]: task for task in [*state["swarm_tasks"], *patch["swarm_tasks"]]}

    assert latest["swarm-rq-1-literature_review"]["status"] == "completed"
    assert latest["swarm-rq-1-data_scout"]["status"] == "completed"
    assert latest["swarm-rq-1-policy_scanner"]["status"] == "completed"
