from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.catalog import default_dataset_catalog
from ai_policy_lab.research_tracks import (
    discover_great_reallocation_data,
    is_great_reallocation_question,
)
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are a data librarian and research data specialist.
Catalog relevant datasets, map them to sub-questions, identify access constraints, and document data gaps."""


class DataScoutAgent(BaseResearchAgent):
    name = "data_scout"
    phase = "phase_1_discovery"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if is_great_reallocation_question(state["root_question"]):
            result = discover_great_reallocation_data(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            summary = runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=(
                    f"Root question: {state['root_question']}\n"
                    f"Sub-questions: {[item['question'] for item in state['research_questions']]}\n"
                    "Turn the following retrieval notes into a data availability matrix and data gap analysis:\n"
                    f"{result.summary}"
                ),
                fallback=result.summary,
            )
            return {
                "datasets": result.datasets,
                "sources": result.sources,
                "data_availability_assessment": summary,
                "flagged_issues": result.issues,
            }

        datasets = default_dataset_catalog()
        prompt = (
            f"Root question: {state['root_question']}\n"
            f"Constraints: {state['domain_constraints']}\n"
            "Summarize the discovered data landscape, likely joins, and the most important missing data."
        )
        fallback = (
            f"Cataloged {len(datasets)} canonical labor-market datasets spanning BLS, Census, FRED, O*NET, "
            "IPUMS, and proprietary postings sources. The biggest likely gaps are firm-level AI adoption data, "
            "worker-level transition costs, and consistently accessible real-time postings data."
        )
        return {
            "datasets": datasets,
            "data_availability_assessment": runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                fallback=fallback,
            ),
        }
