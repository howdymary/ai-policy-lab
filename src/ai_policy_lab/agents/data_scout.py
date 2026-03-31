from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.catalog import default_dataset_catalog, infer_dataset_domain
from ai_policy_lab.research_tracks import (
    discover_great_reallocation_data,
    discover_upskilling_pathways_data,
    is_ai_labor_market_question,
    is_great_reallocation_question,
    is_upskilling_pathways_question,
)
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import wrap_user_content, wrap_user_list
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are a data librarian and research data specialist.
Catalog relevant datasets, map them to sub-questions, identify access constraints, and document data gaps."""


class DataScoutAgent(BaseResearchAgent):
    name = "data_scout"
    phase = "phase_1_discovery"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if is_upskilling_pathways_question(state["root_question"]):
            result = discover_upskilling_pathways_data(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            summary = runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=(
                    f"{wrap_user_content('root_question', state['root_question'])}\n"
                    f"{wrap_user_list('sub_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
                    "Turn the following retrieval notes into a data availability matrix and data gap analysis:\n"
                    f"{wrap_user_content('retrieval_notes', result.summary)}"
                ),
                fallback=result.summary,
            )
            return {
                "datasets": result.datasets,
                "sources": result.sources,
                "data_availability_assessment": summary,
                "flagged_issues": result.issues,
            }

        if is_great_reallocation_question(state["root_question"]):
            result = discover_great_reallocation_data(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            summary = runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=(
                    f"{wrap_user_content('root_question', state['root_question'])}\n"
                    f"{wrap_user_list('sub_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
                    "Turn the following retrieval notes into a data availability matrix and data gap analysis:\n"
                    f"{wrap_user_content('retrieval_notes', result.summary)}"
                ),
                fallback=result.summary,
            )
            return {
                "datasets": result.datasets,
                "sources": result.sources,
                "data_availability_assessment": summary,
                "flagged_issues": result.issues,
            }

        if is_ai_labor_market_question(state["root_question"], state["domain_constraints"]):
            result = discover_great_reallocation_data(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            summary = runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=(
                    f"{wrap_user_content('root_question', state['root_question'])}\n"
                    f"{wrap_user_list('sub_questions', [item['question'] for item in state['research_questions']], item_tag='question')}\n"
                    "Turn the following retrieval notes into a data availability matrix and data gap analysis tailored "
                    "to the occupations or worker segments in the question:\n"
                    f"{wrap_user_content('retrieval_notes', result.summary)}"
                ),
                fallback=result.summary,
            )
            issues = list(result.issues)
            issues.append(
                "NOTE: Generic AI labor-market questions reuse the Great Reallocation public-data spine until a broader live dataset router is added."
            )
            return {
                "datasets": result.datasets,
                "sources": result.sources,
                "data_availability_assessment": summary,
                "flagged_issues": issues,
            }

        domain = infer_dataset_domain(
            question=state["root_question"],
            constraints=state["domain_constraints"],
        )
        datasets = default_dataset_catalog(domain=domain)
        prompt = (
            f"{wrap_user_content('root_question', state['root_question'])}\n"
            f"{wrap_user_list('constraints', state['domain_constraints'], item_tag='constraint')}\n"
            "Summarize the discovered data landscape, likely joins, and the most important missing data."
        )
        if datasets:
            fallback = (
                f"Cataloged {len(datasets)} domain-relevant datasets for the inferred '{domain}' research area. "
                "The biggest likely gaps are restricted-use microdata, contemporaneous program evaluation inputs, "
                "and researcher-built linkage tables that turn raw sources into analysis-ready panels."
            )
            generic_issues: list[str] = []
        else:
            fallback = (
                "No domain-specific datasets are preconfigured for this question yet. Use discovery connectors "
                "and source review to identify the right empirical base before quantitative analysis."
            )
            generic_issues = [
                f"WARNING: No domain-aware dataset catalog is available yet for inferred domain '{domain}'."
            ]
        return {
            "datasets": datasets,
            "data_availability_assessment": runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                fallback=fallback,
            ),
            "flagged_issues": generic_issues,
        }
