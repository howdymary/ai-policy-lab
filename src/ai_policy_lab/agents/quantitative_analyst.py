from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.research_tracks import (
    analyze_great_reallocation_exposure,
    is_great_reallocation_question,
    is_upskilling_pathways_question,
)
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are a quantitative researcher in applied econometrics and data science.
Design a reproducible empirical workflow and report effect sizes, uncertainty, and limitations."""


class QuantitativeAnalystAgent(BaseResearchAgent):
    name = "quantitative_analyst"
    phase = "phase_2_analysis"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if is_great_reallocation_question(state["root_question"]) or is_upskilling_pathways_question(
            state["root_question"]
        ):
            result = analyze_great_reallocation_exposure(
                settings=runtime.settings,
                use_live_lookup=not runtime.settings.use_mock,
            )
            issues = list(result.issues)
            methodology_description = result.methodology_description
            if is_upskilling_pathways_question(state["root_question"]):
                methodology_description = methodology_description.replace(
                    "for the Great Reallocation question",
                    "for this Brookings continuation prompt",
                )
                issues.append(
                    "WARNING: This Brookings continuation prompt still lacks a programmatic mapping of Brookings' 15 clusters and stepping-stone occupations, so current quantitative findings describe the AI exposure backdrop rather than full pathway disruption."
                )
            return {
                "findings": [
                    finding for finding in result.findings if finding["agent"] == self.name
                ],
                "methodology_description": methodology_description,
                "quantitative_results": result.quantitative_results,
                "flagged_issues": issues,
            }

        prompt = (
            f"Question: {state['root_question']}\n"
            f"Datasets available: {[dataset['name'] for dataset in state['datasets']]}\n"
            "Draft a methods plan that starts with descriptive analysis and only then considers causal designs."
        )
        execution_label = "mock mode" if runtime.settings.use_mock else "this run"
        fallback = (
            "Quantitative analysis scaffold prepared. Start with descriptive trend analysis, subgroup splits, and "
            "place-based comparisons; then evaluate whether any causal design is credible given the available data. "
            f"No empirical estimation has been executed yet in {execution_label}."
        )
        return {
            "methodology_description": runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                fallback=fallback,
            ),
            "quantitative_results": {
                "status": "not_executed",
                "mode": "mock" if runtime.settings.use_mock else "live_scaffold",
            },
            "flagged_issues": [
                "Quantitative analysis has not yet been run on live normalized datasets; current output is a methodological workplan."
            ],
        }
