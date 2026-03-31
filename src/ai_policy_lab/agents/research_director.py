from __future__ import annotations

from dataclasses import dataclass, field

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.report import render_report
from ai_policy_lab.research_tracks import (
    get_great_reallocation_subquestions,
    get_upskilling_pathways_subquestions,
    is_great_reallocation_question,
    is_upskilling_pathways_question,
)
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import sanitize_user_input
from ai_policy_lab.state import ResearchQuestion, ResearchQuestionPriority, ResearchState

INTAKE_PROMPT = """You are the Research Director of an autonomous policy research lab.
Decompose the root question into 3-7 specific research questions, define scope boundaries,
and assign each question to the right specialist agents."""

MIDCOURSE_PROMPT = """You are the Research Director at the phase-1.5 refinement checkpoint.
Review discovery outputs and decide whether to refine, add, or defer questions before analysis."""

SYNTHESIS_PROMPT = """You are the Research Director in synthesis mode.
Produce an executive summary and a final report that foreground evidence strength, limitations,
and unresolved disagreements."""


@dataclass(slots=True)
class ResearchDirectorAgent(BaseResearchAgent):
    stage: str
    name: str = field(init=False, default="research_director")
    phase: str = field(init=False)
    system_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        phase_map = {
            "intake": "phase_0_intake",
            "midcourse": "phase_1_5_refinement",
            "synthesis": "phase_4_synthesis",
        }
        prompt_map = {
            "intake": INTAKE_PROMPT,
            "midcourse": MIDCOURSE_PROMPT,
            "synthesis": SYNTHESIS_PROMPT,
        }
        self.phase = phase_map[self.stage]
        self.system_prompt = prompt_map[self.stage]

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        if self.stage == "intake":
            return self._run_intake(state)
        if self.stage == "midcourse":
            return self._run_midcourse(state)
        return self._run_synthesis(state)

    def _run_intake(self, state: ResearchState) -> StatePatch:
        if is_upskilling_pathways_question(state["root_question"]):
            return {
                "research_questions": get_upskilling_pathways_subquestions(),
                "current_phase": "phase_1_discovery",
            }

        if is_great_reallocation_question(state["root_question"]):
            return {
                "research_questions": get_great_reallocation_subquestions(),
                "current_phase": "phase_1_discovery",
            }

        root = sanitize_user_input(state["root_question"]).rstrip("?")
        questions = [
            self._question(
                identifier="rq-1",
                question=f"What does the existing literature say about: {root}?",
                priority="primary",
                assigned_to=["literature_review", "research_director"],
            ),
            self._question(
                identifier="rq-2",
                question=f"Which public and proprietary datasets can answer the core components of: {root}?",
                priority="primary",
                assigned_to=["data_scout"],
            ),
            self._question(
                identifier="rq-3",
                question=f"Which populations, geographies, industries, or occupations are most affected by: {root}?",
                priority="primary",
                assigned_to=["quantitative_analyst", "political_economy"],
            ),
            self._question(
                identifier="rq-4",
                question=f"What policy precedents or institutional responses are relevant to: {root}?",
                priority="secondary",
                assigned_to=["policy_scanner", "political_economy"],
            ),
            self._question(
                identifier="rq-5",
                question=f"What methodological or data gaps prevent stronger causal claims about: {root}?",
                priority="exploratory",
                assigned_to=["methodology_reviewer", "source_quality_auditor"],
            ),
        ]
        return {
            "research_questions": questions,
            "current_phase": "phase_1_discovery",
        }

    def _run_midcourse(self, state: ResearchState) -> StatePatch:
        existing_ids = {question["id"] for question in state["research_questions"]}
        additions: list[ResearchQuestion] = []
        if "rq-6" not in existing_ids:
            additions.append(
                self._question(
                    identifier="rq-6",
                    question="Which findings are blocked by missing or proprietary data, and what public-good datasets should be created next?",
                    priority="exploratory",
                    assigned_to=["data_scout", "research_director"],
                )
            )

        return {
            "research_questions": additions,
            "flagged_issues": [
                "Mid-course refinement is heuristic in the scaffold build; replace with live evidence-driven reprioritization before production use."
            ],
            "current_phase": "phase_2_analysis",
        }

    def _run_synthesis(self, state: ResearchState) -> StatePatch:
        finding_count = len(state["findings"])
        source_count = len(state["sources"])
        dataset_count = len(state["datasets"])
        adversarial_count = len(state["adversarial_review"])
        if state["runtime_mode"] == "mock":
            mode_sentence = (
                "THIS REPORT WAS GENERATED IN EXPLICIT MOCK MODE. Treat it as an orchestration proof, "
                "not as publishable research."
            )
        else:
            mode_sentence = (
                "This run used live LLM generation and live connector retrieval where the current build "
                "has implemented provider support."
            )
        executive_summary = (
            f"This run addressed the question: {state['root_question']}.\n\n"
            f"The scaffold completed all phases of the research DAG, cataloged {dataset_count} datasets, "
            f"captured {source_count} sources, produced {finding_count} validated findings, and logged "
            f"{adversarial_count} adversarial counterarguments. "
            f"{mode_sentence}"
        )

        state_for_render = dict(state)
        state_for_render["executive_summary"] = executive_summary
        state_for_render["datasets_manifest"] = state["datasets"]
        state_for_render["citation_list"] = state["sources"]
        state_for_render["research_agenda"] = [
            question for question in state["research_questions"] if question["status"] != "completed"
        ]
        state_for_render["current_phase"] = "complete"

        return {
            "executive_summary": executive_summary,
            "full_report": render_report(state_for_render),  # type: ignore[arg-type]
            "datasets_manifest": state["datasets"],
            "citation_list": state["sources"],
            "research_agenda": [
                question for question in state["research_questions"] if question["status"] != "completed"
            ],
            "current_phase": "complete",
        }

    def _question(
        self,
        *,
        identifier: str,
        question: str,
        priority: ResearchQuestionPriority,
        assigned_to: list[str],
    ) -> ResearchQuestion:
        return {
            "id": identifier,
            "question": question,
            "parent_question": None,
            "status": "pending",
            "priority": priority,
            "assigned_to": assigned_to,
        }
