from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import (
    PhaseName,
    ResearchQuestion,
    ResearchState,
    SwarmTask,
    SwarmTaskStatus,
)

StageName = Literal[
    "intake_plan",
    "midcourse_review",
    "analysis_review",
    "quality_review",
    "synthesis_review",
]
StatusFunction = Callable[[SwarmTask, ResearchState], tuple[SwarmTaskStatus, str]]

SYSTEM_PROMPT = """You are the swarm coordinator for an autonomous policy research lab.
Assign specialist work, preserve a task ledger, and flag coverage gaps before synthesis."""

_DISCOVERY_AGENTS = {"literature_review", "data_scout", "policy_scanner"}
_ANALYSIS_AGENTS = {"quantitative_analyst", "political_economy", "economic_complexity"}
_QUALITY_AGENTS = {"source_quality_auditor", "methodology_reviewer", "adversarial_reviewer"}
_SPECIALIST_AGENTS = _DISCOVERY_AGENTS | _ANALYSIS_AGENTS | _QUALITY_AGENTS

_AGENT_PHASES: dict[str, PhaseName] = {
    "literature_review": "phase_1_discovery",
    "data_scout": "phase_1_discovery",
    "policy_scanner": "phase_1_discovery",
    "quantitative_analyst": "phase_2_analysis",
    "political_economy": "phase_2_analysis",
    "economic_complexity": "phase_2_analysis",
    "source_quality_auditor": "phase_3_quality_gate",
    "methodology_reviewer": "phase_3_quality_gate",
    "adversarial_reviewer": "phase_3_quality_gate",
    "research_director": "phase_4_synthesis",
}

_EVIDENCE_REQUIREMENTS: dict[str, list[str]] = {
    "literature_review": [
        "Prioritize Tier 1/Tier 2 literature.",
        "Separate consensus, live debate, and unresolved gaps.",
        "Attach source IDs that can be audited later.",
    ],
    "data_scout": [
        "Catalog public datasets before proprietary or speculative sources.",
        "Record access method, geographic coverage, and normalization state.",
        "Name empirical blockers explicitly.",
    ],
    "policy_scanner": [
        "Prefer primary institutional records.",
        "Distinguish enacted policy, proposed rules, and historical precedent.",
        "Record missing legislative or agency coverage.",
    ],
    "quantitative_analyst": [
        "Run reproducible descriptive analysis before causal claims.",
        "Report effect sizes or index values only with methodology.",
        "Declare uncertainty and data limitations.",
    ],
    "political_economy": [
        "Trace distributional consequences to institutions and bargaining power.",
        "Separate descriptive evidence from normative interpretation.",
        "Name affected groups and mechanisms.",
    ],
    "economic_complexity": [
        "Connect place-based capability, adjacency, and transition logic.",
        "Avoid national-average claims when regional data are required.",
        "Flag missing network or RCA metrics.",
    ],
    "source_quality_auditor": [
        "Check provenance, source tier, recency, and conflicts of interest.",
        "Flag unsupported claims as blockers.",
    ],
    "methodology_reviewer": [
        "Check design fit, replicability, claim scope, and robustness gaps.",
        "Downgrade causal language when only descriptive evidence exists.",
    ],
    "adversarial_reviewer": [
        "Produce strongest counterargument for each finding.",
        "Attach counter-evidence source IDs when available.",
    ],
    "research_director": [
        "Synthesize only after discovery, analysis, and quality gates are logged.",
        "Carry unresolved blockers into the final research agenda.",
    ],
}

_OUTPUT_CONTRACTS: dict[str, str] = {
    "literature_review": "existing_literature_summary plus auditable SourceRecord entries",
    "data_scout": "DatasetRecord entries plus data_availability_assessment",
    "policy_scanner": "policy_landscape_summary plus primary-source policy SourceRecord entries",
    "quantitative_analyst": "quantitative_results, methodology_description, and empirical findings",
    "political_economy": "one or more distributional/institutional findings",
    "economic_complexity": "one or more capability, adjacency, or place-based findings",
    "source_quality_auditor": "source_audit_report and citation quality blockers",
    "methodology_reviewer": "methodology_review and claim-scope warnings",
    "adversarial_reviewer": "adversarial_review items for available findings",
    "research_director": "executive_summary, full_report, agenda, and citation/dataset manifests",
}


@dataclass(slots=True)
class SwarmCoordinatorAgent(BaseResearchAgent):
    stage: StageName
    name: str = field(init=False, default="swarm_coordinator")
    phase: str = field(init=False)
    system_prompt: str = field(init=False, default=SYSTEM_PROMPT)

    def __post_init__(self) -> None:
        phase_map = {
            "intake_plan": "phase_0_intake",
            "midcourse_review": "phase_1_5_refinement",
            "analysis_review": "phase_2_analysis",
            "quality_review": "phase_3_quality_gate",
            "synthesis_review": "phase_4_synthesis",
        }
        self.phase = phase_map[self.stage]

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        _ = runtime
        if self.stage == "intake_plan":
            return self._run_intake_plan(state)
        if self.stage == "midcourse_review":
            return self._run_midcourse_review(state)
        if self.stage == "analysis_review":
            return self._run_analysis_review(state)
        if self.stage == "quality_review":
            return self._run_quality_review(state)
        return self._run_synthesis_review(state)

    def _run_intake_plan(self, state: ResearchState) -> StatePatch:
        tasks = _newly_planned_tasks(state)
        issues: list[str] = []
        if not tasks:
            issues.append("NOTE: Swarm coordinator found no new specialist tasks after intake.")
        return {
            "swarm_tasks": tasks,
            "flagged_issues": issues,
            "current_phase": "phase_1_discovery",
        }

    def _run_midcourse_review(self, state: ResearchState) -> StatePatch:
        new_tasks = _settle_elapsed_phase_tasks(
            tasks=_newly_planned_tasks(state),
            state=state,
            elapsed_phase="phase_1_discovery",
        )
        task_events = [
            *_review_tasks_for_phase(
                state=state,
                phase="phase_1_discovery",
                status_fn=_discovery_status,
            ),
            *new_tasks,
        ]
        return {
            "swarm_tasks": task_events,
            "flagged_issues": _blocked_task_issues(task_events, "discovery"),
            "current_phase": "phase_2_analysis",
        }

    def _run_analysis_review(self, state: ResearchState) -> StatePatch:
        task_events = _review_tasks_for_phase(
            state=state,
            phase="phase_2_analysis",
            status_fn=_analysis_status,
        )
        return {
            "swarm_tasks": task_events,
            "flagged_issues": _blocked_task_issues(task_events, "analysis"),
            "current_phase": "phase_3_quality_gate",
        }

    def _run_quality_review(self, state: ResearchState) -> StatePatch:
        task_events = [
            *_review_tasks_for_phase(
                state=state,
                phase="phase_3_quality_gate",
                status_fn=_quality_status,
            ),
            *_review_tasks_for_phase(
                state=state,
                phase="phase_4_synthesis",
                status_fn=_synthesis_status,
                force_status="in_progress",
            ),
        ]
        return {
            "swarm_tasks": task_events,
            "flagged_issues": _blocked_task_issues(task_events, "quality gate"),
            "current_phase": "phase_4_synthesis",
        }

    def _run_synthesis_review(self, state: ResearchState) -> StatePatch:
        task_events = _review_tasks_for_phase(
            state=state,
            phase="phase_4_synthesis",
            status_fn=_synthesis_status,
        )
        return {
            "swarm_tasks": task_events,
            "flagged_issues": _blocked_task_issues(task_events, "synthesis"),
            "current_phase": "complete",
        }


def _newly_planned_tasks(state: ResearchState) -> list[SwarmTask]:
    latest = _latest_task_map(state["swarm_tasks"])
    planned: list[SwarmTask] = []
    for question in state["research_questions"]:
        for agent_name in _specialist_assignees(question):
            task = _make_question_task(question=question, agent_name=agent_name)
            if task["id"] not in latest:
                planned.append(task)

    for task in _standing_quality_and_synthesis_tasks():
        if task["id"] not in latest and task["id"] not in {item["id"] for item in planned}:
            planned.append(task)
    return planned


def _specialist_assignees(question: ResearchQuestion) -> list[str]:
    assignees = [agent for agent in question["assigned_to"] if agent in _SPECIALIST_AGENTS]
    if assignees:
        return assignees
    if not question["assigned_to"]:
        return ["literature_review"]
    return []


def _make_question_task(*, question: ResearchQuestion, agent_name: str) -> SwarmTask:
    phase = _AGENT_PHASES[agent_name]
    return {
        "id": f"swarm-{question['id']}-{agent_name}",
        "research_question_id": question["id"],
        "objective": question["question"],
        "assigned_agent": agent_name,
        "phase": phase,
        "status": "queued",
        "depends_on": _dependencies_for_phase(phase, question["id"]),
        "evidence_requirements": _EVIDENCE_REQUIREMENTS[agent_name],
        "output_contract": _OUTPUT_CONTRACTS[agent_name],
        "notes": "Queued by swarm coordinator after research-question decomposition.",
    }


def _standing_quality_and_synthesis_tasks() -> list[SwarmTask]:
    tasks: list[SwarmTask] = []
    for agent_name in ("source_quality_auditor", "methodology_reviewer", "adversarial_reviewer"):
        tasks.append(
            {
                "id": f"swarm-quality-{agent_name}",
                "research_question_id": "quality-gate",
                "objective": "Validate source provenance, method fit, and adversarial robustness before synthesis.",
                "assigned_agent": agent_name,
                "phase": _AGENT_PHASES[agent_name],
                "status": "queued",
                "depends_on": ["phase_1_discovery", "phase_2_analysis"],
                "evidence_requirements": _EVIDENCE_REQUIREMENTS[agent_name],
                "output_contract": _OUTPUT_CONTRACTS[agent_name],
                "notes": "Standing quality-gate task for every autonomous research run.",
            }
        )
    tasks.append(
        {
            "id": "swarm-synthesis-research_director",
            "research_question_id": "synthesis",
            "objective": "Synthesize validated findings, unresolved gaps, and the follow-on research agenda.",
            "assigned_agent": "research_director",
            "phase": "phase_4_synthesis",
            "status": "queued",
            "depends_on": ["phase_1_discovery", "phase_2_analysis", "phase_3_quality_gate"],
            "evidence_requirements": _EVIDENCE_REQUIREMENTS["research_director"],
            "output_contract": _OUTPUT_CONTRACTS["research_director"],
            "notes": "Standing synthesis task for every autonomous research run.",
        }
    )
    return tasks


def _dependencies_for_phase(phase: PhaseName, question_id: str) -> list[str]:
    if phase == "phase_1_discovery":
        return []
    if phase == "phase_2_analysis":
        return [f"swarm-{question_id}-literature_review", f"swarm-{question_id}-data_scout"]
    if phase == "phase_3_quality_gate":
        return ["phase_1_discovery", "phase_2_analysis"]
    if phase == "phase_4_synthesis":
        return ["phase_3_quality_gate"]
    return []


def _review_tasks_for_phase(
    *,
    state: ResearchState,
    phase: PhaseName,
    status_fn: StatusFunction,
    force_status: SwarmTaskStatus | None = None,
) -> list[SwarmTask]:
    updates: list[SwarmTask] = []
    latest = _latest_task_map(state["swarm_tasks"])
    for task in latest.values():
        if task["phase"] != phase or task["status"] in {"completed", "blocked", "deferred"}:
            continue
        if force_status is None:
            status, notes = status_fn(task, state)
        else:
            status = force_status
            notes = "Synthesis task is ready for the Research Director after quality review."
        if status != task["status"] or notes != task["notes"]:
            updates.append({**task, "status": status, "notes": notes})
    return updates


def _settle_elapsed_phase_tasks(
    *,
    tasks: list[SwarmTask],
    state: ResearchState,
    elapsed_phase: PhaseName,
) -> list[SwarmTask]:
    settled: list[SwarmTask] = []
    for task in tasks:
        if task["phase"] != elapsed_phase:
            settled.append(task)
            continue
        status, notes = _discovery_status(task, state)
        if status == "completed":
            settled.append(
                {
                    **task,
                    "status": "completed",
                    "notes": f"Late midcourse task satisfied by shared discovery outputs. {notes}",
                }
            )
        else:
            settled.append(
                {
                    **task,
                    "status": "deferred",
                    "notes": "Late midcourse discovery task needs a future iterative swarm pass.",
                }
            )
    return settled


def _discovery_status(task: SwarmTask, state: ResearchState) -> tuple[SwarmTaskStatus, str]:
    agent_name = task["assigned_agent"]
    if agent_name == "literature_review":
        if state["existing_literature_summary"].strip() or state["sources"]:
            return "completed", "Literature discovery produced a source inventory or summary."
    elif agent_name == "data_scout":
        if state["datasets"] or state["data_availability_assessment"].strip():
            return "completed", "Data discovery produced datasets or an availability assessment."
    elif agent_name == "policy_scanner" and state["policy_landscape_summary"].strip():
        return "completed", "Policy scanner produced institutional context."
    return "blocked", f"{agent_name} did not leave enough discovery output for this task."


def _analysis_status(task: SwarmTask, state: ResearchState) -> tuple[SwarmTaskStatus, str]:
    agent_name = task["assigned_agent"]
    if agent_name == "quantitative_analyst":
        status = str(state["quantitative_results"].get("status", ""))
        if status == "completed":
            return "completed", "Quantitative analyst completed a reproducible analysis artifact."
        if state["methodology_description"].strip():
            return "deferred", "Quantitative analyst produced a methods plan, but empirical execution remains pending."
    if any(finding["agent"] == agent_name for finding in state["findings"]):
        return "completed", f"{agent_name} produced at least one finding."
    return "blocked", f"{agent_name} did not produce a finding for this task."


def _quality_status(task: SwarmTask, state: ResearchState) -> tuple[SwarmTaskStatus, str]:
    agent_name = task["assigned_agent"]
    if agent_name == "source_quality_auditor" and state["source_audit_report"].strip():
        return "completed", "Source quality audit completed."
    if agent_name == "methodology_reviewer" and state["methodology_review"].strip():
        return "completed", "Methodology review completed."
    if agent_name == "adversarial_reviewer":
        if state["adversarial_review"]:
            return "completed", "Adversarial review produced counterarguments."
        return "blocked", "Adversarial review could not produce counterarguments without findings."
    return "blocked", f"{agent_name} did not produce a quality-gate artifact."


def _synthesis_status(task: SwarmTask, state: ResearchState) -> tuple[SwarmTaskStatus, str]:
    if state["executive_summary"].strip() and state["full_report"].strip():
        return "completed", "Research Director produced synthesis artifacts."
    return "in_progress", "Synthesis task is ready for the Research Director after quality review."


def _latest_task_map(tasks: list[SwarmTask]) -> dict[str, SwarmTask]:
    latest: dict[str, SwarmTask] = {}
    for task in tasks:
        latest[task["id"]] = task
    return latest


def _blocked_task_issues(task_events: list[SwarmTask], label: str) -> list[str]:
    blocked = [task for task in task_events if task["status"] == "blocked"]
    if not blocked:
        return []
    rendered = ", ".join(f"{task['id']}->{task['assigned_agent']}" for task in blocked[:5])
    if len(blocked) > 5:
        rendered = f"{rendered}, and {len(blocked) - 5} more"
    return [f"WARNING: Swarm {label} review found blocked specialist tasks: {rendered}."]
