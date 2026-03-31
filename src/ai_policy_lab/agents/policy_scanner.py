from __future__ import annotations

from ai_policy_lab.agents.base import BaseResearchAgent, StatePatch
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState

SYSTEM_PROMPT = """You are a policy analyst focused on legislation, regulation, institutional initiatives,
and policy precedents. Report institutional facts, not opinions."""


class PolicyScannerAgent(BaseResearchAgent):
    name = "policy_scanner"
    phase = "phase_1_discovery"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        prompt = (
            f"Question: {state['root_question']}\n"
            "Summarize the active policy landscape, key institutions, and historical precedents."
        )
        fallback = (
            "Mock mode: policy scanner not connected to congress.gov, Federal Register, or agency sources yet. "
            "In production, this node should extract active legislation, agency initiatives, evaluation evidence, "
            "and relevant historical precedents."
        )
        return {
            "policy_landscape_summary": runtime.maybe_generate(
                agent_name=self.name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                fallback=fallback,
            )
        }
