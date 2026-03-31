from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.state import ResearchState, make_agent_log_entry
from ai_policy_lab.utils import compact_whitespace

StatePatch = dict[str, Any]


class BaseResearchAgent(ABC):
    name: str
    phase: str
    system_prompt: str

    @abstractmethod
    def run(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        raise NotImplementedError

    def summarize_for_log(self, patch: StatePatch) -> str:
        pieces: list[str] = []
        for key in (
            "existing_literature_summary",
            "data_availability_assessment",
            "policy_landscape_summary",
            "methodology_description",
            "source_audit_report",
            "methodology_review",
            "executive_summary",
        ):
            value = patch.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value)
                break

        if not pieces:
            adversarial_items = patch.get("adversarial_review")
            if isinstance(adversarial_items, list) and adversarial_items:
                pieces.append(
                    f"Adversarial review completed for {len(adversarial_items)} findings."
                )

        if not pieces:
            pieces.append(f"{self.name} completed.")

        return compact_whitespace(pieces[0])[:240]

    def invoke(self, state: ResearchState, runtime: ResearchRuntime) -> StatePatch:
        patch = self.run(state, runtime)
        patch["agent_log"] = [make_agent_log_entry(self.name, self.phase, self.summarize_for_log(patch))]
        patch["messages"] = [{"role": "assistant", "content": self.summarize_for_log(patch)}]
        return patch

    def as_node(self, runtime: ResearchRuntime) -> Callable[[ResearchState], StatePatch]:
        def _node(state: ResearchState) -> StatePatch:
            return self.invoke(state, runtime)

        return _node
