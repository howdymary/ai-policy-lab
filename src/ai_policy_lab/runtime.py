from __future__ import annotations

import logging
from dataclasses import dataclass

from ai_policy_lab.config import Settings
from ai_policy_lab.llm import LLMNotConfiguredError, LLMResponseError, OpenAICompatibleLLM

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchRuntime:
    settings: Settings
    llm: OpenAICompatibleLLM

    @classmethod
    def from_env(cls) -> ResearchRuntime:
        settings = Settings.load()
        return cls(settings=settings, llm=OpenAICompatibleLLM(settings=settings))

    def use_live_llm(self) -> bool:
        return not self.settings.use_mock and self.llm.is_available()

    def maybe_generate(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        fallback: str,
        temperature: float | None = None,
    ) -> str:
        if not self.use_live_llm():
            return fallback
        try:
            return self.llm.generate(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except (LLMNotConfiguredError, LLMResponseError):
            logger.warning("Falling back for agent %s after LLM generation failed.", agent_name)
            return fallback
