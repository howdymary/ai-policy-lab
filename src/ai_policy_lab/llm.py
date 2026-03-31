from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ai_policy_lab.config import Settings


class LLMNotConfiguredError(RuntimeError):
    """Raised when the runtime is asked to use a live LLM without credentials."""


@dataclass(slots=True)
class OpenAICompatibleLLM:
    settings: Settings

    def is_available(self) -> bool:
        return bool(self.settings.openai_api_key)

    def generate(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        if not self.is_available():
            raise LLMNotConfiguredError(
                "OPENAI_API_KEY is not configured. Set APL_USE_MOCK=true or provide credentials."
            )

        payload: dict[str, Any] = {
            "model": self.settings.model_for(agent_name),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.settings.default_temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.settings.http_timeout_seconds) as client:
            response = client.post(
                f"{self.settings.openai_base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()

        return str(body["choices"][0]["message"]["content"]).strip()
