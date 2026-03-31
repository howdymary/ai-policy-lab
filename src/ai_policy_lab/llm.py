from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from ai_policy_lab.config import Settings


class LLMNotConfiguredError(RuntimeError):
    """Raised when the runtime is asked to use a live LLM without credentials."""


class LLMResponseError(RuntimeError):
    """Raised when an LLM request fails or returns malformed data."""


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

        try:
            with httpx.Client(timeout=self.settings.http_timeout_seconds) as client:
                response = client.post(
                    f"{self.settings.openai_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
            if not isinstance(body, dict):
                raise ValueError("LLM response body was not a JSON object")
            choices = body.get("choices")
            if not choices or not isinstance(choices, list):
                raise ValueError(f"LLM returned no choices: {body.get('error', 'unknown')}")
            content = choices[0].get("message", {}).get("content")
            if content is None:
                raise ValueError("LLM response missing content field")
            return str(content).strip()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise LLMResponseError(f"LLM request failed with HTTP status error: {detail}") from exc
        except httpx.TimeoutException as exc:
            raise LLMResponseError("LLM request timed out.") from exc
        except json.JSONDecodeError as exc:
            raise LLMResponseError("LLM returned invalid JSON.") from exc
        except (AttributeError, ValueError, KeyError) as exc:
            raise LLMResponseError(f"LLM response validation failed: {exc}") from exc
