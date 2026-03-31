from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ai_policy_lab.config import Settings


class ConnectorConfigurationError(RuntimeError):
    """Raised when a connector is missing required credentials."""


@dataclass(slots=True)
class BaseConnector:
    settings: Settings

    @property
    def timeout(self) -> float:
        return self.settings.http_timeout_seconds

    def _get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.text

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
