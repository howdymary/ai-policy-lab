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

    def _get(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return dict(response.json())
