from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class ONETConnector(BaseConnector):
    base_url = "https://services.onetcenter.org/ws"

    def occupations(self) -> dict[str, Any]:
        if not self.settings.onet_username or not self.settings.onet_password:
            raise ConnectorConfigurationError(
                "ONET_USERNAME and ONET_PASSWORD are required for O*NET web service access."
            )

        return self._get(
            f"{self.base_url}/online/occupations",
            params=None,
        )
