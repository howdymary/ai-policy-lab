from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class WebSearchConnector(BaseConnector):
    def search(self, *, query: str, limit: int = 10) -> dict[str, Any]:
        if not self.settings.web_search_api_key:
            raise ConnectorConfigurationError(
                "WEB_SEARCH_API_KEY is required for live web search integration."
            )

        # Placeholder contract for future provider wiring.
        return {
            "query": query,
            "limit": limit,
            "status": "not_implemented",
            "message": "Wire your preferred web search provider here.",
        }
