from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class ScholarSearchConnector(BaseConnector):
    def search(self, *, query: str, limit: int = 10) -> dict[str, Any]:
        if not self.settings.semantic_scholar_api_key:
            raise ConnectorConfigurationError(
                "SEMANTIC_SCHOLAR_API_KEY is required for live scholar search."
            )

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,url,citationCount,publicationTypes",
        }
        return self._get(url, params=params)

    def scholar_url(self, query: str) -> str:
        return f"https://scholar.google.com/scholar?q={quote_plus(query)}"
