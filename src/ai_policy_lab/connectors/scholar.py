from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from ai_policy_lab.connectors.base import BaseConnector


class ScholarSearchConnector(BaseConnector):
    rate_limit_max_calls = 90
    rate_limit_period_seconds = 300.0

    def search(self, *, query: str, limit: int = 10) -> dict[str, Any]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,url,citationCount,publicationTypes",
        }
        headers = (
            {"x-api-key": self.settings.semantic_scholar_api_key}
            if self.settings.semantic_scholar_api_key
            else None
        )
        result = self._get_json(url, params=params, headers=headers)
        return dict(result)

    def scholar_url(self, query: str) -> str:
        return f"https://scholar.google.com/scholar?q={quote_plus(query)}"
