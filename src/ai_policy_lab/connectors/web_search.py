from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector


class WebSearchConnector(BaseConnector):
    def search(self, *, query: str, limit: int = 10) -> dict[str, Any]:
        raise NotImplementedError(
            "WebSearchConnector is a placeholder. Wire a provider (SerpAPI, Bing Search, Google Custom Search) to enable live web search."
        )
