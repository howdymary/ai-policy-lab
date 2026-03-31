from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector


class WebSearchConnector(BaseConnector):
    def search(self, *, query: str, limit: int = 10) -> dict[str, Any]:
        raise NotImplementedError(
            "WebSearchConnector is an interface reserved for a future provider integration. "
            "Wire SerpAPI, Bing Search, or Google Custom Search to enable live web search."
        )
