from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector


class CrossrefConnector(BaseConnector):
    base_url = "https://api.crossref.org/works"

    def works(self, *, query_title: str, rows: int = 5) -> dict[str, Any]:
        params = {
            "query.title": query_title,
            "rows": rows,
            "mailto": "mfl2147@columbia.edu",
        }
        headers = {"User-Agent": "ai-policy-lab/0.1.0 (mailto:mfl2147@columbia.edu)"}
        result = self._get_json(self.base_url, params=params, headers=headers)
        return dict(result)
