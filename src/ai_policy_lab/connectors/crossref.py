from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class CrossrefConnector(BaseConnector):
    base_url = "https://api.crossref.org/works"
    rate_limit_max_calls = 50
    rate_limit_period_seconds = 60.0

    def works(self, *, query_title: str, rows: int = 5) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query.title": query_title,
            "rows": rows,
        }
        if self.settings.crossref_contact_email:
            params["mailto"] = self.settings.crossref_contact_email
            headers = {
                "User-Agent": f"ai-policy-lab/0.1.0 (mailto:{self.settings.crossref_contact_email})"
            }
        else:
            headers = {"User-Agent": "ai-policy-lab/0.1.0"}
        result = self._get_json(self.base_url, params=params, headers=headers)
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise ConnectorConfigurationError("Crossref API error: unexpected response status")
        return dict(result)
