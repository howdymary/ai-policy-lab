from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class FederalRegisterConnector(BaseConnector):
    base_url = "https://www.federalregister.gov/api/v1/documents.json"
    rate_limit_max_calls = 100
    rate_limit_period_seconds = 60.0

    def search_documents(
        self,
        *,
        term: str,
        agency_slug: str | None = None,
        per_page: int = 5,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "conditions[term]": term,
            "per_page": str(per_page),
            "order": "newest",
        }
        if agency_slug:
            params["conditions[agencies][]"] = agency_slug
        result = self._get_json(self.base_url, params=params)
        if not isinstance(result, dict) or "results" not in result:
            raise ConnectorConfigurationError("Federal Register API error: unexpected response payload")
        return dict(result)
