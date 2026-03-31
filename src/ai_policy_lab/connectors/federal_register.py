from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector


class FederalRegisterConnector(BaseConnector):
    base_url = "https://www.federalregister.gov/api/v1/documents.json"

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
        return dict(result)
