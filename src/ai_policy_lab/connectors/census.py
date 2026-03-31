from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector


class CensusConnector(BaseConnector):
    def dataset(
        self,
        *,
        dataset: str,
        year: int,
        variables: list[str],
        for_clause: str,
        in_clause: str | None = None,
        predicates: dict[str, str] | None = None,
    ) -> Any:
        params: dict[str, Any] = {
            "get": ",".join(variables),
            "for": for_clause,
        }
        if in_clause:
            params["in"] = in_clause
        if predicates:
            params.update(predicates)
        if self.settings.census_api_key:
            params["key"] = self.settings.census_api_key

        return self._get_json(f"https://api.census.gov/data/{year}/{dataset}", params=params)
