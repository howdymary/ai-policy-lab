from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class CensusConnector(BaseConnector):
    rate_limit_max_calls = 50
    rate_limit_period_seconds = 60.0

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
            headers = {"X-Census-Key": self.settings.census_api_key}
        else:
            headers = None

        result = self._get_json(
            f"https://api.census.gov/data/{year}/{dataset}",
            params=params,
            headers=headers,
        )
        if isinstance(result, dict) and "error" in result:
            raise ConnectorConfigurationError(f"Census API error: {result['error']}")
        return result
