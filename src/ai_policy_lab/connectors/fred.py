from __future__ import annotations

from typing import Any, cast

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class FREDConnector(BaseConnector):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    rate_limit_max_calls = 100
    rate_limit_period_seconds = 120.0

    def series_observations(
        self,
        *,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> dict[str, Any]:
        if not self.settings.fred_api_key:
            raise ConnectorConfigurationError("FRED_API_KEY is required for FRED requests.")

        params: dict[str, Any] = {
            "series_id": series_id,
            # FRED requires the API key in the query string for this endpoint.
            "api_key": self.settings.fred_api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end

        result = self._get_json(self.base_url, params=params)
        if not isinstance(result, dict) or "error_code" in result:
            raise ConnectorConfigurationError(f"FRED API error: {result.get('error_message', 'unknown') if isinstance(result, dict) else 'unknown'}")
        return dict(cast(dict[str, Any], result))
