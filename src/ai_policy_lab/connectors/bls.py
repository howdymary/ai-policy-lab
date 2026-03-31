from __future__ import annotations

from typing import Any

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class BLSConnector(BaseConnector):
    base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    # BLS v2 API: 500 req/day (registered), 25 req/day (unregistered)
    rate_limit_max_calls = 8
    rate_limit_period_seconds = 60.0

    def timeseries(
        self,
        *,
        series_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self.settings.bls_api_key:
            payload["registrationkey"] = self.settings.bls_api_key

        result = self._post_json(self.base_url, json_body=payload)
        if not isinstance(result, dict) or result.get("status") != "REQUEST_SUCCEEDED":
            message = "unknown"
            if isinstance(result, dict):
                message = str(result.get("message", result.get("responseError", "unknown")))
            raise ConnectorConfigurationError(f"BLS API error: {message}")
        return dict(result)
