from __future__ import annotations

from typing import Any

import httpx

from ai_policy_lab.connectors.base import BaseConnector


class BLSConnector(BaseConnector):
    base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

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

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url, json=payload)
            response.raise_for_status()
            return dict(response.json())
