from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


class ONETConnector(BaseConnector):
    base_url = "https://services.onetcenter.org/ws"
    public_database_url = "https://www.onetcenter.org/database.html"
    public_download_base_url = "https://www.onetcenter.org/dl_files/database"

    def occupations(self) -> dict[str, Any]:
        if not self.settings.onet_username or not self.settings.onet_password:
            raise ConnectorConfigurationError(
                "ONET_USERNAME and ONET_PASSWORD are required for O*NET web service access."
            )

        return self._get_json(
            f"{self.base_url}/online/occupations",
            params=None,
        )

    def latest_text_release(self) -> str:
        html = self._get_text(self.public_database_url)
        match = re.search(r"dl_files/database/(db_[0-9_]+_text)/", html)
        if match is None:
            raise RuntimeError("Could not determine the latest public O*NET text release.")
        return match.group(1)

    def public_text_file(self, *, filename: str) -> str:
        release = self.latest_text_release()
        return self._get_text(
            f"{self.public_download_base_url}/{release}/{quote(filename)}",
        )
