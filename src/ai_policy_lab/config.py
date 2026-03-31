from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    use_mock: bool
    runs_dir: Path
    default_model: str
    default_temperature: float
    openai_base_url: str
    openai_api_key: str
    http_timeout_seconds: float
    bls_api_key: str | None
    fred_api_key: str | None
    census_api_key: str | None
    onet_username: str | None
    onet_password: str | None
    semantic_scholar_api_key: str | None
    web_search_api_key: str | None

    @classmethod
    def load(cls) -> Settings:
        load_dotenv()
        return cls(
            use_mock=_env_bool("APL_USE_MOCK", True),
            runs_dir=Path(os.getenv("APL_RUNS_DIR", "runs")),
            default_model=os.getenv("APL_DEFAULT_MODEL", "gpt-4o-mini"),
            default_temperature=float(os.getenv("APL_DEFAULT_TEMPERATURE", "0.2")),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            http_timeout_seconds=float(os.getenv("APL_HTTP_TIMEOUT_SECONDS", "90")),
            bls_api_key=os.getenv("BLS_API_KEY") or None,
            fred_api_key=os.getenv("FRED_API_KEY") or None,
            census_api_key=os.getenv("CENSUS_API_KEY") or None,
            onet_username=os.getenv("ONET_USERNAME") or None,
            onet_password=os.getenv("ONET_PASSWORD") or None,
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
            web_search_api_key=os.getenv("WEB_SEARCH_API_KEY") or None,
        )

    def model_for(self, agent_name: str) -> str:
        env_name = f"APL_MODEL_{agent_name.upper().replace('-', '_')}"
        return os.getenv(env_name, self.default_model)
