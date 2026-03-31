from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

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
    cache_ttl_hours: int
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
    crossref_contact_email: str | None

    @classmethod
    def load(cls) -> Settings:
        load_dotenv()
        settings = cls(
            use_mock=_env_bool("APL_USE_MOCK", False),
            runs_dir=Path(os.getenv("APL_RUNS_DIR", "runs")),
            cache_ttl_hours=int(os.getenv("APL_CACHE_TTL_HOURS", "24")),
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
            crossref_contact_email=os.getenv("CROSSREF_CONTACT_EMAIL") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.use_mock and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when APL_USE_MOCK is false, even for local OpenAI-compatible endpoints."
            )
        if self.http_timeout_seconds <= 0:
            raise ValueError("APL_HTTP_TIMEOUT_SECONDS must be greater than zero.")
        if not 0 <= self.default_temperature <= 2:
            raise ValueError("APL_DEFAULT_TEMPERATURE must be between 0 and 2.")
        if self.cache_ttl_hours <= 0:
            raise ValueError("APL_CACHE_TTL_HOURS must be greater than zero.")
        parsed = urlparse(self.openai_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OPENAI_BASE_URL must be a valid http or https URL.")

    def model_for(self, agent_name: str) -> str:
        env_name = f"APL_MODEL_{agent_name.upper().replace('-', '_')}"
        return os.getenv(env_name, self.default_model)
