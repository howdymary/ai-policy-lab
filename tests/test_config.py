from __future__ import annotations

from pathlib import Path

import pytest

from ai_policy_lab.config import Settings


def test_settings_validate_fails_fast_without_llm_key() -> None:
    settings = Settings(
        use_mock=False,
        runs_dir=Path("runs"),
        cache_ttl_hours=24,
        default_model="test-model",
        default_temperature=0.2,
        openai_base_url="http://localhost:11434/v1",
        openai_api_key="",
        http_timeout_seconds=30.0,
        bls_api_key=None,
        fred_api_key=None,
        census_api_key=None,
        onet_username=None,
        onet_password=None,
        semantic_scholar_api_key=None,
        web_search_api_key=None,
        crossref_contact_email=None,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.validate()


def test_settings_validate_openai_base_url() -> None:
    settings = Settings(
        use_mock=True,
        runs_dir=Path("runs"),
        cache_ttl_hours=24,
        default_model="test-model",
        default_temperature=0.2,
        openai_base_url="not-a-url",
        openai_api_key="ollama",
        http_timeout_seconds=30.0,
        bls_api_key=None,
        fred_api_key=None,
        census_api_key=None,
        onet_username=None,
        onet_password=None,
        semantic_scholar_api_key=None,
        web_search_api_key=None,
        crossref_contact_email=None,
    )

    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        settings.validate()
