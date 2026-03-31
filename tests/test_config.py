from __future__ import annotations

from pathlib import Path

import pytest

from ai_policy_lab.config import Settings


def _settings(**overrides) -> Settings:
    data = {
        "use_mock": True,
        "runs_dir": Path("runs"),
        "cache_ttl_hours": 24,
        "default_model": "test-model",
        "default_temperature": 0.2,
        "openai_base_url": "http://localhost:11434/v1",
        "openai_api_key": "ollama",
        "http_timeout_seconds": 30.0,
        "bls_api_key": None,
        "fred_api_key": None,
        "census_api_key": None,
        "onet_username": None,
        "onet_password": None,
        "semantic_scholar_api_key": None,
        "web_search_api_key": None,
        "crossref_contact_email": None,
    }
    data.update(overrides)
    return Settings(**data)


def test_settings_validate_fails_fast_without_llm_key() -> None:
    settings = _settings(use_mock=False, openai_api_key="")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.validate()


def test_settings_validate_openai_base_url() -> None:
    settings = _settings(openai_base_url="not-a-url")

    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        settings.validate()


@pytest.mark.parametrize("temperature", [0.0, 2.0])
def test_settings_validate_accepts_temperature_boundaries(temperature: float) -> None:
    _settings(default_temperature=temperature).validate()


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_settings_validate_rejects_temperature_outside_bounds(temperature: float) -> None:
    with pytest.raises(ValueError, match="APL_DEFAULT_TEMPERATURE"):
        _settings(default_temperature=temperature).validate()


@pytest.mark.parametrize("timeout", [0.001, 1.0])
def test_settings_validate_accepts_positive_timeout(timeout: float) -> None:
    _settings(http_timeout_seconds=timeout).validate()


def test_settings_validate_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="APL_HTTP_TIMEOUT_SECONDS"):
        _settings(http_timeout_seconds=0.0).validate()


def test_settings_validate_rejects_nonpositive_cache_ttl() -> None:
    with pytest.raises(ValueError, match="APL_CACHE_TTL_HOURS"):
        _settings(cache_ttl_hours=0).validate()


def test_settings_validate_happy_path() -> None:
    _settings().validate()


def test_settings_load_defaults_to_live_mode(monkeypatch) -> None:
    monkeypatch.delenv("APL_USE_MOCK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    assert Settings.load().use_mock is False
