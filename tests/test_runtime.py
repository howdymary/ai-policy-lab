from __future__ import annotations

from pathlib import Path

import pytest

from ai_policy_lab import runtime as runtime_module
from ai_policy_lab.config import Settings
from ai_policy_lab.llm import LLMNotConfiguredError, LLMResponseError
from ai_policy_lab.runtime import ResearchRuntime


class _FakeLLM:
    def __init__(self, *, available: bool = True, raise_error: Exception | None = None) -> None:
        self.available = available
        self.raise_error = raise_error
        self.calls: list[dict[str, object]] = []

    def is_available(self) -> bool:
        return self.available

    def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if self.raise_error is not None:
            raise self.raise_error
        return "live response"


def _settings(*, use_mock: bool, openai_api_key: str = "ollama") -> Settings:
    return Settings(
        use_mock=use_mock,
        runs_dir=Path("runs"),
        cache_ttl_hours=24,
        default_model="test-model",
        default_temperature=0.2,
        openai_base_url="http://localhost:11434/v1",
        openai_api_key=openai_api_key,
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


def test_use_live_llm_false_in_mock_mode() -> None:
    runtime = ResearchRuntime(settings=_settings(use_mock=True), llm=_FakeLLM())
    assert runtime.use_live_llm() is False


def test_use_live_llm_false_when_llm_is_unavailable() -> None:
    runtime = ResearchRuntime(settings=_settings(use_mock=False), llm=_FakeLLM(available=False))
    assert runtime.use_live_llm() is False


def test_maybe_generate_returns_fallback_in_mock_mode() -> None:
    fake_llm = _FakeLLM()
    runtime = ResearchRuntime(settings=_settings(use_mock=True), llm=fake_llm)

    assert runtime.maybe_generate(
        agent_name="research_director",
        system_prompt="system",
        user_prompt="user",
        fallback="fallback text",
    ) == "fallback text"
    assert fake_llm.calls == []


def test_maybe_generate_returns_live_output_when_available() -> None:
    fake_llm = _FakeLLM()
    runtime = ResearchRuntime(settings=_settings(use_mock=False), llm=fake_llm)

    assert runtime.maybe_generate(
        agent_name="research_director",
        system_prompt="system",
        user_prompt="user",
        fallback="fallback text",
    ) == "live response"
    assert fake_llm.calls


def test_maybe_generate_raises_on_llm_not_configured_error() -> None:
    fake_llm = _FakeLLM(raise_error=LLMNotConfiguredError("missing key"))
    runtime = ResearchRuntime(settings=_settings(use_mock=False), llm=fake_llm)

    with pytest.raises(LLMNotConfiguredError, match="missing key"):
        runtime.maybe_generate(
            agent_name="research_director",
            system_prompt="system",
            user_prompt="user",
            fallback="fallback text",
        )


def test_maybe_generate_raises_on_llm_response_error() -> None:
    fake_llm = _FakeLLM(raise_error=LLMResponseError("bad response"))
    runtime = ResearchRuntime(settings=_settings(use_mock=False), llm=fake_llm)

    with pytest.raises(LLMResponseError, match="bad response"):
        runtime.maybe_generate(
            agent_name="research_director",
            system_prompt="system",
            user_prompt="user",
            fallback="fallback text",
        )


def test_from_env_builds_runtime(monkeypatch) -> None:
    monkeypatch.setenv("APL_USE_MOCK", "true")
    monkeypatch.setenv("APL_DEFAULT_MODEL", "runtime-test-model")
    runtime = runtime_module.ResearchRuntime.from_env()

    assert runtime.settings.use_mock is True
    assert runtime.settings.default_model == "runtime-test-model"
