from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ai_policy_lab.config import Settings
from ai_policy_lab.llm import LLMResponseError, OpenAICompatibleLLM


def _settings() -> Settings:
    return Settings(
        use_mock=False,
        runs_dir=Path("runs"),
        cache_ttl_hours=24,
        default_model="test-model",
        default_temperature=0.2,
        openai_base_url="http://localhost:11434/v1",
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


class _FakeResponse:
    def __init__(self, body: object, *, exc: Exception | None = None, text: str | None = None):
        self._body = body
        self._exc = exc
        self.text = text if text is not None else json.dumps(body)

    def raise_for_status(self) -> None:
        if self._exc is not None:
            raise self._exc

    def json(self) -> object:
        return self._body


class _FakeClient:
    def __init__(self, response: object):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)

    def post(self, url, json, headers):  # noqa: A002
        _ = (url, json, headers)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _patch_client(monkeypatch, response: object) -> None:
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(response))


def _http_error(response: _FakeResponse) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_llm_generate_returns_content(monkeypatch) -> None:
    _patch_client(monkeypatch, _FakeResponse({"choices": [{"message": {"content": " answer "}}]}))
    llm = OpenAICompatibleLLM(settings=_settings())

    assert llm.generate(agent_name="research_director", system_prompt="system", user_prompt="user") == "answer"


def test_llm_generate_validates_missing_choices(monkeypatch) -> None:
    _patch_client(monkeypatch, _FakeResponse({"unexpected": []}))
    llm = OpenAICompatibleLLM(settings=_settings())

    with pytest.raises(LLMResponseError, match="no choices"):
        llm.generate(agent_name="research_director", system_prompt="system", user_prompt="user")


def test_llm_generate_validates_empty_choices_array(monkeypatch) -> None:
    _patch_client(monkeypatch, _FakeResponse({"choices": []}))
    llm = OpenAICompatibleLLM(settings=_settings())

    with pytest.raises(LLMResponseError, match="no choices"):
        llm.generate(agent_name="research_director", system_prompt="system", user_prompt="user")


def test_llm_generate_validates_missing_content(monkeypatch) -> None:
    _patch_client(monkeypatch, _FakeResponse({"choices": [{"message": {}}]}))
    llm = OpenAICompatibleLLM(settings=_settings())

    with pytest.raises(LLMResponseError, match="missing content"):
        llm.generate(agent_name="research_director", system_prompt="system", user_prompt="user")


def test_llm_generate_wraps_http_errors(monkeypatch) -> None:
    response = _FakeResponse({"error": "failure"})
    response._exc = _http_error(response)  # type: ignore[attr-defined]
    _patch_client(monkeypatch, response)
    llm = OpenAICompatibleLLM(settings=_settings())

    with pytest.raises(LLMResponseError, match="HTTP status error"):
        llm.generate(agent_name="research_director", system_prompt="system", user_prompt="user")


def test_llm_generate_wraps_timeouts(monkeypatch) -> None:
    _patch_client(monkeypatch, httpx.TimeoutException("timeout"))
    llm = OpenAICompatibleLLM(settings=_settings())

    with pytest.raises(LLMResponseError, match="timed out"):
        llm.generate(agent_name="research_director", system_prompt="system", user_prompt="user")


def test_llm_generate_wraps_invalid_json(monkeypatch) -> None:
    class _BadResponse(_FakeResponse):
        def json(self) -> object:
            raise json.JSONDecodeError("bad json", doc="", pos=0)

    _patch_client(monkeypatch, _BadResponse({}, text="not json"))
    llm = OpenAICompatibleLLM(settings=_settings())

    with pytest.raises(LLMResponseError, match="invalid JSON"):
        llm.generate(agent_name="research_director", system_prompt="system", user_prompt="user")
