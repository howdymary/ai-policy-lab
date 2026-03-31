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
    def __init__(self, body):
        self._body = body
        self.text = json.dumps(body)

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, response_body):
        self._response_body = response_body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)

    def post(self, url, json, headers):  # noqa: A002
        _ = (url, json, headers)
        return _FakeResponse(self._response_body)


def test_llm_generate_validates_missing_choices(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient({"unexpected": []}))
    llm = OpenAICompatibleLLM(settings=_settings())

    with pytest.raises(LLMResponseError, match="no choices"):
        llm.generate(
            agent_name="research_director",
            system_prompt="system",
            user_prompt="user",
        )
