from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_policy_lab.config import Settings
from ai_policy_lab.connectors import (
    BLSConnector,
    CensusConnector,
    CrossrefConnector,
    FederalRegisterConnector,
    FREDConnector,
    ScholarSearchConnector,
    WebSearchConnector,
)
from ai_policy_lab.connectors import base as base_module
from ai_policy_lab.connectors.base import BaseConnector, ConnectorConfigurationError


def _settings() -> Settings:
    return Settings(
        use_mock=True,
        runs_dir=Path("runs"),
        cache_ttl_hours=24,
        default_model="test-model",
        default_temperature=0.2,
        openai_base_url="http://localhost:11434/v1",
        openai_api_key="ollama",
        http_timeout_seconds=30.0,
        bls_api_key="bls-key",
        fred_api_key="fred-key",
        census_api_key="census-key",
        onet_username="onet-user",
        onet_password="onet-pass",
        semantic_scholar_api_key="scholar-key",
        web_search_api_key=None,
        crossref_contact_email="test@example.com",
    )


class _ScriptedClient:
    def __init__(self, events: list[Any]):
        self.events = list(events)
        self.requests: list[dict[str, Any]] = []

    def __enter__(self) -> _ScriptedClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,  # noqa: A002
        auth: tuple[str, str] | None = None,
    ) -> httpx.Response:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "headers": headers,
                "json": json,
                "auth": auth,
            }
        )
        if not self.events:
            raise AssertionError("No scripted HTTP events remaining.")
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        request = httpx.Request(method, url)
        return httpx.Response(
            status_code=event["status_code"],
            request=request,
            headers=event.get("headers"),
            content=event.get("content", b""),
        )


class _DummyConnector(BaseConnector):
    def fetch_json(self, *, params: dict[str, Any]) -> Any:
        return self._get_json("https://example.com/data", params=params)

    @property
    def cache_dir(self) -> Path:
        return Path.home() / ".cache" / "ai-policy-lab-test"


def _patch_http_client(monkeypatch, client: _ScriptedClient) -> None:
    monkeypatch.setattr(base_module.httpx, "Client", lambda timeout: client)


class _NoCacheBLSConnector(BLSConnector):
    cache_enabled = False


def test_bls_happy_path_and_rate_limit(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps({"status": "REQUEST_SUCCEEDED", "Results": {}}).encode(),
            }
        ]
    )
    _patch_http_client(monkeypatch, client)
    connector = _NoCacheBLSConnector(_settings())

    result = connector.timeseries(series_ids=["LNS14000000"], start_year=2024, end_year=2024)

    assert connector.rate_limit_max_calls == 8
    assert result["status"] == "REQUEST_SUCCEEDED"
    assert client.requests[0]["json"]["registrationkey"] == "bls-key"


def test_bls_validation_rejects_error_payload(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps({"status": "REQUEST_FAILED", "message": "bad key"}).encode(),
            }
        ]
    )
    _patch_http_client(monkeypatch, client)
    connector = _NoCacheBLSConnector(_settings())

    with pytest.raises(ConnectorConfigurationError, match="BLS API error"):
        connector.timeseries(series_ids=["LNS14000000"], start_year=2024, end_year=2024)


def test_census_uses_header_and_validates_error_payload(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps([["NAME"], ["California"]]).encode(),
            },
            {
                "status_code": 200,
                "content": json.dumps({"error": "invalid geography"}).encode(),
            },
        ]
    )
    _patch_http_client(monkeypatch, client)
    class NoCacheCensusConnector(CensusConnector):
        cache_enabled = False

    connector = NoCacheCensusConnector(_settings())

    result = connector.dataset(
        dataset="acs/acs5",
        year=2022,
        variables=["NAME"],
        for_clause="state:*",
    )
    assert result[1][0] == "California"
    assert client.requests[0]["headers"]["X-Census-Key"] == "census-key"
    assert "key" not in (client.requests[0]["params"] or {})

    with pytest.raises(ConnectorConfigurationError, match="Census API error"):
        connector.dataset(
            dataset="acs/acs5",
            year=2022,
            variables=["NAME"],
            for_clause="state:*",
        )


def test_fred_validation_and_missing_key(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps({"observations": []}).encode(),
            },
            {
                "status_code": 200,
                "content": json.dumps({"error_code": 400, "error_message": "bad series"}).encode(),
            },
        ]
    )
    _patch_http_client(monkeypatch, client)
    class NoCacheFREDConnector(FREDConnector):
        cache_enabled = False

    connector = NoCacheFREDConnector(_settings())

    result = connector.series_observations(series_id="UNRATE")
    assert result["observations"] == []
    assert client.requests[0]["params"]["api_key"] == "fred-key"

    with pytest.raises(ConnectorConfigurationError, match="FRED API error"):
        connector.series_observations(series_id="UNRATE")

    missing = _settings()
    missing.fred_api_key = None
    with pytest.raises(ConnectorConfigurationError, match="FRED_API_KEY"):
        FREDConnector(missing).series_observations(series_id="UNRATE")


def test_crossref_validation(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps({"status": "ok", "message": {"items": []}}).encode(),
            },
            {
                "status_code": 200,
                "content": json.dumps({"status": "error"}).encode(),
            },
        ]
    )
    _patch_http_client(monkeypatch, client)
    class NoCacheCrossrefConnector(CrossrefConnector):
        cache_enabled = False

    connector = NoCacheCrossrefConnector(_settings())

    result = connector.works(query_title="Automation and New Tasks")
    assert result["status"] == "ok"

    with pytest.raises(ConnectorConfigurationError, match="Crossref API error"):
        connector.works(query_title="Automation and New Tasks")


def test_scholar_validation(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps({"data": [], "total": 0}).encode(),
            },
            {
                "status_code": 200,
                "content": json.dumps({"error": "rate limited"}).encode(),
            },
        ]
    )
    _patch_http_client(monkeypatch, client)
    class NoCacheScholarConnector(ScholarSearchConnector):
        cache_enabled = False

    connector = NoCacheScholarConnector(_settings())

    result = connector.search(query="AI automation labor market")
    assert result["total"] == 0

    with pytest.raises(ConnectorConfigurationError, match="Semantic Scholar API error"):
        connector.search(query="AI automation labor market")


def test_federal_register_validation(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps({"results": []}).encode(),
            },
            {
                "status_code": 200,
                "content": json.dumps({"message": "error"}).encode(),
            },
        ]
    )
    _patch_http_client(monkeypatch, client)
    class NoCacheFederalRegisterConnector(FederalRegisterConnector):
        cache_enabled = False

    connector = NoCacheFederalRegisterConnector(_settings())

    result = connector.search_documents(term="artificial intelligence")
    assert result["results"] == []

    with pytest.raises(ConnectorConfigurationError, match="Federal Register API error"):
        connector.search_documents(term="artificial intelligence")


def test_retry_after_and_timeout(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "3"},
                "content": b"",
            },
            {
                "status_code": 200,
                "content": json.dumps({"status": "REQUEST_SUCCEEDED", "Results": {}}).encode(),
            },
        ]
    )
    _patch_http_client(monkeypatch, client)
    sleeps: list[float] = []
    monkeypatch.setattr(base_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    connector = _NoCacheBLSConnector(_settings())

    result = connector.timeseries(series_ids=["LNS14000000"], start_year=2024, end_year=2024)

    assert result["status"] == "REQUEST_SUCCEEDED"
    assert sleeps == [3.0]
    assert len(client.requests) == 2


def test_timeout_retry_then_success(monkeypatch) -> None:
    request = httpx.Request("GET", "https://example.com")
    timeout = httpx.ReadTimeout("timed out", request=request)
    client = _ScriptedClient(
        [
            timeout,
            timeout,
            {
                "status_code": 200,
                "content": json.dumps({"status": "REQUEST_SUCCEEDED", "Results": {}}).encode(),
            },
        ]
    )
    _patch_http_client(monkeypatch, client)
    sleeps: list[float] = []
    monkeypatch.setattr(base_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    connector = _NoCacheBLSConnector(_settings())

    result = connector.timeseries(series_ids=["LNS14000000"], start_year=2024, end_year=2024)

    assert result["status"] == "REQUEST_SUCCEEDED"
    assert sleeps == [1.0, 2.0]


def test_invalid_json_propagates(monkeypatch) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": b"not-json",
            }
        ]
    )
    _patch_http_client(monkeypatch, client)
    connector = _NoCacheBLSConnector(_settings())

    with pytest.raises(json.JSONDecodeError):
        connector.timeseries(series_ids=["LNS14000000"], start_year=2024, end_year=2024)


def test_cache_hit_strips_auth_params(monkeypatch, tmp_path: Path) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps({"answer": 1}).encode(),
            }
        ]
    )
    _patch_http_client(monkeypatch, client)

    class CachedDummyConnector(_DummyConnector):
        @property
        def cache_dir(self) -> Path:
            return tmp_path

    connector = CachedDummyConnector(_settings())
    first = connector.fetch_json(params={"query": "alpha", "api_key": "secret-a"})
    second = connector.fetch_json(params={"query": "alpha", "api_key": "secret-b"})

    assert first == second == {"answer": 1}
    assert len(client.requests) == 1


def test_cache_expiry_triggers_refresh(monkeypatch, tmp_path: Path) -> None:
    client = _ScriptedClient(
        [
            {
                "status_code": 200,
                "content": json.dumps({"answer": 1}).encode(),
            },
            {
                "status_code": 200,
                "content": json.dumps({"answer": 2}).encode(),
            },
        ]
    )
    _patch_http_client(monkeypatch, client)

    class CachedDummyConnector(_DummyConnector):
        @property
        def cache_dir(self) -> Path:
            return tmp_path

    connector = CachedDummyConnector(_settings())
    params = {"query": "alpha", "api_key": "secret"}
    cache_key = connector._cache_key(
        method="GET",
        url="https://example.com/data",
        params=params,
        json_body=None,
    )
    connector.cache_dir.mkdir(parents=True, exist_ok=True)
    connector.cache_dir.joinpath(f"{cache_key}.json").write_text(
        json.dumps({"cached_at_epoch": time.time() - (connector.cache_ttl_seconds + 10), "body": json.dumps({"answer": 0})}),
        encoding="utf-8",
    )

    result = connector.fetch_json(params=params)

    assert result == {"answer": 1}
    assert len(client.requests) == 1


def test_web_search_placeholder() -> None:
    connector = WebSearchConnector(_settings())

    with pytest.raises(NotImplementedError, match="placeholder"):
        connector.search(query="AI policy")
