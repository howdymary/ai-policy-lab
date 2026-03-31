from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import httpx

from ai_policy_lab.config import Settings

logger = logging.getLogger(__name__)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_AUTH_PARAM_NAMES = {"key", "api_key", "registrationkey", "auth"}


class ConnectorConfigurationError(RuntimeError):
    """Raised when a connector is missing required credentials."""


@dataclass(slots=True)
class RateLimiter:
    key: str
    max_calls: int
    period_seconds: float

    _buckets: ClassVar[dict[str, deque[float]]] = {}
    _locks: ClassVar[dict[str, Lock]] = {}

    def acquire(self) -> None:
        bucket = self._buckets.setdefault(self.key, deque())
        lock = self._locks.setdefault(self.key, Lock())

        while True:
            wait_seconds = 0.0
            with lock:
                now = time.monotonic()
                while bucket and now - bucket[0] >= self.period_seconds:
                    bucket.popleft()
                if len(bucket) < self.max_calls:
                    bucket.append(now)
                    return
                wait_seconds = max(self.period_seconds - (now - bucket[0]), 0.01)
            logger.debug("Rate limit hit for %s; sleeping %.2fs", self.key, wait_seconds)
            time.sleep(wait_seconds)


@dataclass(slots=True)
class BaseConnector:
    settings: Settings
    rate_limit_max_calls: ClassVar[int] = 60
    rate_limit_period_seconds: ClassVar[float] = 60.0
    cache_enabled: ClassVar[bool] = True

    @property
    def timeout(self) -> float:
        return self.settings.http_timeout_seconds

    @property
    def cache_dir(self) -> Path:
        return Path.home() / ".cache" / "ai-policy-lab"

    @property
    def cache_ttl_seconds(self) -> int:
        return self.settings.cache_ttl_hours * 3600

    def _rate_limiter(self) -> RateLimiter:
        return RateLimiter(
            key=self.__class__.__name__,
            max_calls=self.rate_limit_max_calls,
            period_seconds=self.rate_limit_period_seconds,
        )

    def _get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> str:
        return self._request_text(
            "GET",
            url,
            params=params,
            headers=headers,
            auth=auth,
        )

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> Any:
        text = self._request_text(
            "GET",
            url,
            params=params,
            headers=headers,
            auth=auth,
        )
        return json.loads(text)

    def _post_json(
        self,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> Any:
        text = self._request_text(
            "POST",
            url,
            headers=headers,
            json_body=json_body,
            auth=auth,
        )
        return json.loads(text)

    def _request_text(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> str:
        cache_key = self._cache_key(
            method=method,
            url=url,
            params=params,
            json_body=json_body,
        )
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        for attempt in range(3):
            self._rate_limiter().acquire()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        json=json_body,
                        auth=auth,
                    )
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < 2:
                    delay = self._retry_delay(response.headers.get("Retry-After"), attempt)
                    logger.warning(
                        "Retryable response from %s %s (status=%s). Sleeping %.1fs before retry.",
                        method,
                        url,
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.text
                self._write_cache(cache_key, payload)
                return payload
            except httpx.TimeoutException:
                if attempt == 2:
                    raise
                delay = float(2**attempt)
                logger.warning("Timeout from %s %s. Sleeping %.1fs before retry.", method, url, delay)
                time.sleep(delay)
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in _RETRYABLE_STATUS_CODES and attempt < 2:
                    retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                    delay = self._retry_delay(retry_after, attempt)
                    logger.warning(
                        "Retryable HTTP status from %s %s (status=%s). Sleeping %.1fs before retry.",
                        method,
                        url,
                        status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"Unreachable request state for {method} {url}")

    def _cache_key(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> str:
        clean_params = self._strip_auth_fields(params)
        clean_json_body = self._strip_auth_fields(json_body)
        payload = {
            "connector": self.__class__.__name__,
            "method": method,
            "url": url,
            "params": clean_params,
            "json_body": clean_json_body,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _strip_auth_fields(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if not payload:
            return {}
        return {key: value for key, value in payload.items() if key.lower() not in _AUTH_PARAM_NAMES}

    def _retry_delay(self, retry_after: str | None, attempt: int) -> float:
        if retry_after:
            try:
                return max(float(retry_after), 0.5)
            except ValueError:
                pass
        return float(2**attempt)

    def _read_cache(self, cache_key: str) -> str | None:
        if not self.cache_enabled:
            return None
        cache_path = self.cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        cached_at = float(payload.get("cached_at_epoch", 0))
        if time.time() - cached_at > self.cache_ttl_seconds:
            return None
        body = payload.get("body")
        return body if isinstance(body, str) else None

    def _write_cache(self, cache_key: str, body: str) -> None:
        if not self.cache_enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{cache_key}.json"
        payload = {"cached_at_epoch": time.time(), "body": body}
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
