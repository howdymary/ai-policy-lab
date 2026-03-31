from __future__ import annotations

import re
from datetime import UTC, datetime


def utcnow_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "research-run"


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
