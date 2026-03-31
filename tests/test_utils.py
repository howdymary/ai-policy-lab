from __future__ import annotations

from datetime import UTC, datetime

from ai_policy_lab.utils import compact_whitespace, slugify, utcnow_iso


def test_slugify_normalizes_text() -> None:
    assert slugify("  AI Policy Lab: Great Reallocation!  ") == "ai-policy-lab-great-reallocation"


def test_slugify_falls_back_for_empty_input() -> None:
    assert slugify("   ") == "research-run"


def test_compact_whitespace_collapses_runs() -> None:
    assert compact_whitespace("alpha\n\n beta\t\tgamma") == "alpha beta gamma"


def test_utcnow_iso_returns_utc_timestamp() -> None:
    parsed = datetime.fromisoformat(utcnow_iso())
    assert parsed.tzinfo == UTC
