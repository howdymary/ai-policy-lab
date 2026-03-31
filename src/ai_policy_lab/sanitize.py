from __future__ import annotations

import re
import unicodedata

from ai_policy_lab.utils import compact_whitespace

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_ROLE_MARKER_RE = re.compile(
    r"(?i)(?:\b(?:system|assistant|developer|tool|user)\s*:)"
    r"|(?:[\[<]\s*/?\s*(?:system|assistant|developer|tool|user)\s*(?:role\s*=\s*[^>\]]+)?\s*[\]>]?\s*:?)"
    r"|(?:</?(?:system|assistant|developer|tool|user)(?:\s+role\s*=\s*[^>]+)?\s*>)"
)
_PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?earlier\s+instructions",
        r"disregard\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+[^.:\n]+",
        r"follow\s+these\s+new\s+instructions",
        r"execute\s+these\s+new\s+instructions",
        r"follow\s+new\s+rules",
        r"pretend\s+to\s+be\s+[^.:\n]+",
        r"jailbreak",
    )
]
_MAX_USER_INPUT_LENGTH = 2000


def sanitize_user_input(text: str, *, max_length: int = _MAX_USER_INPUT_LENGTH) -> str:
    sanitized = unicodedata.normalize("NFKC", text)
    sanitized = _CONTROL_CHAR_RE.sub(" ", sanitized)
    sanitized = _ROLE_MARKER_RE.sub("[role marker removed]:", sanitized)
    for pattern in _PROMPT_INJECTION_PATTERNS:
        sanitized = pattern.sub("[filtered instruction]", sanitized)
    sanitized = compact_whitespace(sanitized)
    if len(sanitized) > max_length:
        sanitized = sanitized[: max_length - 3].rstrip() + "..."
    return sanitized


def sanitize_user_inputs(values: list[str]) -> list[str]:
    return [sanitize_user_input(value) for value in values]


def wrap_user_content(tag: str, text: str) -> str:
    sanitized = sanitize_user_input(text)
    return f"<{tag}>\n{sanitized}\n</{tag}>"


def wrap_user_list(tag: str, values: list[str], *, item_tag: str = "item") -> str:
    if not values:
        return f"<{tag}></{tag}>"
    rendered = "\n".join(f"<{item_tag}>{sanitize_user_input(value)}</{item_tag}>" for value in values)
    return f"<{tag}>\n{rendered}\n</{tag}>"
