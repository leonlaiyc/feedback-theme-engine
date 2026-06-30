"""Conservative text cleaning helpers for customer feedback."""

from __future__ import annotations

import re

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim leading or trailing whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def basic_clean_text(text: str) -> str:
    """Apply minimal cleaning while preserving customer wording and punctuation."""
    return normalize_whitespace(text)

