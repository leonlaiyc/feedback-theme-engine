import pytest

from feedback_theme_engine.text_cleaning import basic_clean_text, normalize_whitespace


def test_normalize_whitespace_collapses_repeated_whitespace() -> None:
    text = "  Great\n\nbattery\tlife.   Still works.  "

    assert normalize_whitespace(text) == "Great battery life. Still works."


def test_basic_clean_text_is_conservative() -> None:
    text = "  Works well!!!  Would buy again.  "

    assert basic_clean_text(text) == "Works well!!! Would buy again."


def test_normalize_whitespace_raises_for_non_string_input() -> None:
    with pytest.raises(TypeError, match="string"):
        normalize_whitespace(None)  # type: ignore[arg-type]

