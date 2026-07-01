import json

import pandas as pd
import pytest

from feedback_theme_engine.llm_labeling import (
    ThemeLabelingConfig,
    build_theme_labeling_prompt,
    label_theme,
    label_theme_with_mock,
    parse_theme_labeling_response,
)


def _theme_row() -> dict[str, object]:
    return {
        "cluster_id": 2,
        "theme_count": 12,
        "total_count": 100,
        "prevalence": 0.12,
        "prevalence_ci_lower": 0.07,
        "prevalence_ci_upper": 0.20,
        "rating_gap": -1.2,
        "p_value_adjusted": 0.03,
        "interpretation": "higher_risk_theme",
        "insufficient_sample": False,
    }


def _representatives() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [2, 2],
            "review_id": ["r1", "r2"],
            "review_text": ["The bottle leaked in shipping.", "Packaging arrived broken."],
        }
    )


def _keywords() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [2, 2],
            "keyword": ["leaked", "packaging"],
            "score": [0.8, 0.7],
        }
    )


def test_prompt_includes_constraints_and_statistical_fields() -> None:
    prompt = build_theme_labeling_prompt(
        _theme_row(),
        _representatives(),
        _keywords(),
        ThemeLabelingConfig(),
    )

    assert "Do not invent facts" in prompt
    assert "Use only the supplied statistics" in prompt
    assert "Do not claim causality" in prompt
    assert "valid JSON" in prompt
    assert "rating_gap" in prompt
    assert "p_value_adjusted" in prompt
    assert "prevalence_ci_lower" in prompt


def test_parse_theme_labeling_response_accepts_valid_json() -> None:
    response = {
        "cluster_id": 2,
        "theme_label": "Packaging Leakage",
        "theme_type": "customer_pain_point",
        "summary": "Customers mention leaked or broken packaging.",
        "evidence": ["The bottle leaked in shipping."],
        "suggested_action": "Review packaging examples.",
        "confidence": "medium",
        "caution": "Exploratory label.",
    }

    parsed = parse_theme_labeling_response(json.dumps(response))

    assert parsed == response


def test_parse_theme_labeling_response_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_theme_labeling_response("{not json")


def test_parse_theme_labeling_response_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError, match="theme_label"):
        parse_theme_labeling_response(json.dumps({"cluster_id": 1}))


def test_label_theme_with_mock_is_deterministic_and_grounded() -> None:
    first = label_theme_with_mock(
        _theme_row(),
        _representatives(),
        _keywords(),
        ThemeLabelingConfig(),
    )
    second = label_theme_with_mock(
        _theme_row(),
        _representatives(),
        _keywords(),
        ThemeLabelingConfig(),
    )

    assert first == second
    assert first["cluster_id"] == 2
    assert first["theme_type"] == "customer_pain_point"
    assert "Leaked" in first["theme_label"]
    assert first["evidence"] == [
        "The bottle leaked in shipping.",
        "Packaging arrived broken.",
    ]


def test_label_theme_non_mock_provider_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="environment variables"):
        label_theme(
            _theme_row(),
            _representatives(),
            _keywords(),
            ThemeLabelingConfig(provider="openai"),
        )
