"""LLM-assisted theme labeling helpers with an offline mock provider."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


REQUIRED_LABEL_FIELDS: tuple[str, ...] = (
    "cluster_id",
    "theme_label",
    "theme_type",
    "summary",
    "evidence",
    "suggested_action",
    "confidence",
    "caution",
)


@dataclass(frozen=True)
class ThemeLabelingConfig:
    """Configuration for evidence-grounded theme labeling."""

    provider: str = "mock"
    model_name: str | None = None
    max_representative_reviews: int = 5
    max_keywords: int = 10
    require_evidence: bool = True


def build_theme_labeling_prompt(
    theme_row: Mapping[str, Any] | pd.Series,
    representative_reviews: pd.DataFrame | Sequence[Mapping[str, Any]],
    keywords: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[str],
    config: ThemeLabelingConfig,
) -> str:
    """Build a structured, evidence-bounded prompt for one theme."""
    theme = _mapping_from_row(theme_row)
    examples = _representative_review_records(
        representative_reviews,
        max_reviews=config.max_representative_reviews,
    )
    keyword_records = _keyword_records(keywords, max_keywords=config.max_keywords)
    prompt_payload = {
        "theme_statistics": _selected_theme_fields(theme),
        "keywords": keyword_records,
        "representative_reviews": examples,
        "required_json_fields": list(REQUIRED_LABEL_FIELDS),
    }

    return (
        "You are labeling one discovered customer feedback theme.\n"
        "Use only the supplied statistics, keywords, and representative reviews.\n"
        "Do not invent facts, reviews, products, causes, companies, or outcomes.\n"
        "Do not claim causality or confirmatory proof.\n"
        "If evidence is weak or sample size is insufficient, say so clearly.\n"
        "Return valid JSON only with exactly these fields: "
        f"{', '.join(REQUIRED_LABEL_FIELDS)}.\n\n"
        f"Input evidence:\n{json.dumps(prompt_payload, indent=2, sort_keys=True)}"
    )


def parse_theme_labeling_response(response_text: str) -> dict[str, Any]:
    """Parse and validate a JSON theme labeling response."""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("theme labeling response must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("theme labeling response must be a JSON object")

    missing_fields = [field for field in REQUIRED_LABEL_FIELDS if field not in parsed]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"theme labeling response is missing required field(s): {missing}")
    return parsed


def label_theme_with_mock(
    theme_row: Mapping[str, Any] | pd.Series,
    representative_reviews: pd.DataFrame | Sequence[Mapping[str, Any]],
    keywords: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[str],
    config: ThemeLabelingConfig,
) -> dict[str, Any]:
    """Return a deterministic offline label for tests and local development."""
    theme = _mapping_from_row(theme_row)
    cluster_id = int(theme.get("cluster_id"))
    examples = _representative_review_records(
        representative_reviews,
        max_reviews=config.max_representative_reviews,
    )
    keyword_records = _keyword_records(keywords, max_keywords=config.max_keywords)
    keyword_values = [record["keyword"] for record in keyword_records]
    interpretation = str(theme.get("interpretation", "no_clear_rating_signal"))

    if keyword_values:
        label = " / ".join(keyword.title() for keyword in keyword_values[:3])
    else:
        label = f"Theme {cluster_id}"

    insufficient = interpretation == "insufficient_sample" or (
        config.require_evidence and not examples
    )
    theme_type = _theme_type_from_interpretation(interpretation, insufficient)
    evidence = [
        str(example.get("review_text", ""))
        for example in examples
        if str(example.get("review_text", "")).strip()
    ]

    summary = _mock_summary(keyword_values, evidence, insufficient)
    return {
        "cluster_id": cluster_id,
        "theme_label": label,
        "theme_type": theme_type,
        "summary": summary,
        "evidence": evidence,
        "suggested_action": _mock_action(theme_type, insufficient),
        "confidence": "low" if insufficient else "medium",
        "caution": _mock_caution(theme, insufficient),
    }


def label_theme(
    theme_row: Mapping[str, Any] | pd.Series,
    representative_reviews: pd.DataFrame | Sequence[Mapping[str, Any]],
    keywords: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[str],
    config: ThemeLabelingConfig,
    client: object | None = None,
) -> dict[str, Any]:
    """Label a theme with the configured provider.

    Phase 5 only implements the deterministic mock provider. Future live API
    integrations should read credentials from environment variables and must
    never hard-code or commit API keys.
    """
    if config.provider == "mock":
        return label_theme_with_mock(theme_row, representative_reviews, keywords, config)

    raise NotImplementedError(
        "Only the mock provider is implemented. Future live providers should be "
        "optional, disabled by default, and configured with environment variables."
    )


def _mapping_from_row(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _selected_theme_fields(theme: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "cluster_id",
        "theme_count",
        "total_count",
        "prevalence",
        "prevalence_ci_lower",
        "prevalence_ci_upper",
        "rating_gap",
        "p_value_adjusted",
        "interpretation",
        "insufficient_sample",
    )
    return {field: _json_safe(theme.get(field)) for field in fields if field in theme}


def _representative_review_records(
    representative_reviews: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    max_reviews: int,
) -> list[dict[str, Any]]:
    if max_reviews <= 0:
        raise ValueError("max_representative_reviews must be greater than zero")
    if isinstance(representative_reviews, pd.DataFrame):
        records = representative_reviews.to_dict(orient="records")
    else:
        records = [dict(record) for record in representative_reviews]
    return [
        {
            "review_id": str(record.get("review_id", "")),
            "review_text": str(record.get("review_text", "")),
        }
        for record in records[:max_reviews]
    ]


def _keyword_records(
    keywords: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    max_keywords: int,
) -> list[dict[str, Any]]:
    if max_keywords <= 0:
        raise ValueError("max_keywords must be greater than zero")
    if isinstance(keywords, pd.DataFrame):
        records = keywords.to_dict(orient="records")
    else:
        records = [
            {"keyword": keyword} if isinstance(keyword, str) else dict(keyword)
            for keyword in keywords
        ]
    return [
        {
            "keyword": str(record.get("keyword", "")),
            "score": _json_safe(record.get("score")),
        }
        for record in records[:max_keywords]
        if str(record.get("keyword", "")).strip()
    ]


def _theme_type_from_interpretation(interpretation: str, insufficient: bool) -> str:
    if insufficient:
        return "insufficient_evidence"
    if interpretation == "higher_risk_theme":
        return "customer_pain_point"
    if interpretation == "potential_positive_theme":
        return "positive_driver"
    return "mixed_or_unclear"


def _mock_summary(keywords: list[str], evidence: list[str], insufficient: bool) -> str:
    if insufficient:
        return "Evidence is insufficient for a strong label; review more examples before using this theme."
    keyword_text = ", ".join(keywords[:3]) if keywords else "the provided examples"
    if evidence:
        return f"Reviews in this theme repeatedly mention {keyword_text}."
    return f"The supplied keywords suggest a theme around {keyword_text}."


def _mock_action(theme_type: str, insufficient: bool) -> str:
    if insufficient:
        return "Collect or review more evidence before prioritizing action."
    if theme_type == "customer_pain_point":
        return "Review representative complaints and investigate whether the issue is addressable."
    if theme_type == "positive_driver":
        return "Preserve the positive driver and consider using it in evidence-backed messaging."
    return "Use the examples and statistics as exploratory context before deciding next steps."


def _mock_caution(theme: Mapping[str, Any], insufficient: bool) -> str:
    if insufficient:
        return "Insufficient sample or evidence; treat this label as tentative."
    return (
        "Exploratory label based only on supplied keywords, examples, and "
        "statistical signals; not causal or company-specific."
    )


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
