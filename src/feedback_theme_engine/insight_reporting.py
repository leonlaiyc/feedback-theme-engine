"""Build evidence-backed theme insight records and Markdown reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from feedback_theme_engine.llm_labeling import ThemeLabelingConfig, label_theme


REPORT_CAVEAT = "These insights are exploratory and evidence-backed, not causal conclusions."


def build_theme_insight_record(
    theme_row: Mapping[str, Any] | pd.Series,
    representative_reviews: pd.DataFrame,
    keywords: pd.DataFrame,
    label_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine statistics, evidence, keywords, and label output for one theme."""
    theme = theme_row.to_dict() if isinstance(theme_row, pd.Series) else dict(theme_row)
    return {
        "cluster_id": int(theme["cluster_id"]),
        "theme_label": label_output["theme_label"],
        "theme_type": label_output["theme_type"],
        "summary": label_output["summary"],
        "evidence": list(label_output["evidence"]),
        "suggested_action": label_output["suggested_action"],
        "confidence": label_output["confidence"],
        "caution": label_output["caution"],
        "keywords": _keyword_list(keywords),
        "representative_reviews": _review_list(representative_reviews),
        "statistics": _statistics_from_theme(theme),
    }


def build_insight_report(
    signal_frame: pd.DataFrame,
    representative_reviews: pd.DataFrame,
    keywords: pd.DataFrame,
    config: ThemeLabelingConfig,
) -> list[dict[str, Any]]:
    """Build insight records for all themes in the signal table."""
    _validate_signal_frame(signal_frame)
    records: list[dict[str, Any]] = []
    for _, theme_row in signal_frame.sort_values("cluster_id").iterrows():
        cluster_id = int(theme_row["cluster_id"])
        theme_examples = _filter_cluster(representative_reviews, cluster_id)
        theme_keywords = _filter_cluster(keywords, cluster_id)
        label_output = label_theme(theme_row, theme_examples, theme_keywords, config)
        records.append(
            build_theme_insight_record(
                theme_row,
                theme_examples,
                theme_keywords,
                label_output,
            )
        )
    return records


def render_markdown_insight_report(
    insight_records: Sequence[Mapping[str, Any]],
    title: str = "Theme Insight Drafts",
) -> str:
    """Render insight records as a concise Markdown report."""
    lines = [f"# {title}", "", REPORT_CAVEAT, ""]
    for record in insight_records:
        lines.extend(
            [
                f"## Cluster {record['cluster_id']}: {record['theme_label']}",
                "",
                f"- Type: {record['theme_type']}",
                f"- Confidence: {record['confidence']}",
                f"- Summary: {record['summary']}",
                f"- Suggested action: {record['suggested_action']}",
                f"- Caution: {record['caution']}",
            ]
        )
        statistics = record.get("statistics", {})
        if statistics:
            lines.append(
                "- Signals: "
                f"prevalence={statistics.get('prevalence')}, "
                f"rating_gap={statistics.get('rating_gap')}, "
                f"adjusted_p={statistics.get('p_value_adjusted')}, "
                f"interpretation={statistics.get('interpretation')}"
            )
        keywords = record.get("keywords", [])
        if keywords:
            lines.append(f"- Keywords: {', '.join(keywords)}")
        evidence = record.get("evidence", [])
        if evidence:
            lines.append("- Evidence:")
            lines.extend(f"  - {text}" for text in evidence)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def records_to_json(records: Sequence[Mapping[str, Any]]) -> str:
    """Serialize insight records as stable JSON."""
    return json.dumps(list(records), indent=2, sort_keys=True)


def _validate_signal_frame(signal_frame: pd.DataFrame) -> None:
    if not isinstance(signal_frame, pd.DataFrame):
        raise TypeError("signal_frame must be a pandas DataFrame")
    if "cluster_id" not in signal_frame.columns:
        raise ValueError("signal_frame must include cluster_id")


def _filter_cluster(frame: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    if frame.empty or "cluster_id" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["cluster_id"].astype(int) == cluster_id].copy()


def _keyword_list(keywords: pd.DataFrame) -> list[str]:
    if keywords.empty or "keyword" not in keywords.columns:
        return []
    return [str(keyword) for keyword in keywords["keyword"].tolist()]


def _review_list(representative_reviews: pd.DataFrame) -> list[dict[str, str]]:
    if representative_reviews.empty:
        return []
    return [
        {
            "review_id": str(row.get("review_id", "")),
            "review_text": str(row.get("review_text", "")),
        }
        for row in representative_reviews.to_dict(orient="records")
    ]


def _statistics_from_theme(theme: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "theme_count",
        "total_count",
        "prevalence",
        "prevalence_ci_lower",
        "prevalence_ci_upper",
        "rating_gap",
        "p_value_adjusted",
        "effect_size_rank_biserial",
        "interpretation",
        "insufficient_sample",
    )
    return {field: _json_safe(theme.get(field)) for field in fields if field in theme}


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
