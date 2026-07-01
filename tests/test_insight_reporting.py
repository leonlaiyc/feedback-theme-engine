import pandas as pd

from feedback_theme_engine.insight_reporting import (
    REPORT_CAVEAT,
    build_insight_report,
    build_theme_insight_record,
    render_markdown_insight_report,
)
from feedback_theme_engine.llm_labeling import ThemeLabelingConfig


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [1],
            "theme_count": [10],
            "total_count": [50],
            "prevalence": [0.2],
            "prevalence_ci_lower": [0.1],
            "prevalence_ci_upper": [0.33],
            "rating_gap": [-0.8],
            "p_value_adjusted": [0.04],
            "effect_size_rank_biserial": [-0.5],
            "interpretation": ["higher_risk_theme"],
            "insufficient_sample": [False],
        }
    )


def _representatives() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [1],
            "review_id": ["r1"],
            "review_text": ["The pump stopped working."],
        }
    )


def _keywords() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [1],
            "keyword": ["pump"],
            "score": [0.9],
        }
    )


def test_build_theme_insight_record_combines_sources() -> None:
    label_output = {
        "cluster_id": 1,
        "theme_label": "Pump Failure",
        "theme_type": "customer_pain_point",
        "summary": "Customers mention pump issues.",
        "evidence": ["The pump stopped working."],
        "suggested_action": "Review pump complaints.",
        "confidence": "medium",
        "caution": "Exploratory.",
    }

    record = build_theme_insight_record(
        _signal_frame().iloc[0],
        _representatives(),
        _keywords(),
        label_output,
    )

    assert record["cluster_id"] == 1
    assert record["theme_label"] == "Pump Failure"
    assert record["keywords"] == ["pump"]
    assert record["statistics"]["rating_gap"] == -0.8


def test_build_insight_report_uses_mock_labeling() -> None:
    records = build_insight_report(
        _signal_frame(),
        _representatives(),
        _keywords(),
        ThemeLabelingConfig(provider="mock"),
    )

    assert len(records) == 1
    assert records[0]["cluster_id"] == 1
    assert records[0]["theme_type"] == "customer_pain_point"


def test_render_markdown_insight_report_contains_caveat() -> None:
    records = build_insight_report(
        _signal_frame(),
        _representatives(),
        _keywords(),
        ThemeLabelingConfig(provider="mock"),
    )

    markdown = render_markdown_insight_report(records)

    assert REPORT_CAVEAT in markdown
    assert "Cluster 1" in markdown
    assert "The pump stopped working." in markdown
