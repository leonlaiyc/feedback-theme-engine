import math

import pandas as pd
import pytest

from feedback_theme_engine.statistical_signals import (
    StatisticalSignalConfig,
    adjust_theme_p_values,
    build_statistical_signal_table,
    compute_rating_association,
    compute_theme_prevalence,
    interpret_theme_signals,
)


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(24)],
            "review_text": [f"review {i}" for i in range(24)],
            "cluster_id": [0] * 10 + [1] * 10 + [-1] * 4,
            "rating": [1] * 10 + [5] * 10 + [3, 3, 4, 4],
        }
    )


def test_compute_theme_prevalence_excludes_noise_and_adds_wilson_bounds() -> None:
    prevalence = compute_theme_prevalence(_signal_frame(), StatisticalSignalConfig())

    assert prevalence["cluster_id"].tolist() == [0, 1]
    assert prevalence.loc[0, "theme_count"] == 10
    assert prevalence.loc[0, "total_count"] == 24
    assert 0 <= prevalence.loc[0, "prevalence_ci_lower"] <= prevalence.loc[0, "prevalence"]
    assert prevalence.loc[0, "prevalence"] <= prevalence.loc[0, "prevalence_ci_upper"] <= 1


def test_compute_theme_prevalence_handles_all_noise() -> None:
    frame = pd.DataFrame({"cluster_id": [-1, -1], "rating": [1, 2]})

    prevalence = compute_theme_prevalence(frame, StatisticalSignalConfig())

    assert prevalence.empty
    assert list(prevalence.columns) == [
        "cluster_id",
        "theme_count",
        "total_count",
        "prevalence",
        "prevalence_ci_lower",
        "prevalence_ci_upper",
    ]


def test_compute_rating_association_detects_direction_on_synthetic_groups() -> None:
    association = compute_rating_association(
        _signal_frame(),
        StatisticalSignalConfig(min_theme_size=5),
    )
    cluster_zero = association.loc[association["cluster_id"] == 0].iloc[0]
    cluster_one = association.loc[association["cluster_id"] == 1].iloc[0]

    assert cluster_zero["rating_gap"] < 0
    assert cluster_zero["effect_size_rank_biserial"] < 0
    assert cluster_zero["p_value"] < 0.05
    assert cluster_one["rating_gap"] > 0
    assert cluster_one["effect_size_rank_biserial"] > 0


def test_compute_rating_association_handles_missing_and_insufficient_ratings() -> None:
    frame = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 1, 1, 1],
            "rating": [1, None, 2, 4, 5, 4],
        }
    )

    association = compute_rating_association(frame, StatisticalSignalConfig(min_theme_size=3))
    cluster_zero = association.loc[association["cluster_id"] == 0].iloc[0]

    assert cluster_zero["missing_rating_count_theme"] == 1
    assert bool(cluster_zero["insufficient_sample"]) is True
    assert math.isnan(cluster_zero["p_value"])


def test_adjust_theme_p_values_adds_adjusted_values_and_reject_flag() -> None:
    signal_frame = pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "p_value": [0.001, 0.02, None],
        }
    )

    adjusted = adjust_theme_p_values(signal_frame, StatisticalSignalConfig(alpha=0.05))

    assert "p_value_adjusted" in adjusted.columns
    assert "reject_null" in adjusted.columns
    assert bool(adjusted.loc[0, "reject_null"]) is True
    assert pd.isna(adjusted.loc[2, "p_value_adjusted"])


def test_interpret_theme_signals_uses_cautious_non_causal_labels() -> None:
    signal_frame = pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "rating_gap": [-1.0, 1.0, 0.1],
            "p_value_adjusted": [0.01, 0.01, 0.8],
            "insufficient_sample": [False, False, False],
        }
    )

    interpreted = interpret_theme_signals(signal_frame, StatisticalSignalConfig(alpha=0.05))

    assert interpreted["interpretation"].tolist() == [
        "higher_risk_theme",
        "potential_positive_theme",
        "no_clear_rating_signal",
    ]
    forbidden_words = ("causal", "proof", "impact")
    assert not any(
        word in label
        for label in interpreted["interpretation"].tolist()
        for word in forbidden_words
    )


def test_build_statistical_signal_table_runs_full_workflow() -> None:
    signals = build_statistical_signal_table(
        _signal_frame(),
        StatisticalSignalConfig(min_theme_size=5),
    )

    assert set(signals["cluster_id"]) == {0, 1}
    assert "prevalence" in signals.columns
    assert "p_value_adjusted" in signals.columns
    assert "interpretation" in signals.columns


def test_build_statistical_signal_table_requires_rating_column() -> None:
    frame = pd.DataFrame({"cluster_id": [0, 1]})

    with pytest.raises(ValueError, match="rating"):
        build_statistical_signal_table(frame, StatisticalSignalConfig())
