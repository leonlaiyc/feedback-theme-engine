"""Exploratory statistical signals for discovered feedback themes."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from feedback_theme_engine.effect_sizes import rank_biserial_from_u


PREVALENCE_COLUMNS: tuple[str, ...] = (
    "cluster_id",
    "theme_count",
    "total_count",
    "prevalence",
    "prevalence_ci_lower",
    "prevalence_ci_upper",
)

ASSOCIATION_COLUMNS: tuple[str, ...] = (
    "cluster_id",
    "rating_count_theme",
    "rating_count_non_theme",
    "missing_rating_count_theme",
    "mean_rating_theme",
    "mean_rating_non_theme",
    "rating_gap",
    "u_statistic",
    "p_value",
    "effect_size_rank_biserial",
    "insufficient_sample",
)


@dataclass(frozen=True)
class StatisticalSignalConfig:
    """Configuration for exploratory theme signal calculations."""

    alpha: float = 0.05
    min_theme_size: int = 10
    p_adjust_method: str = "fdr_bh"
    rating_column: str = "rating"
    cluster_column: str = "cluster_id"
    noise_label: int = -1


def compute_theme_prevalence(
    cluster_frame: pd.DataFrame,
    config: StatisticalSignalConfig,
) -> pd.DataFrame:
    """Compute non-noise theme prevalence with Wilson confidence intervals."""
    _validate_cluster_frame_columns(cluster_frame, config, require_rating=False)
    _validate_config(config)

    total_count = int(len(cluster_frame))
    non_noise = cluster_frame.loc[cluster_frame[config.cluster_column] != config.noise_label]
    rows: list[dict[str, float | int]] = []
    for cluster_id, theme_count in non_noise[config.cluster_column].value_counts().sort_index().items():
        lower, upper = _wilson_interval(int(theme_count), total_count, config.alpha)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "theme_count": int(theme_count),
                "total_count": total_count,
                "prevalence": float(theme_count / total_count) if total_count else 0.0,
                "prevalence_ci_lower": lower,
                "prevalence_ci_upper": upper,
            }
        )
    return pd.DataFrame(rows, columns=list(PREVALENCE_COLUMNS))


def compute_rating_association(
    cluster_frame: pd.DataFrame,
    config: StatisticalSignalConfig,
) -> pd.DataFrame:
    """Compare each non-noise theme rating distribution against non-theme rows."""
    _validate_cluster_frame_columns(cluster_frame, config, require_rating=True)
    _validate_config(config)

    frame = cluster_frame.copy()
    frame[config.rating_column] = pd.to_numeric(frame[config.rating_column], errors="coerce")
    non_noise_labels = sorted(
        int(label)
        for label in frame.loc[
            frame[config.cluster_column] != config.noise_label,
            config.cluster_column,
        ].dropna().unique()
    )

    rows: list[dict[str, float | int | bool]] = []
    for cluster_id in non_noise_labels:
        in_theme = frame[config.cluster_column] == cluster_id
        theme_ratings = frame.loc[in_theme, config.rating_column].dropna().astype(float)
        non_theme_ratings = frame.loc[~in_theme, config.rating_column].dropna().astype(float)
        missing_theme = int(frame.loc[in_theme, config.rating_column].isna().sum())
        insufficient = bool(
            len(theme_ratings) < config.min_theme_size or len(non_theme_ratings) < 2
        )

        row = {
            "cluster_id": int(cluster_id),
            "rating_count_theme": int(len(theme_ratings)),
            "rating_count_non_theme": int(len(non_theme_ratings)),
            "missing_rating_count_theme": missing_theme,
            "mean_rating_theme": _safe_mean(theme_ratings),
            "mean_rating_non_theme": _safe_mean(non_theme_ratings),
            "rating_gap": float("nan"),
            "u_statistic": float("nan"),
            "p_value": float("nan"),
            "effect_size_rank_biserial": float("nan"),
            "insufficient_sample": insufficient,
        }

        if not np.isnan(row["mean_rating_theme"]) and not np.isnan(row["mean_rating_non_theme"]):
            row["rating_gap"] = float(row["mean_rating_theme"] - row["mean_rating_non_theme"])

        if not insufficient:
            u_statistic, p_value = _mann_whitney_u(theme_ratings, non_theme_ratings)
            row["u_statistic"] = u_statistic
            row["p_value"] = p_value
            row["effect_size_rank_biserial"] = rank_biserial_from_u(
                u_statistic,
                len(theme_ratings),
                len(non_theme_ratings),
            )

        rows.append(row)

    return pd.DataFrame(rows, columns=list(ASSOCIATION_COLUMNS))


def adjust_theme_p_values(
    signal_frame: pd.DataFrame,
    config: StatisticalSignalConfig,
) -> pd.DataFrame:
    """Add FDR-adjusted p-values while preserving raw p-values."""
    _validate_config(config)
    if "p_value" not in signal_frame.columns:
        raise ValueError("signal_frame must include p_value")
    if config.p_adjust_method != "fdr_bh":
        raise ValueError("only fdr_bh p-value adjustment is currently supported")

    adjusted = signal_frame.copy()
    p_values = pd.to_numeric(adjusted["p_value"], errors="coerce")
    adjusted["p_value_adjusted"] = np.nan
    adjusted["reject_null"] = False

    valid_mask = p_values.notna()
    if valid_mask.any():
        adjusted_values = _benjamini_hochberg(p_values.loc[valid_mask].to_numpy(dtype=float))
        adjusted.loc[valid_mask, "p_value_adjusted"] = adjusted_values
        adjusted.loc[valid_mask, "reject_null"] = adjusted_values <= config.alpha

    return adjusted


def interpret_theme_signals(
    signal_frame: pd.DataFrame,
    config: StatisticalSignalConfig,
) -> pd.DataFrame:
    """Add cautious exploratory interpretation labels to theme signals."""
    _validate_config(config)
    required = {"insufficient_sample", "rating_gap", "p_value_adjusted"}
    missing = sorted(required.difference(signal_frame.columns))
    if missing:
        raise ValueError(f"signal_frame is missing required column(s): {', '.join(missing)}")

    interpreted = signal_frame.copy()
    interpreted["interpretation"] = interpreted.apply(
        lambda row: _interpret_row(row, config),
        axis=1,
    )
    return interpreted


def build_statistical_signal_table(
    cluster_frame: pd.DataFrame,
    config: StatisticalSignalConfig,
) -> pd.DataFrame:
    """Build the full exploratory theme signal table."""
    prevalence = compute_theme_prevalence(cluster_frame, config)
    association = compute_rating_association(cluster_frame, config)
    signal_frame = prevalence.merge(association, on="cluster_id", how="left")
    adjusted = adjust_theme_p_values(signal_frame, config)
    return interpret_theme_signals(adjusted, config)


def _validate_cluster_frame_columns(
    cluster_frame: pd.DataFrame,
    config: StatisticalSignalConfig,
    *,
    require_rating: bool,
) -> None:
    if not isinstance(cluster_frame, pd.DataFrame):
        raise TypeError("cluster_frame must be a pandas DataFrame")
    required_columns = [config.cluster_column]
    if require_rating:
        required_columns.append(config.rating_column)
    missing = [column for column in required_columns if column not in cluster_frame.columns]
    if missing:
        raise ValueError(f"Missing required signal column(s): {', '.join(missing)}")


def _validate_config(config: StatisticalSignalConfig) -> None:
    if not 0 < config.alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if config.min_theme_size <= 0:
        raise ValueError("min_theme_size must be greater than zero")


def _wilson_interval(successes: int, total: int, alpha: float) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)

    z = NormalDist().inv_cdf(1 - alpha / 2)
    proportion = successes / total
    denominator = 1 + (z**2 / total)
    center = (proportion + (z**2 / (2 * total))) / denominator
    margin = (
        z
        * np.sqrt((proportion * (1 - proportion) / total) + (z**2 / (4 * total**2)))
        / denominator
    )
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def _safe_mean(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def _mann_whitney_u(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:  # pragma: no cover - exercised by users locally
        raise ImportError(
            "Install scipy to compute Mann-Whitney U tests: "
            "py -3 -m pip install -r requirements.txt"
        ) from exc

    result = mannwhitneyu(x.to_numpy(), y.to_numpy(), alternative="two-sided")
    return (float(result.statistic), float(result.pvalue))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(m, dtype=float)

    running_min = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        value = sorted_p[index] * m / rank
        running_min = min(running_min, value)
        adjusted_sorted[index] = min(running_min, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _interpret_row(row: pd.Series, config: StatisticalSignalConfig) -> str:
    if bool(row["insufficient_sample"]) or pd.isna(row["p_value_adjusted"]):
        return "insufficient_sample"
    if row["p_value_adjusted"] <= config.alpha and row["rating_gap"] < 0:
        return "higher_risk_theme"
    if row["p_value_adjusted"] <= config.alpha and row["rating_gap"] > 0:
        return "potential_positive_theme"
    return "no_clear_rating_signal"
