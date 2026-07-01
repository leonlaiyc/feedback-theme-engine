"""Effect size helpers for exploratory theme-rating comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rank_biserial_from_u(u_statistic: float, n_x: int, n_y: int) -> float:
    """Return rank-biserial correlation from a Mann-Whitney U statistic.

    The value ranges from -1 to 1. Positive values indicate that observations in
    ``x`` tend to be larger than observations in ``y``. Empty inputs return
    ``nan`` because the statistic is undefined.
    """
    if n_x <= 0 or n_y <= 0:
        return float("nan")
    return float((2.0 * u_statistic / (n_x * n_y)) - 1.0)


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Return Cliff's delta for two numeric samples.

    Positive values indicate that values in ``x`` tend to be larger than values
    in ``y``. Empty samples return ``nan`` because the effect size is undefined.
    """
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    if x_values.size == 0 or y_values.size == 0:
        return float("nan")

    greater = 0
    lower = 0
    for value in x_values:
        greater += int((value > y_values).sum())
        lower += int((value < y_values).sum())
    return float((greater - lower) / (x_values.size * y_values.size))
