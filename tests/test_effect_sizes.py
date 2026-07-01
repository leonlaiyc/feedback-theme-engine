import math

import pytest

from feedback_theme_engine.effect_sizes import cliffs_delta, rank_biserial_from_u


def test_rank_biserial_from_u_returns_expected_direction() -> None:
    assert rank_biserial_from_u(9, 3, 3) == pytest.approx(1.0)
    assert rank_biserial_from_u(0, 3, 3) == pytest.approx(-1.0)


def test_rank_biserial_from_u_handles_empty_sample_size() -> None:
    assert math.isnan(rank_biserial_from_u(0, 0, 3))


def test_cliffs_delta_returns_expected_direction() -> None:
    assert cliffs_delta([4, 5], [1, 2]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2], [4, 5]) == pytest.approx(-1.0)


def test_cliffs_delta_handles_empty_inputs() -> None:
    assert math.isnan(cliffs_delta([], [1, 2]))
