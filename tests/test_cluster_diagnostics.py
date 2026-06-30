import pytest

from feedback_theme_engine.cluster_diagnostics import (
    summarize_clusters,
    validate_cluster_outputs,
)


def test_summarize_clusters_with_noise_and_clusters() -> None:
    summary = summarize_clusters([-1, 0, 0, 1, 1, 1])

    assert summary["total_rows"] == 6
    assert summary["number_of_clusters"] == 2
    assert summary["noise_count"] == 1
    assert summary["noise_share"] == pytest.approx(1 / 6)
    assert summary["cluster_sizes"] == [
        {"cluster_id": 0, "row_count": 2},
        {"cluster_id": 1, "row_count": 3},
    ]


def test_summarize_clusters_handles_all_noise() -> None:
    summary = summarize_clusters([-1, -1])

    assert summary["total_rows"] == 2
    assert summary["number_of_clusters"] == 0
    assert summary["noise_count"] == 2
    assert summary["noise_share"] == 1.0
    assert summary["cluster_sizes"] == []


def test_validate_cluster_outputs_rejects_non_1d_labels() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_cluster_outputs([[0, 1]])


def test_validate_cluster_outputs_rejects_non_integer_labels() -> None:
    with pytest.raises(TypeError, match="integers"):
        validate_cluster_outputs([0.1, 1.2])
