"""Diagnostics for candidate theme cluster outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_cluster_outputs(labels: np.ndarray | list[int]) -> np.ndarray:
    """Validate cluster labels and return them as a one-dimensional array."""
    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("cluster labels must be a one-dimensional array")
    if label_array.size == 0:
        return label_array.astype(int)
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError("cluster labels must be integers")
    return label_array.astype(int, copy=False)


def summarize_clusters(labels: np.ndarray | list[int]) -> dict[str, Any]:
    """Summarize cluster counts, preserving HDBSCAN noise label ``-1``."""
    label_array = validate_cluster_outputs(labels)
    total_rows = int(label_array.shape[0])
    noise_count = int((label_array == -1).sum())
    non_noise_labels = label_array[label_array != -1]
    unique_clusters = sorted(int(label) for label in np.unique(non_noise_labels))

    cluster_size_table = pd.Series(non_noise_labels, dtype="int64").value_counts().sort_index()
    cluster_sizes = [
        {"cluster_id": int(cluster_id), "row_count": int(row_count)}
        for cluster_id, row_count in cluster_size_table.items()
    ]

    return {
        "total_rows": total_rows,
        "number_of_clusters": int(len(unique_clusters)),
        "noise_count": noise_count,
        "noise_share": float(noise_count / total_rows) if total_rows else 0.0,
        "cluster_sizes": cluster_sizes,
    }
