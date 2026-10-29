"""Error-rate metrics.

Predictions may be hard labels (0/1) or probabilities of predicting 1. Probabilities
arise from the randomised classifier that fictitious play outputs; with them every rate
below is the *expected* rate, as in Kearns et al.
"""
from __future__ import annotations

import numpy as np

# For each metric: which rows it is measured on, and the per-row "error" it averages.
# fpr: among true negatives, how often we predict 1.
# fnr: among true positives, how often we predict 0.
# positive_rate: among everyone, how often we predict 1 (statistical parity).
# error_rate: among everyone, how often the prediction is wrong.
# calibration: among everyone, predicted risk minus observed outcome. Needs risk scores;
#   a positive subgroup gap means the model over-states that group's risk relative to
#   everyone else, a negative gap that it under-states it.
METRICS = ("fpr", "fnr", "positive_rate", "error_rate", "calibration")
METRIC_LABELS = {
    "fpr": "false positive rate",
    "fnr": "false negative rate",
    "positive_rate": "positive prediction rate",
    "error_rate": "error rate",
    "calibration": "calibration gap (predicted minus observed risk)",
}


def check_metric(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose one of {METRICS}")
    return metric


def base_mask(y: np.ndarray, metric: str) -> np.ndarray:
    """Rows on which ``metric`` is defined (its conditioning population)."""
    check_metric(metric)
    y = np.asarray(y)
    if metric == "fpr":
        return y == 0
    if metric == "fnr":
        return y == 1
    return np.ones(len(y), dtype=bool)


def row_errors(y: np.ndarray, p: np.ndarray, metric: str) -> np.ndarray:
    """Per-row contribution to ``metric`` (only meaningful on ``base_mask`` rows)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if metric in ("fpr", "positive_rate"):
        return p
    if metric == "fnr":
        return 1.0 - p
    if metric == "calibration":
        return p - y
    return np.abs(y - p)  # error_rate


def rate(y: np.ndarray, p: np.ndarray, metric: str, mask: np.ndarray | None = None) -> float:
    """``metric`` computed on the rows selected by ``mask`` (all rows if None)."""
    sel = base_mask(y, metric)
    if mask is not None:
        sel = sel & np.asarray(mask, dtype=bool)
    if not sel.any():
        return float("nan")
    return float(row_errors(y, p, metric)[sel].mean())


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Overall accuracy and error rates."""
    out = {m: rate(y, p, m) for m in METRICS}
    out["accuracy"] = 1.0 - out["error_rate"]
    out["balanced_accuracy"] = 1.0 - (out["fpr"] + out["fnr"]) / 2
    return out
