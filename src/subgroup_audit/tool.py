"""Input validation and the one-call audit used by the API and the UI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .audit import audit
from .metrics import METRICS

MAX_ROWS = 500_000
MAX_PROTECTED = 5
MAX_LEVELS = 12
_TRUE = {"1", "true", "yes", "y", "t", "positive", "pos"}
_FALSE = {"0", "false", "no", "n", "f", "negative", "neg"}


class AuditInputError(ValueError):
    """Raised for user-fixable problems with the uploaded data or settings."""


def _binary(col: pd.Series, name: str, allow_prob: bool = False) -> np.ndarray:
    if pd.api.types.is_bool_dtype(col):
        return col.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(col):
        v = col.to_numpy(dtype=float)
        if np.isnan(v).any():
            raise AuditInputError(f"column '{name}' has missing values")
        if set(np.unique(v)) <= {0.0, 1.0}:
            return v.astype(int)
        if allow_prob and v.min() >= 0 and v.max() <= 1:
            return v  # probabilities; rates become expected rates
        raise AuditInputError(f"column '{name}' must be 0/1" + (" or a probability in [0, 1]" if allow_prob else ""))
    s = col.astype(str).str.strip().str.lower()
    if not set(s.unique()) <= _TRUE | _FALSE:
        raise AuditInputError(f"column '{name}' must be binary (0/1, true/false, yes/no)")
    return s.isin(_TRUE).astype(int).to_numpy()


def _categorical(col: pd.Series, name: str) -> pd.Series:
    if col.isna().any():
        col = col.astype(object).where(col.notna(), "missing")
    if pd.api.types.is_numeric_dtype(col) and col.nunique() > MAX_LEVELS:
        # e.g. raw age: bin into quartiles so subgroups stay interpretable
        return pd.qcut(col, 4, duplicates="drop").astype(str)
    if col.nunique() > MAX_LEVELS:
        raise AuditInputError(f"protected attribute '{name}' has {col.nunique()} values; "
                              f"group it into at most {MAX_LEVELS} categories first")
    return col.astype(str)


def prepare(df: pd.DataFrame, label_col: str, pred_col: str, protected_cols: list[str]):
    if len(df) == 0:
        raise AuditInputError("the file has no rows")
    if len(df) > MAX_ROWS:
        raise AuditInputError(f"the file has {len(df):,} rows; the limit is {MAX_ROWS:,}")
    missing = [c for c in [label_col, pred_col, *protected_cols] if c not in df.columns]
    if missing:
        raise AuditInputError(f"columns not found: {', '.join(missing)}")
    if not protected_cols:
        raise AuditInputError("choose at least one protected attribute")
    if len(protected_cols) > MAX_PROTECTED:
        raise AuditInputError(f"choose at most {MAX_PROTECTED} protected attributes")
    if label_col in protected_cols or pred_col in protected_cols or label_col == pred_col:
        raise AuditInputError("label, prediction and protected columns must all be different")
    y = _binary(df[label_col], label_col)
    p = _binary(df[pred_col], pred_col, allow_prob=True)
    groups = pd.DataFrame({c: _categorical(df[c], c) for c in protected_cols}).reset_index(drop=True)
    return y, p, groups


def flag_top(scores: np.ndarray, top_pct: float) -> np.ndarray:
    """Flag the ``top_pct`` percent highest-risk rows, as a programme with fixed capacity would."""
    return (scores >= np.quantile(scores, 1 - top_pct / 100)).astype(int)


def run_audit(df: pd.DataFrame, label_col: str, pred_col: str, protected_cols: list[str],
              metric: str = "fpr", max_order: int = 3, min_size: int = 30,
              flag_top_pct: float | None = None) -> dict:
    """Audit predictions or risk scores.

    ``flag_top_pct``: treat ``pred_col`` as a risk score and flag the top X% of rows
    (e.g. 20 for a care-management programme that can enrol one patient in five).
    Calibration is always computed on the scores themselves.
    """
    if metric not in METRICS:
        raise AuditInputError(f"metric must be one of {', '.join(METRICS)}")
    if not 1 <= max_order <= 3:
        raise AuditInputError("max_order must be 1, 2 or 3")
    if flag_top_pct is not None and not 0 < flag_top_pct < 100:
        raise AuditInputError("flag_top_pct must be between 0 and 100")
    y, p, groups = prepare(df, label_col, pred_col, protected_cols)
    if metric in ("fpr", "fnr", "calibration") and len(np.unique(y)) < 2:
        raise AuditInputError("the outcome column has only one class, so this metric is undefined")
    warnings = []
    is_score = not set(np.unique(p)) <= {0.0, 1.0}
    if metric == "calibration" and not is_score:
        warnings.append("Calibration needs risk scores; with 0/1 predictions it only compares flag "
                        "rates with outcome rates.")
    if flag_top_pct is not None and metric != "calibration":
        p = flag_top(p, flag_top_pct)
    elif is_score and metric != "calibration":
        warnings.append("Predictions are probabilities, so rates are expected rates. Set a capacity "
                        "(flag the top X%) to audit how the score is actually used.")
    result = audit(y, p, groups, metric, max_order=max_order, min_size=min_size)
    from .report import describe

    for key in ("single_attribute", "top_weighted", "top_raw_gap", "worst_single", "worst_intersection"):
        rows = result[key] if isinstance(result[key], list) else [result[key]] if result[key] else []
        for r in rows:
            r["label"] = describe(r["subgroup"])
    result["protected_attributes"] = protected_cols
    result["flag_top_pct"] = flag_top_pct if metric != "calibration" else None
    result["warnings"] = warnings
    return result
