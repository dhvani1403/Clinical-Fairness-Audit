"""Subgroup fairness auditing.

Following Kearns, Neel, Roth & Wu (2018), a subgroup g is scored by its
*size-weighted disparity*

    gamma(g) = alpha(g) * |rate(g) - rate(all)|

where ``rate`` is the chosen error metric (e.g. false positive rate) and
``alpha(g)`` is the fraction of all rows that are both in g and in the metric's
base population (e.g. true negatives for FPR). Weighting by alpha keeps tiny
groups from dominating the audit; we also report the raw gap with a standard
error so large-but-noisy disparities in small groups are visible and labelled.

Two auditors are provided:

* :class:`EnumerationAuditor` -- exhaustively checks every conjunction of
  attribute values up to ``max_order`` attributes (e.g. sex x race x age band).
  Order 1 = the usual "marginal" single-attribute check.
* :class:`LinearThresholdAuditor` -- the paper's heuristic over the richer class
  of linear threshold functions of the protected attributes (fit by regression).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .metrics import METRIC_LABELS, base_mask, check_metric, row_errors, summary


# --------------------------------------------------------------------------- #
# Enumerated subgroups
# --------------------------------------------------------------------------- #

def enumerate_subgroups(groups: pd.DataFrame, max_order: int = 3) -> tuple[list[dict], np.ndarray]:
    """All non-empty conjunctions ``attr1=v1 & attr2=v2 ...`` up to ``max_order``.

    Returns (descriptions, membership) where membership is a boolean (n, G) matrix.
    """
    groups = groups.astype(str).reset_index(drop=True)
    cols = list(groups.columns)
    descs, members = [], []
    for order in range(1, min(max_order, len(cols)) + 1):
        for attrs in combinations(cols, order):
            keys = groups[list(attrs)].agg(" & ".join, axis=1) if order > 1 else groups[attrs[0]]
            codes, uniques = pd.factorize(keys)
            for k in range(len(uniques)):
                values = dict(zip(attrs, groups.loc[codes == k, list(attrs)].iloc[0]))
                descs.append({
                    "subgroup": " & ".join(f"{a}={v}" for a, v in values.items()),
                    "order": order,
                    "attributes": list(attrs),
                    "values": values,
                })
                members.append(codes == k)
    return descs, np.column_stack(members) if members else np.zeros((len(groups), 0), bool)


def _score(y, p, metric, membership):
    """Vectorised rate / gap / weighted gap for each column of ``membership``."""
    n = len(y)
    base = base_mask(y, metric)
    err = row_errors(y, p, metric)
    overall = float(err[base].mean())
    mb = membership & base[:, None]
    base_size = mb.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_g = (mb * err[:, None]).sum(axis=0) / base_size
    gap = rate_g - overall
    alpha = base_size / n
    weighted = alpha * np.abs(gap)
    se = np.sqrt(np.clip(rate_g * (1 - rate_g), 1e-12, None) / np.maximum(base_size, 1))
    return overall, membership.sum(axis=0), base_size, rate_g, gap, weighted, se


def subgroup_table(y, p, groups: pd.DataFrame, metric: str = "fpr", max_order: int = 3,
                   min_size: int = 30) -> tuple[pd.DataFrame, float]:
    """Score every enumerated subgroup; return (table sorted by weighted gap, overall rate).

    ``min_size`` is the minimum number of base-population rows (e.g. true negatives
    for FPR) for a subgroup's raw gap to be treated as reliable.
    """
    check_metric(metric)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    descs, membership = enumerate_subgroups(groups, max_order)
    overall, size, base_size, rate_g, gap, weighted, se = _score(y, p, metric, membership)
    table = pd.DataFrame(descs)
    table["size"] = size
    table["share"] = size / len(y)
    table["base_size"] = base_size
    table["rate"] = rate_g
    table["gap"] = gap
    table["weighted_gap"] = weighted
    table["std_err"] = se
    table["z"] = gap / se
    table["reliable"] = base_size >= min_size
    table = table[table["base_size"] > 0]
    return table.sort_values("weighted_gap", ascending=False).reset_index(drop=True), overall


def audit(y, p, groups: pd.DataFrame, metric: str = "fpr", max_order: int = 3,
          min_size: int = 30, top_k: int = 10) -> dict:
    """Full audit: overall metrics, single-attribute results, worst intersections.

    The *gerrymandering ratio* compares the largest reliable raw gap among
    intersectional subgroups (order >= 2) with the largest among single
    attributes. A ratio well above 1 means the single-attribute check hides
    unfairness that only shows up at intersections.
    """
    table, overall = subgroup_table(y, p, groups, metric, max_order, min_size)
    reliable = table[table["reliable"]]
    single = table[table["order"] == 1].sort_values(["attributes", "subgroup"], key=lambda s: s.astype(str))
    inter = reliable[reliable["order"] >= 2]
    rel_single = reliable[reliable["order"] == 1]

    def worst(df):
        if df.empty:
            return None
        row = df.loc[df["gap"].abs().idxmax()]
        return _row(row)

    worst_single, worst_inter = worst(rel_single), worst(inter)
    ratio = None
    if worst_single and worst_inter and abs(worst_single["gap"]) > 0:
        ratio = abs(worst_inter["gap"]) / abs(worst_single["gap"])

    by_raw = reliable.reindex(reliable["gap"].abs().sort_values(ascending=False).index)
    return {
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "n_rows": int(len(y)),
        "max_order": max_order,
        "min_size": min_size,
        "n_subgroups_checked": int(len(table)),
        "overall": summary(np.asarray(y), np.asarray(p, dtype=float)),
        "overall_rate": overall,
        "single_attribute": [_row(r) for _, r in single.iterrows()],
        "top_weighted": [_row(r) for _, r in table.head(top_k).iterrows()],
        "top_raw_gap": [_row(r) for _, r in by_raw.head(top_k).iterrows()],
        "worst_single": worst_single,
        "worst_intersection": worst_inter,
        "gerrymandering_ratio": ratio,
        "max_weighted_gap": float(table["weighted_gap"].max()) if len(table) else 0.0,
        "max_weighted_gap_single": float(rel_single["weighted_gap"].max()) if len(rel_single) else 0.0,
    }


def _row(r: pd.Series) -> dict:
    return {
        "subgroup": r["subgroup"],
        "order": int(r["order"]),
        "size": int(r["size"]),
        "share": float(r["share"]),
        "base_size": int(r["base_size"]),
        "rate": float(r["rate"]),
        "gap": float(r["gap"]),
        "weighted_gap": float(r["weighted_gap"]),
        "std_err": float(r["std_err"]),
        "z": float(r["z"]),
        "reliable": bool(r["reliable"]),
    }


# --------------------------------------------------------------------------- #
# Auditors used inside the fair learner (and for the rich-class audit)
# --------------------------------------------------------------------------- #

class EnumerationAuditor:
    """Exact best response over conjunctions of up to ``max_order`` attributes."""

    def __init__(self, groups: pd.DataFrame, max_order: int = 3):
        self.descs, self.membership = enumerate_subgroups(groups, max_order)

    def best_group(self, y, p, metric):
        """Return (membership vector, signed weighted disparity, description)."""
        n = len(y)
        base = base_mask(y, metric)
        err = row_errors(y, p, metric)
        resid = np.where(base, err - err[base].mean(), 0.0)
        signed = self.membership.T.astype(float) @ resid / n  # = alpha(g) * (rate_g - rate)
        k = int(np.argmax(np.abs(signed)))
        return self.membership[:, k], float(signed[k]), self.descs[k]["subgroup"]


class LinearThresholdAuditor:
    """The paper's heuristic auditor over linear threshold functions of X'.

    Regress each base row's residual (its error minus the overall rate) on the
    protected attributes; the group is {x : w.x > 0} (or < 0 for the other
    direction). This mirrors GerryFair's ``Auditor.get_group``.
    """

    def __init__(self, X_prime: pd.DataFrame):
        self.X_prime = np.asarray(X_prime, dtype=float)

    def best_group(self, y, p, metric):
        n = len(y)
        base = base_mask(y, metric)
        err = row_errors(y, p, metric)
        resid = np.where(base, err - err[base].mean(), 0.0)
        reg = LinearRegression().fit(self.X_prime[base], resid[base])
        score = reg.predict(self.X_prime)
        best = (np.zeros(n, bool), 0.0)
        for g in (score > 0, score < 0):
            signed = float(resid[g].sum() / n)
            if abs(signed) > abs(best[1]):
                best = (g, signed)
        return best[0], best[1], "linear threshold over protected attributes"


def rich_audit(y, p, X_prime: pd.DataFrame, metric: str = "fpr") -> dict:
    """Audit against the linear-threshold class; returns the paper's gamma-unfairness."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    g, signed, _ = LinearThresholdAuditor(X_prime).best_group(y, p, metric)
    base = base_mask(y, metric)
    err = row_errors(y, p, metric)
    sel = g & base
    return {
        "weighted_gap": abs(signed),
        "group_share": float(g.mean()),
        "group_rate": float(err[sel].mean()) if sel.any() else float("nan"),
        "overall_rate": float(err[base].mean()),
    }
