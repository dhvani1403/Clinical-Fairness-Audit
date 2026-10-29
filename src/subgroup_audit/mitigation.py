"""Subgroup-fair learning by fictitious play (Kearns, Neel, Roth & Wu, 2018).

The constrained problem

    min_h  err(h)   s.t.   |alpha(g) * (rate_g(h) - rate(h))| <= gamma   for every g in G

is solved as a zero-sum game between a Learner (picks a classifier) and an
Auditor (picks a violated subgroup and a Lagrange multiplier). Each round:

1. the Learner best-responds to the Auditor's *average* multipliers by solving a
   cost-sensitive classification problem (reduced here to weighted classification,
   so any scikit-learn classifier works);
2. the Auditor best-responds to the Learner's *average* classifier by finding the
   subgroup with the largest size-weighted disparity; if it exceeds ``gamma`` the
   multiplier on that subgroup grows (bounded by ``C``).

The output is the uniform mixture of the Learner's classifiers: ``predict_proba``
returns the probability that the randomised classifier predicts 1.

Setting ``auditor=EnumerationAuditor(groups, max_order=1)`` constrains only
single attributes ("marginal fairness"), which is how we reproduce the
gerrymandering effect; ``max_order=3`` or ``LinearThresholdAuditor`` gives
subgroup fairness.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .metrics import base_mask, row_errors

TRAINABLE_METRICS = ("fpr", "fnr", "positive_rate")


class _Constant:
    def __init__(self, value: int):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _fit_weighted(est, X, z, w):
    if len(np.unique(z)) < 2:
        return _Constant(int(z[0]))
    est = clone(est)
    if isinstance(est, Pipeline):
        est.fit(X, z, **{f"{est.steps[-1][0]}__sample_weight": w})
    else:
        est.fit(X, z, sample_weight=w)
    return est


class SubgroupFairClassifier:
    """Fictitious-play learner for subgroup fairness.

    Parameters
    ----------
    base_estimator : scikit-learn classifier supporting ``sample_weight``.
    metric : "fpr", "fnr" or "positive_rate".
    C : bound on the Lagrange multipliers (fairness strength ceiling).
    gamma : tolerated size-weighted disparity; smaller = stricter.
    max_iter : rounds of fictitious play.
    positive_cost : cost of a false negative relative to a false positive (use > 1
        for rare positive classes, e.g. readmission).
    reference_index_bug : reproduce the indexing bug in GerryFair's
        ``Auditor.update_costs`` (every row receives the penalty of the first base
        row instead of its own). Only used to study the reference implementation.
    """

    def __init__(self, base_estimator, metric: str = "fpr", C: float = 10.0,
                 gamma: float = 0.01, max_iter: int = 50, positive_cost: float = 1.0,
                 reference_index_bug: bool = False):
        if metric not in TRAINABLE_METRICS:
            raise ValueError(f"metric must be one of {TRAINABLE_METRICS}")
        self.base_estimator = base_estimator
        self.metric = metric
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.positive_cost = positive_cost
        self.reference_index_bug = reference_index_bug

    def fit(self, X, y, auditor):
        y = np.asarray(y).astype(int)
        n = len(y)
        base = base_mask(y, self.metric)
        # Cost of predicting 1 / 0 for each row before any fairness penalty.
        err_c1 = (y == 0) / n
        err_c0 = self.positive_cost * (y == 1) / n
        penalty_sum = np.zeros(n)
        self.classifiers_, self.history_ = [], []
        avg = np.zeros(n)
        for t in range(1, self.max_iter + 1):
            penalty = self.C * penalty_sum / t
            if self.metric == "fnr":
                c1, c0 = err_c1, err_c0 + penalty
            else:
                c1, c0 = err_c1 + penalty, err_c0
            z = (c1 < c0).astype(int)
            w = np.abs(c1 - c0)
            w = w / w.mean() if w.mean() > 0 else np.ones(n)
            h = _fit_weighted(self.base_estimator, X, z, w)
            self.classifiers_.append(h)
            avg += (h.predict(X) - avg) / t

            g, signed, desc = auditor.best_group(y, avg, self.metric)
            violated = abs(signed) > self.gamma
            if violated:
                g_base = g & base
                share = g_base.sum() / base.sum()
                if self.reference_index_bug:
                    first = np.flatnonzero(base)[0]
                    v = np.where(base, (float(g[first]) - share) / n, 0.0)
                else:
                    v = np.where(base, (g.astype(float) - share) / n, 0.0)
                penalty_sum += np.sign(signed) * v
            self.history_.append({
                "iteration": t,
                "train_error": float(np.abs(y - avg).mean()),
                "worst_weighted_gap": abs(signed),
                "worst_group": desc,
                "violated": bool(violated),
            })
        return self

    def predict_proba(self, X):
        """Probability that the randomised (mixture) classifier predicts 1."""
        return np.mean([h.predict(X) for h in self.classifiers_], axis=0)

    def predict(self, X, rng=None):
        """Sample a hard label per row from the mixture (keeps the fairness guarantee)."""
        rng = np.random.default_rng(rng)
        return (rng.random(len(X)) < self.predict_proba(X)).astype(int)


def expected_error(y, p) -> float:
    return float(row_errors(np.asarray(y), np.asarray(p), "error_rate").mean())
