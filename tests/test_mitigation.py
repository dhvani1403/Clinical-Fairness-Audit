import numpy as np
from sklearn.linear_model import LogisticRegression

from subgroup_audit.audit import EnumerationAuditor, LinearThresholdAuditor, audit
from subgroup_audit.mitigation import SubgroupFairClassifier


def _biased(seed=0, n=4000):
    """Label depends on x; an unconstrained model is also pushed to over-predict 1 for
    one intersection because the feature it uses is shifted there."""
    rng = np.random.default_rng(seed)
    import pandas as pd
    sex = rng.choice(["M", "F"], n)
    race = rng.choice(["A", "B"], n)
    y = rng.binomial(1, 0.4, n)
    shift = ((sex == "M") == (race == "B")).astype(float) * 1.2
    x = rng.normal(size=n) + 1.5 * y + shift
    X = np.column_stack([x, (sex == "M"), (race == "B"), (sex == "M") & (race == "B")]).astype(float)
    groups = pd.DataFrame({"sex": sex, "race": race})
    return X, y, groups


def test_subgroup_fair_reduces_weighted_disparity():
    X, y, groups = _biased()
    base = LogisticRegression().fit(X[:, :1], y)
    before = audit(y, base.predict(X[:, :1]), groups, "fpr", 2)["max_weighted_gap"]
    clf = SubgroupFairClassifier(LogisticRegression(), "fpr", C=10, gamma=0.002, max_iter=40)
    clf.fit(X, y, EnumerationAuditor(groups, 2))
    after = audit(y, clf.predict_proba(X), groups, "fpr", 2)["max_weighted_gap"]
    assert before > 0.02
    assert after < before / 3
    assert len(clf.history_) == 40


def test_fnr_and_linear_auditor_run():
    X, y, groups = _biased(1, 1500)
    clf = SubgroupFairClassifier(LogisticRegression(), "fnr", gamma=0.005, max_iter=10)
    clf.fit(X, y, LinearThresholdAuditor(X[:, 1:]))
    p = clf.predict_proba(X)
    assert p.shape == (1500,) and 0 <= p.min() and p.max() <= 1
    assert set(np.unique(clf.predict(X, rng=0))) <= {0, 1}
