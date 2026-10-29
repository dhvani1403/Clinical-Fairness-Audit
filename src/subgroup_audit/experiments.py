"""End-to-end replication experiment for one dataset.

1. Train baseline logistic regression and gradient boosting.
2. Audit single attributes and intersections on the held-out test set.
3. Train a *marginally* fair model (auditor sees single attributes only) and a
   *subgroup* fair model (auditor sees intersections) with the same fictitious-play
   learner; audit both. The gap between them is the gerrymandering effect.
4. Sweep the fairness strength gamma to trace the accuracy-fairness trade-off.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import audit as A
from .data import Dataset
from .mitigation import SubgroupFairClassifier

CONFIG = {
    # The paper's dataset and metric (false positive rate).
    "communities": {"metric": "fpr", "positive_cost": 1.0, "gamma": 0.005},
    "adult": {"metric": "fpr", "positive_cost": 1.0, "gamma": 0.005},
    # Readmission: missing a high-risk patient is the costly error, so audit FNR and
    # weight positives so the model actually flags patients (9% base rate). FNR disparities
    # are weighted by the share of positives (9%), so the headline gamma is scaled down.
    "diabetes": {"metric": "fnr", "positive_cost": None, "gamma": 0.001},
    # Predicting high healthcare utilisation (the Obermeyer et al. setting): a false
    # negative is a high-need patient the care-management programme never reaches.
    "meps": {"metric": "fnr", "positive_cost": None, "gamma": 0.002},
}
CLINICAL = ("diabetes", "meps")
CAPACITY = 0.20      # share of patients a care-management programme can enrol
HIGH_NEED = 3        # chronic conditions that count as high need (MEPS)
GAMMAS = [0.02, 0.01, 0.005, 0.0025, 0.001]


def logistic(positive_cost=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight={0: 1, 1: positive_cost}))


def boosting(positive_cost=1.0):
    # Shallow, regularised trees: an unregularised booster memorises the 1,400 training
    # communities, leaving the in-sample auditor nothing to find.
    return HistGradientBoostingClassifier(max_iter=100, max_depth=3, min_samples_leaf=50,
                                          early_stopping=False, random_state=0,
                                          class_weight={0: 1, 1: positive_cost})


def run(ds: Dataset, max_order: int = 3, min_size: int = 30, iters: int = 50,
        gbm_iters: int = 25, gammas=GAMMAS, seed: int = 0, log=print) -> dict:
    cfg = CONFIG[ds.name]
    metric = cfg["metric"]
    pos_cost = cfg["positive_cost"] or float((1 - ds.y.mean()) / ds.y.mean())
    idx = np.arange(len(ds.y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=ds.y)
    Xtr, Xte, ytr, yte = ds.X.iloc[tr], ds.X.iloc[te], ds.y[tr], ds.y[te]
    gtr, gte = ds.groups.iloc[tr].reset_index(drop=True), ds.groups.iloc[te].reset_index(drop=True)
    Ptr, Pte = ds.X_prime.iloc[tr].reset_index(drop=True), ds.X_prime.iloc[te].reset_index(drop=True)

    def evaluate(p_test, p_train=None):
        res = A.audit(yte, p_test, gte, metric, max_order, min_size)
        res["rich"] = A.rich_audit(yte, p_test, Pte, metric)
        if p_train is not None:
            res["train_max_weighted_gap"] = A.audit(ytr, p_train, gtr, metric, max_order, min_size)["max_weighted_gap"]
            res["train_rich"] = A.rich_audit(ytr, p_train, Ptr, metric)
        return res

    def fair(order_or_linear, gamma, base, n_iter, bug=False):
        auditor = (A.LinearThresholdAuditor(Ptr) if order_or_linear == "linear"
                   else A.EnumerationAuditor(gtr, order_or_linear))
        clf = SubgroupFairClassifier(base, metric=metric, C=10, gamma=gamma, max_iter=n_iter,
                                     positive_cost=pos_cost, reference_index_bug=bug)
        return clf.fit(Xtr, ytr, auditor)

    out = {"dataset": ds.name, "label": ds.label_desc, "notes": ds.notes, "metric": metric,
           "positive_cost": pos_cost, "n_train": int(len(tr)), "n_test": int(len(te)),
           "base_rate": float(ds.y.mean()), "models": {}, "tradeoff": {}}
    lr, gbm = logistic(pos_cost), boosting(pos_cost)
    # The fair learner applies positive_cost itself, so its base estimators are unweighted.
    lr_base, gbm_base = logistic(), boosting()

    for name, est in (("logistic", lr), ("boosting", gbm)):
        t = time.time()
        est.fit(Xtr, ytr)
        out["models"][name] = evaluate(est.predict(Xte), est.predict(Xtr))
        log(f"[{ds.name}] {name} baseline fit in {time.time() - t:.1f}s")

    headline_gamma = cfg["gamma"]
    out["headline_gamma"] = headline_gamma
    for label, kind in (("marginal_fair", 1), ("subgroup_fair", max_order)):
        t = time.time()
        clf = fair(kind, headline_gamma, lr_base, iters)
        out["models"][label] = evaluate(clf.predict_proba(Xte), clf.predict_proba(Xtr))
        out["models"][label]["history"] = clf.history_
        log(f"[{ds.name}] {label} (logistic) in {time.time() - t:.1f}s")

    t = time.time()
    clf = fair(max_order, headline_gamma, gbm_base, gbm_iters)
    out["models"]["subgroup_fair_boosting"] = evaluate(clf.predict_proba(Xte), clf.predict_proba(Xtr))
    log(f"[{ds.name}] subgroup_fair (boosting) in {time.time() - t:.1f}s")

    for label, kind in (("marginal_fair", 1), ("subgroup_fair", max_order)):
        pts = []
        for g in gammas:
            clf = fair(kind, g, lr_base, iters)
            p = clf.predict_proba(Xte)
            r = A.audit(yte, p, gte, metric, max_order, min_size)
            pts.append({"gamma": g, "accuracy": r["overall"]["accuracy"],
                        "balanced_accuracy": r["overall"]["balanced_accuracy"],
                        "max_weighted_gap": r["max_weighted_gap"],
                        "worst_intersection_gap": abs((r["worst_intersection"] or {"gap": 0})["gap"]),
                        "worst_single_gap": abs((r["worst_single"] or {"gap": 0})["gap"])})
        out["tradeoff"][label] = pts
        log(f"[{ds.name}] trade-off sweep {label} done")

    if ds.name in CLINICAL:
        # Unweighted model for risk scores, so predicted probabilities are calibrated overall.
        scorer = logistic().fit(Xtr, ytr)
        out["clinical"] = clinical_analysis(scorer.predict_proba(Xte)[:, 1], yte, gte, ds, te, max_order, min_size)
        log(f"[{ds.name}] clinical analyses done")
    if ds.name == "communities":
        out["paper"] = paper_replication(fair, lr_base, gammas, iters, Xtr, Xte, ytr, yte, Ptr, Pte, metric, log)
    return out


def paper_replication(fair, lr, gammas, iters, Xtr, Xte, ytr, yte, Ptr, Pte, metric, log):
    """The paper's setting: linear-threshold subgroups over 18 protected attributes.

    Also trains with GerryFair's indexing bug switched on to show its effect.
    """
    res = {"sweep": [], "bug": {}}
    for g in gammas:
        clf = fair("linear", g, lr, iters)
        ptr, pte = clf.predict_proba(Xtr), clf.predict_proba(Xte)
        res["sweep"].append({
            "gamma": g,
            "train_error": float(np.abs(ytr - ptr).mean()),
            "test_error": float(np.abs(yte - pte).mean()),
            "train_unfairness": A.rich_audit(ytr, ptr, Ptr, metric)["weighted_gap"],
            "test_unfairness": A.rich_audit(yte, pte, Pte, metric)["weighted_gap"],
        })
    for bug in (False, True):
        clf = fair("linear", 0.005, lr, iters, bug=bug)
        ptr, pte = clf.predict_proba(Xtr), clf.predict_proba(Xte)
        res["bug"]["with_bug" if bug else "fixed"] = {
            "history": clf.history_,
            "train_error": float(np.abs(ytr - ptr).mean()),
            "train_unfairness": A.rich_audit(ytr, ptr, Ptr, metric)["weighted_gap"],
            "test_error": float(np.abs(yte - pte).mean()),
            "test_unfairness": A.rich_audit(yte, pte, Pte, metric)["weighted_gap"],
        }
    log("[communities] paper-setting sweep and reference-bug comparison done")
    return res


def clinical_analysis(scores, yte, gte, ds: Dataset, te, max_order, min_size) -> dict:
    """How a risk score is actually used: enrol the top ``CAPACITY`` of patients.

    * capacity: who is missed (FNR) when only the top 20% by risk are flagged;
    * calibration: does a given score mean the same risk in every subgroup?
    * need (if the dataset has a need measure): audit the same flags against an
      independent measure of need instead of the proxy label, and compare need
      across races at the same risk score, as in Obermeyer et al. (2019).
    """
    flags = (scores >= np.quantile(scores, 1 - CAPACITY)).astype(int)
    out = {
        "capacity": CAPACITY,
        "capacity_fnr": A.audit(yte, flags, gte, "fnr", max_order, min_size),
        "calibration": A.audit(yte, scores, gte, "calibration", max_order, min_size),
    }
    if ds.need is not None:
        need = ds.need[te]
        y_need = (need >= HIGH_NEED).astype(int)
        out["high_need_threshold"] = HIGH_NEED
        out["need_fnr"] = A.audit(y_need, flags, gte, "fnr", max_order, min_size)
        pct = np.argsort(np.argsort(scores)) / (len(scores) - 1)  # risk-score percentile
        decile = np.minimum((pct * 10).astype(int), 9)
        race = gte["race"].to_numpy()
        curves = {}
        for r in ("White", "Black", "Hispanic"):
            m = race == r
            curves[r] = [float(need[m & (decile == d)].mean()) for d in range(10)]
        top = flags == 1
        out["need_by_score"] = {
            "deciles": list(range(1, 11)),
            "mean_conditions": curves,
            "flagged_mean_conditions": {r: float(need[top & (race == r)].mean()) for r in curves},
            "flagged_share": {r: float(top[race == r].mean()) for r in curves},
            "high_need_share_flagged": {r: float(top[(race == r) & (y_need == 1)].mean()) for r in curves},
            "high_need_count": {r: int(((race == r) & (y_need == 1)).sum()) for r in curves},
        }
    return out
