"""Eval 1: does the audit find unfairness that is really there, and stay quiet when it isn't?

We simulate patient populations, plant a miss-rate (FNR) increase of known size in a
known subgroup, run the audit, and check what it reports. Four designs:

* null           -- no bias anywhere (measures false alarms);
* intersection-2 -- +delta for Hispanic patients 65+ (~5% of patients);
* intersection-3 -- +delta for Black women aged 18-39 (~2% of patients);
* gerrymandered  -- +delta for Black women, -delta for Black men and smaller
                    opposite effects for White patients, tuned so every single
                    attribute is exactly fair in expectation.

    python evals/audit_detection.py            # ~3 minutes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subgroup_audit import plots  # noqa: E402
from subgroup_audit.audit import audit  # noqa: E402

SEX = (["Female", "Male"], [0.5, 0.5])
RACE = (["White", "Black", "Hispanic", "Asian"], [0.60, 0.18, 0.15, 0.07])
AGE = (["18-39", "40-64", "65+"], [0.25, 0.40, 0.35])
BASE_FNR, POS_RATE = 0.30, 0.25
DESIGNS = {
    "null": {},
    "intersection-2": {("race=Hispanic", "age_band=65+"): 1.0},
    "intersection-3": {("sex=Female", "race=Black", "age_band=18-39"): 1.0},
    # sex and race are independent, so with these weights every marginal effect is zero:
    # F: .09(+1) + .30(-.3) = 0; B: .09(+1) + .09(-1) = 0; W: .30(-.3) + .30(+.3) = 0.
    "gerrymandered": {("sex=Female", "race=Black"): 1.0, ("sex=Male", "race=Black"): -1.0,
                      ("sex=Female", "race=White"): -0.3, ("sex=Male", "race=White"): 0.3},
}
TARGET = {  # the planted group a correct audit should report (or a subset of it)
    "intersection-2": [("race=Hispanic", "age_band=65+")],
    "intersection-3": [("sex=Female", "race=Black", "age_band=18-39")],
    "gerrymandered": [("sex=Female", "race=Black"), ("sex=Male", "race=Black")],
}
DELTAS = [0.0, 0.05, 0.10, 0.15, 0.20]
SIZES = [5_000, 20_000]
REPS = 50


def simulate(n: int, design: str, delta: float, rng: np.random.Generator):
    g = pd.DataFrame({
        "sex": rng.choice(SEX[0], n, p=SEX[1]),
        "race": rng.choice(RACE[0], n, p=RACE[1]),
        "age_band": rng.choice(AGE[0], n, p=AGE[1]),
    })
    y = (rng.random(n) < POS_RATE).astype(int)
    miss = np.full(n, BASE_FNR)
    for cell, weight in DESIGNS[design].items():
        mask = np.ones(n, bool)
        for cond in cell:
            a, v = cond.split("=")
            mask &= g[a].to_numpy() == v
        miss[mask] += weight * delta
    pred = np.where(y == 1, (rng.random(n) >= miss).astype(int), 0)
    return y, pred, g


def in_target(subgroup: str, design: str) -> bool:
    conds = set(subgroup.split(" & "))
    return any(set(t) <= conds for t in TARGET.get(design, []))


def run_one(n, design, delta, rng) -> dict:
    y, p, g = simulate(n, design, delta, rng)
    res = audit(y, p, g, "fnr", max_order=3, min_size=30, top_k=10)
    rows = res["top_raw_gap"]
    single_sig = any(r["significant"] for r in res["single_attribute"])
    any_sig = any(r["significant"] for r in res["top_raw_gap"]) or single_sig
    any_naive = any(abs(r["z"]) > 1.96 for r in res["top_raw_gap"] + res["single_attribute"] if r["reliable"])
    target_sig = any(r["significant"] and in_target(r["subgroup"], design) for r in rows)
    wi = res["worst_intersection"]
    return {
        "hit_at_1": bool(wi and in_target(wi["subgroup"], design)),
        "hit_at_1_weighted": in_target(res["top_weighted"][0]["subgroup"], design),
        "detected": target_sig,
        "single_attribute_flag": single_sig,
        "false_alarm_bh": any_sig,
        "false_alarm_uncorrected": any_naive,
    }


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for design in DESIGNS:
        for n in SIZES:
            for delta in DELTAS if design != "null" else [0.0]:
                runs = [run_one(n, design, delta, rng) for _ in range(REPS)]
                agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
                rows.append({"design": design, "n": n, "delta": delta, **agg})
                print(f"{design:15s} n={n:6d} delta={delta:.2f} " +
                      " ".join(f"{k}={v:.2f}" for k, v in agg.items()))
    out_dir = ROOT / "results" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_detection.json").write_text(json.dumps({"reps": REPS, "rows": rows}, indent=1))
    fig = plots.detection_curves(rows, DELTAS, SIZES)
    fig.savefig(ROOT / "figures" / "eval_audit_detection.png", dpi=150)
    print("wrote results/evals/audit_detection.json and figures/eval_audit_detection.png")


if __name__ == "__main__":
    main()
