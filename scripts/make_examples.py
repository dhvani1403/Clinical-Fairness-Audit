"""Write example prediction files for the audit tool (baseline logistic regression,
held-out test rows, same split as the replication).

    python scripts/make_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subgroup_audit import data  # noqa: E402
from subgroup_audit.experiments import CONFIG, logistic  # noqa: E402


def main() -> None:
    (ROOT / "examples").mkdir(exist_ok=True)
    for name in ("adult", "diabetes"):
        ds = data.load(name)
        pos_cost = CONFIG[name]["positive_cost"] or float((1 - ds.y.mean()) / ds.y.mean())
        idx = np.arange(len(ds.y))
        tr, te = train_test_split(idx, test_size=0.3, random_state=0, stratify=ds.y)
        model = logistic(pos_cost).fit(ds.X.iloc[tr], ds.y[tr])
        out = ds.groups.iloc[te].reset_index(drop=True).copy()
        out.insert(0, "label", ds.y[te])
        out.insert(1, "prediction", model.predict(ds.X.iloc[te]))
        path = ROOT / "examples" / f"{name}_test_predictions.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path.relative_to(ROOT)} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
