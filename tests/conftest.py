import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture
def xor_data():
    """Classic fairness-gerrymandering construction.

    Among true negatives, FPR is 40% for (Male, Black) and (Female, White) and 10% for
    (Male, White) and (Female, Black). Every single attribute therefore has FPR 25%,
    but intersections differ by 30 points.
    """
    rng = np.random.default_rng(0)
    n_cell = 400
    rows = []
    for sex in ("Male", "Female"):
        for race in ("White", "Black"):
            fpr = 0.4 if (sex == "Male") == (race == "Black") else 0.1
            k = int(fpr * n_cell)
            preds = np.r_[np.ones(k), np.zeros(n_cell - k)]
            for p in preds:
                rows.append({"sex": sex, "race": race, "label": 0, "prediction": int(p)})
            for _ in range(n_cell):  # positives, all predicted correctly
                rows.append({"sex": sex, "race": race, "label": 1, "prediction": 1})
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    df["x1"] = rng.normal(size=len(df)) + df["label"]
    return df
