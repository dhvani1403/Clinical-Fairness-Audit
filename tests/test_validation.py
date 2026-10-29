"""Project 2 validation: the tool, fed the replication's own predictions, finds the
same worst subgroups as the replication pipeline."""
import json
from pathlib import Path

import pandas as pd
import pytest

from subgroup_audit.tool import run_audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["adult", "diabetes"])
def test_tool_reproduces_replication(name):
    csv, res = ROOT / "examples" / f"{name}_test_predictions.csv", ROOT / "results" / f"{name}.json"
    if not (csv.exists() and res.exists()):
        pytest.skip("run scripts/run_experiments.py and scripts/make_examples.py first")
    expected = json.loads(res.read_text())
    tool = run_audit(pd.read_csv(csv), "label", "prediction", ["sex", "race", "age_band"],
                     expected["metric"], 3, 30)
    exp = expected["models"]["logistic"]
    assert tool["worst_intersection"]["subgroup"] == exp["worst_intersection"]["subgroup"]
    assert tool["worst_single"]["subgroup"] == exp["worst_single"]["subgroup"]
    assert [r["subgroup"] for r in tool["top_weighted"]] == [r["subgroup"] for r in exp["top_weighted"]]
    assert tool["max_weighted_gap"] == pytest.approx(exp["max_weighted_gap"])
