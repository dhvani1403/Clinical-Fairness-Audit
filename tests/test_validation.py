"""Project 2 validation: the tool, fed the replication's own predictions, finds the
same worst subgroups as the replication pipeline, for 0/1 predictions, for risk scores
flagged at a fixed capacity, and for calibration."""
import json
from pathlib import Path

import pandas as pd
import pytest

from subgroup_audit.tool import run_audit

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ["sex", "race", "age_band"]


def _load(name):
    csv, res = ROOT / "examples" / f"{name}_test_predictions.csv", ROOT / "results" / f"{name}.json"
    if not (csv.exists() and res.exists()):
        pytest.skip("run scripts/run_experiments.py and scripts/make_examples.py first")
    return pd.read_csv(csv), json.loads(res.read_text())


def _same(tool, exp):
    assert tool["worst_intersection"]["subgroup"] == exp["worst_intersection"]["subgroup"]
    assert tool["worst_single"]["subgroup"] == exp["worst_single"]["subgroup"]
    assert [r["subgroup"] for r in tool["top_weighted"]] == [r["subgroup"] for r in exp["top_weighted"]]
    assert tool["max_weighted_gap"] == pytest.approx(exp["max_weighted_gap"])


@pytest.mark.parametrize("name", ["meps", "diabetes"])
def test_tool_reproduces_baseline_audit(name):
    df, expected = _load(name)
    _same(run_audit(df, "label", "prediction", ATTRS, expected["metric"], 3, 30), expected["models"]["logistic"])


@pytest.mark.parametrize("name", ["meps", "diabetes"])
def test_tool_reproduces_capacity_and_calibration_audits(name):
    df, expected = _load(name)
    c = expected["clinical"]
    _same(run_audit(df, "label", "risk_score", ATTRS, "fnr", 3, 30, flag_top_pct=100 * c["capacity"]),
          c["capacity_fnr"])
    _same(run_audit(df, "label", "risk_score", ATTRS, "calibration", 3, 30), c["calibration"])
