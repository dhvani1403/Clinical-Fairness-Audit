import io

import numpy as np
import json

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def _post(df, **params):
    data = {"label_col": "label", "pred_col": "prediction", "protected_cols": "sex,race",
            "metric": "fpr", "max_order": 2, "use_llm": False, **params}
    return client.post("/api/audit", files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")}, data=data)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_audit_roundtrip_and_pdf(xor_data):
    r = _post(xor_data)
    assert r.status_code == 200
    body = r.json()
    assert body["worst_intersection"]["gap"] == pytest.approx(0.15) or body["worst_intersection"]["gap"] == pytest.approx(-0.15)
    assert body["summary"]["source"] == "template" and body["summary"]["verified"]
    json.dumps(body)  # fully serialisable
    pdf = client.post("/api/report/pdf", json={"result": body})
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


@pytest.mark.parametrize("params,msg", [
    ({"protected_cols": "sex,missing_col"}, "columns not found"),
    ({"metric": "precision"}, "metric must be one of"),
    ({"pred_col": "x1"}, "must be 0/1"),
    ({"protected_cols": ""}, "at least one"),
])
def test_validation_errors(xor_data, params, msg):
    r = _post(xor_data, **params)
    assert r.status_code == 422 and msg in r.json()["detail"]


def test_bad_csv():
    r = client.post("/api/audit", files={"file": ("d.csv", b"\x00\x01garbage", "text/csv")},
                    data={"label_col": "a", "pred_col": "b", "protected_cols": "c"})
    assert r.status_code == 422


def test_continuous_protected_attribute_is_binned(xor_data):
    df = xor_data.assign(age=range(len(xor_data)))
    r = _post(df, protected_cols="sex,age")
    assert r.status_code == 200
    assert any("age=" in g["subgroup"] for g in r.json()["single_attribute"])


def test_risk_scores_with_capacity(xor_data):
    df = xor_data.assign(score=np.linspace(0, 1, len(xor_data)))
    r = _post(df, pred_col="score", metric="fnr", flag_top_pct=20)
    assert r.status_code == 200
    body = r.json()
    assert body["flag_top_pct"] == 20
    # 20% flagged; half the rows are positives, so at least 60% of them are missed
    assert body["overall"]["positive_rate"] == pytest.approx(0.2, abs=0.001)
    assert body["overall_rate"] >= 0.6


def test_calibration_on_scores_ignores_capacity(xor_data):
    df = xor_data.assign(score=0.5)
    r = _post(df, pred_col="score", metric="calibration", flag_top_pct=20)
    assert r.status_code == 200 and r.json()["flag_top_pct"] is None


def test_capacity_out_of_range(xor_data):
    r = _post(xor_data.assign(score=0.5), pred_col="score", metric="fnr", flag_top_pct=150)
    assert r.status_code == 422


def test_examples_endpoint():
    ex = client.get("/api/examples").json()
    assert {e["id"] for e in ex} <= {"meps", "diabetes"}
    if ex:
        r = client.get(f"/api/examples/{ex[0]['id']}.csv")
        assert r.status_code == 200 and r.text.startswith("label,prediction,risk_score")
    assert client.get("/api/examples/../../.env.csv").status_code == 404


def test_rows_have_readable_labels(xor_data):
    body = _post(xor_data).json()
    assert body["worst_intersection"]["label"] in ("Black men", "White women", "Black women", "White men")
