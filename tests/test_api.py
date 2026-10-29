import io
import json

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def _post(df, **params):
    data = {"label_col": "label", "pred_col": "prediction", "protected_cols": "sex,race",
            "metric": "fpr", "max_order": 2, "use_llm": False, **params}
    return client.post("/audit", files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")}, data=data)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_audit_roundtrip_and_pdf(xor_data):
    r = _post(xor_data)
    assert r.status_code == 200
    body = r.json()
    assert body["worst_intersection"]["gap"] == pytest.approx(0.15) or body["worst_intersection"]["gap"] == pytest.approx(-0.15)
    assert body["summary"]["source"] == "template" and body["summary"]["verified"]
    json.dumps(body)  # fully serialisable
    pdf = client.post("/report/pdf", json={"result": body})
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
    r = client.post("/audit", files={"file": ("d.csv", b"\x00\x01garbage", "text/csv")},
                    data={"label_col": "a", "pred_col": "b", "protected_cols": "c"})
    assert r.status_code == 422


def test_continuous_protected_attribute_is_binned(xor_data):
    df = xor_data.assign(age=range(len(xor_data)))
    r = _post(df, protected_cols="sex,age")
    assert r.status_code == 200
    assert any("age=" in g["subgroup"] for g in r.json()["single_attribute"])
