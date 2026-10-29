import numpy as np

from subgroup_audit.audit import audit
from subgroup_audit.report import facts_from_audit, pdf_report, summarize, template_summary, verify_numbers


def _result(xor_data):
    return audit(xor_data["label"].to_numpy(), xor_data["prediction"].to_numpy(),
                 xor_data[["sex", "race"]], "fpr", 2)


def test_verifier_accepts_facts_and_rejects_invented_numbers(xor_data):
    facts = facts_from_audit(_result(xor_data))
    assert verify_numbers("Overall the false positive rate is 25.0% across 3,200 rows.", facts) == []
    assert verify_numbers("The worst group's rate is 41.3%.", facts) == ["41.3"]
    # "15" and "15.0" are the same value; a differently rounded value is not
    assert verify_numbers("The gap is 15 percentage points.", facts) == []
    assert verify_numbers("The gap is 14.9 percentage points.", facts) == ["14.9"]


def test_template_summary_is_fully_verified(xor_data):
    facts = facts_from_audit(_result(xor_data))
    text = template_summary(facts)
    assert verify_numbers(text, facts) == []
    assert "15.0" in text


def test_summarize_rejects_llm_text_with_unverified_numbers(monkeypatch, xor_data):
    import subgroup_audit.report as R
    monkeypatch.setattr(R, "llm_summary", lambda facts, model=None: "Black men see a 99% false positive rate.")
    s = summarize(_result(xor_data), use_llm=True)
    assert s["source"] == "template" and s["unverified_numbers"] == ["99"]
    monkeypatch.setattr(R, "llm_summary", lambda facts, model=None: "The overall rate is 25.0%.")
    assert summarize(_result(xor_data), use_llm=True)["source"] == "llm"


def test_summarize_survives_llm_failure(monkeypatch, xor_data):
    import subgroup_audit.report as R

    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(R, "llm_summary", boom)
    s = summarize(_result(xor_data), use_llm=True)
    assert s["source"] == "template" and "ConnectionError" in s["note"]


def test_pdf_is_generated(xor_data):
    res = _result(xor_data)
    pdf = pdf_report(res, summarize(res, use_llm=False))
    assert pdf[:4] == b"%PDF" and len(pdf) > 5000
