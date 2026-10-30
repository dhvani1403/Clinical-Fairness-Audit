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
    monkeypatch.setattr(R, "llm_summary", lambda facts, model=None: ("Black men see a 99% false positive rate.", "mock"))
    s = summarize(_result(xor_data), use_llm=True)
    assert s["source"] == "template" and s["unverified_numbers"] == ["99"]
    monkeypatch.setattr(R, "llm_summary", lambda facts, model=None: ("The overall rate is 25.0%.", "mock"))
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


def test_describe():
    from subgroup_audit.report import describe
    assert describe("sex=Female & race=Black & age_band=<40") == "Black women under 40"
    assert describe("race=Hispanic & age_band=18-29") == "Hispanic patients aged 18–29"
    assert describe("sex=Male & race=Unknown & age_band=80+") == "men of unknown race aged 80 and over"
    assert describe("pct_white=low") == "pct_white=low"


def test_number_attached_to_wrong_group_is_caught():
    from subgroup_audit.report import check_summary
    facts = {"overall_metric_pct": 43.9,
             "worst_intersectional_group": {"subgroup": "race=Hispanic & age_band=18-29", "gap_pct_points": 53.7},
             "worst_single_attribute_group": {"subgroup": "age_band=18-29", "gap_pct_points": 43.8}}
    ok = check_summary("Overall 43.9% are missed. Hispanic patients aged 18–29 are missed most, a gap of 53.7 points.", facts)
    assert ok["verified"] and {m["fact"] for m in ok["matches"]} == {
        "overall_metric_pct", "worst_intersectional_group.gap_pct_points"}
    swapped = check_summary("Overall 43.9% are missed. Black patients aged 18–29 are missed most, a gap of 53.7 points.", facts)
    assert not swapped["verified"] and swapped["unbound_numbers"] == ["53.7"]
    # the group may be named in the previous sentence
    assert check_summary("The worst group is Hispanic patients aged 18 to 29. Their gap is 53.7 points.", facts)["verified"]


def test_truncated_llm_reply_is_rejected(monkeypatch, xor_data):
    import requests

    import subgroup_audit.report as R

    class Fake:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"finish_reason": "length", "message": {"content": "The overall rate is 25.0% and"}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(R, "OPENROUTER_MODELS", ["some/model:free"])
    monkeypatch.setattr(requests, "post", lambda *a, **k: Fake())
    s = summarize(_result(xor_data), use_llm=True)
    assert s["source"] == "template" and "unavailable" in s["note"]


def test_paid_models_are_refused(monkeypatch, xor_data):
    import subgroup_audit.report as R
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.delenv("ALLOW_PAID_MODELS", raising=False)
    try:
        R._openrouter({}, "anthropic/claude-opus")
    except ValueError as e:
        assert "refusing paid model" in str(e)
    else:
        raise AssertionError("paid model was not refused")


def test_timeout_falls_back_to_next_free_model(monkeypatch):
    import requests

    import subgroup_audit.report as R
    calls = []

    def fake(facts, model):
        calls.append(model)
        if model == "a/first:free":
            raise requests.Timeout("read timed out")
        return "ok."

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(R, "OPENROUTER_MODELS", ["a/first:free", "b/second:free"])
    monkeypatch.setattr(R, "_openrouter", fake)
    assert R.llm_summary({}) == ("ok.", "b/second:free") and calls == ["a/first:free", "b/second:free"]
