"""Streamlit UI for the subgroup fairness audit.

If API_URL is set (e.g. in docker-compose) the UI calls the FastAPI backend;
otherwise it runs the same audit code in-process, which is what the single-container
free-hosting demo uses.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subgroup_audit import plots  # noqa: E402
from subgroup_audit.metrics import METRIC_LABELS  # noqa: E402
from subgroup_audit.report import pdf_report, summarize  # noqa: E402
from subgroup_audit.tool import AuditInputError, run_audit  # noqa: E402

API_URL = os.environ.get("API_URL", "").rstrip("/")
EXAMPLES = {
    "Adult income (logistic regression, test set)": ROOT / "examples" / "adult_test_predictions.csv",
    "Diabetes readmission (logistic regression, test set)": ROOT / "examples" / "diabetes_test_predictions.csv",
}

st.set_page_config(page_title="Subgroup Fairness Audit", layout="wide")
st.title("Subgroup Fairness Audit")
st.caption("Find where a model is unfair to the groups no one checked: single attributes *and* their "
           "intersections, scored the way Kearns, Neel, Roth & Wu (2018) define subgroup fairness.")


def call_api(raw: bytes, params: dict) -> dict:
    import requests

    r = requests.post(f"{API_URL}/audit", files={"file": ("data.csv", raw, "text/csv")},
                      data={**params, "protected_cols": ",".join(params["protected_cols"])}, timeout=300)
    if r.status_code != 200:
        raise AuditInputError(r.json().get("detail", r.text))
    return r.json()


# ---------------------------------------------------------------- 1. data
st.header("1 · Upload predictions")
source = st.radio("Data", ["Upload a CSV", *EXAMPLES], horizontal=True, label_visibility="collapsed")
if source == "Upload a CSV":
    up = st.file_uploader("CSV with one row per person: true outcome, model prediction, and the attributes "
                          "to check", type="csv")
    raw = up.getvalue() if up else None
else:
    raw = EXAMPLES[source].read_bytes()
if raw is None:
    st.info("Upload a file, or pick an example above to see how it works.")
    st.stop()
df = pd.read_csv(io.BytesIO(raw))
st.dataframe(df.head(), use_container_width=True)

# ---------------------------------------------------------------- 2. settings
st.header("2 · Settings")
cols = list(df.columns)
c1, c2, c3 = st.columns(3)
label_col = c1.selectbox("True outcome column", cols, index=cols.index("label") if "label" in cols else 0)
pred_col = c2.selectbox("Prediction column", cols, index=cols.index("prediction") if "prediction" in cols else 1)
metric = c3.selectbox("Fairness metric", list(METRIC_LABELS), format_func=lambda m: METRIC_LABELS[m])
rest = [c for c in cols if c not in (label_col, pred_col)]
protected = st.multiselect("Protected attributes", rest, default=rest[:3])
c4, c5, c6 = st.columns(3)
max_order = c4.select_slider("Combine up to … attributes", [1, 2, 3], value=3)
min_size = c5.number_input("Minimum rows for a reliable gap", 10, 1000, 30, step=10)
use_llm = c6.toggle("Plain-English summary by Claude", value=bool(os.environ.get("ANTHROPIC_API_KEY")) or bool(API_URL))

if not st.button("Run audit", type="primary"):
    st.stop()

params = {"label_col": label_col, "pred_col": pred_col, "protected_cols": protected, "metric": metric,
          "max_order": max_order, "min_size": int(min_size), "use_llm": use_llm}
try:
    with st.spinner("Searching subgroups…"):
        if API_URL:
            result = call_api(raw, params)
        else:
            result = run_audit(df, label_col, pred_col, protected, metric, max_order, int(min_size))
            result["summary"] = summarize(result, use_llm=None if use_llm else False)
except AuditInputError as exc:
    st.error(str(exc))
    st.stop()

# ---------------------------------------------------------------- 3. report
st.header("3 · Report")
s = result["summary"]
o = result["overall"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{result['n_rows']:,}")
m2.metric("Accuracy", f"{100 * o['accuracy']:.1f}%")
m3.metric(f"Overall {result['metric_label']}", f"{100 * result['overall_rate']:.1f}%")
ratio = result["gerrymandering_ratio"]
m4.metric("Worst intersection ÷ worst single gap", f"{ratio:.1f}×" if ratio else "–",
          help="Above ~1.5× means checking one attribute at a time hides unfairness.")

st.subheader("Summary")
badge = ("✅ Written by Claude; every number checked against the computed results"
         if s["source"] == "llm" else "Template summary (no LLM)")
st.caption(badge + (f" · {s['note']}" if s.get("note") else ""))
st.write(s["text"])
with st.expander("Figures the summary was allowed to use"):
    st.json(s["facts"])

st.subheader("Single attributes vs. intersections")
st.pyplot(plots.gap_panels(result, f"Gaps in {result['metric_label']} (95% intervals)"))

def table(rows):
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.assign(rate=100 * t["rate"], gap=100 * t["gap"], ci95=196 * t["std_err"])
    return t[["subgroup", "base_size", "rate", "gap", "ci95", "weighted_gap", "reliable"]].rename(columns={
        "base_size": "rows in base", "rate": "rate %", "gap": "gap (pp)", "ci95": "± 95% CI (pp)"})

t1, t2, t3 = st.tabs(["Largest gaps", "Largest size-weighted gaps", "Single attributes"])
t1.dataframe(table(result["top_raw_gap"]), use_container_width=True)
t2.dataframe(table(result["top_weighted"]), use_container_width=True)
t3.dataframe(table(result["single_attribute"]), use_container_width=True)

st.download_button("Download PDF report", pdf_report(result, s), "fairness_audit.pdf", "application/pdf")
st.caption("Limitations: gaps are measured on the data you upload; small groups are noisy (see intervals); "
           "a gap is evidence to investigate, not proof of discrimination.")
