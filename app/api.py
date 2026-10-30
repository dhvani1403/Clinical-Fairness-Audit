"""FastAPI backend: validate an upload, run the subgroup audit, return JSON or a PDF.
Also serves the built web app (frontend/dist) at /.

    uvicorn app.api:app --reload
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subgroup_audit.env import load_dotenv  # noqa: E402

load_dotenv()

from subgroup_audit.report import pdf_report, summarize  # noqa: E402
from subgroup_audit.tool import AuditInputError, run_audit  # noqa: E402

MAX_BYTES = 50 * 1024 * 1024

app = FastAPI(title="Clinical Subgroup Fairness Audit API", version="1.1.0")


EXAMPLES = {
    "meps": {"title": "MEPS 2015: high healthcare use",
             "description": "7,674 adults; outcome is 10 or more visits in the year."},
    "diabetes": {"title": "Diabetes: 30-day readmission",
                 "description": "20,991 hospital patients; outcome is readmission within 30 days."},
}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/examples")
def list_examples():
    return [{"id": k, **v, "columns": {"label": "label", "score": "risk_score",
                                         "protected": ["sex", "race", "age_band"]}}
            for k, v in EXAMPLES.items() if (ROOT / "examples" / f"{k}_test_predictions.csv").exists()]


@app.get("/api/examples/{name}.csv")
def get_example(name: str):
    path = ROOT / "examples" / f"{name}_test_predictions.csv"
    if name not in EXAMPLES or not path.exists():
        raise HTTPException(404, "no such example")
    return FileResponse(path, media_type="text/csv")


@app.post("/api/audit")
async def audit_endpoint(
    file: UploadFile = File(..., description="CSV with outcomes, predictions or risk scores, and protected attributes"),
    label_col: str = Form(...),
    pred_col: str = Form(...),
    protected_cols: str = Form("", description="comma-separated column names"),
    metric: str = Form("fpr"),
    max_order: int = Form(3),
    min_size: int = Form(30),
    use_llm: bool = Form(True),
    flag_top_pct: float | None = Form(None, description="treat predictions as risk scores; flag the top X%"),
):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "file is larger than 50 MB")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, f"could not read CSV: {exc}") from exc
    cols = [c.strip() for c in protected_cols.split(",") if c.strip()]
    try:
        result = run_audit(df, label_col, pred_col, cols, metric, max_order, min_size, flag_top_pct)
    except AuditInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    # None = use the LLM only if an API key is configured.
    result["summary"] = summarize(result, use_llm=None if use_llm else False)
    return result


class ReportRequest(BaseModel):
    result: dict
    title: str = "Subgroup fairness audit"


@app.post("/api/report/pdf")
def report_pdf(req: ReportRequest):
    summary = req.result.get("summary") or summarize(req.result, use_llm=False)
    pdf = pdf_report(req.result, summary, req.title)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="fairness_audit.pdf"'})


# The web app, if it has been built (npm run build in frontend/). Mounted last so the
# API routes above take precedence.
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
