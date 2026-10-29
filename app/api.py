"""FastAPI backend: validate an upload, run the subgroup audit, return JSON or a PDF.

    uvicorn app.api:app --reload
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subgroup_audit.report import pdf_report, summarize  # noqa: E402
from subgroup_audit.tool import AuditInputError, run_audit  # noqa: E402

MAX_BYTES = 50 * 1024 * 1024

app = FastAPI(title="Clinical Subgroup Fairness Audit API", version="1.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/audit")
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


@app.post("/report/pdf")
def report_pdf(req: ReportRequest):
    summary = req.result.get("summary") or summarize(req.result, use_llm=False)
    pdf = pdf_report(req.result, summary, req.title)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="fairness_audit.pdf"'})
