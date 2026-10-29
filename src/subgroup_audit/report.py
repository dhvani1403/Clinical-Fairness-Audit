"""Plain-English summaries and PDF reports for an audit result.

Trust model for the LLM summary:

1. the model only ever sees ``facts`` -- a small dict of numbers already computed
   by the audit (no raw data, no rows);
2. every number in its answer is extracted and checked against ``facts``; one
   number that is not there and the LLM text is rejected in favour of a
   deterministic template;
3. the UI and PDF always print the computed figures next to the text.
"""
from __future__ import annotations

import io
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from . import plots  # noqa: E402

DEFAULT_MODEL = os.environ.get("AUDIT_LLM_MODEL", "claude-opus-5-5")
_NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")


def _pct(x: float) -> float:
    return round(100 * x, 1)


def facts_from_audit(result: dict) -> dict:
    """The only information the LLM receives. Rates are percentages, 1 decimal."""
    ws, wi = result.get("worst_single"), result.get("worst_intersection")

    def grp(r):
        if not r:
            return None
        return {"subgroup": r["subgroup"], "rate_pct": _pct(r["rate"]), "gap_pct_points": _pct(r["gap"]),
                "rows_in_metric_base": r["base_size"], "share_of_data_pct": _pct(r["share"])}

    ratio = result.get("gerrymandering_ratio")
    return {
        "metric": result["metric_label"],
        "rows": result["n_rows"],
        "subgroups_checked": result["n_subgroups_checked"],
        "min_rows_for_reliable_gap": result["min_size"],
        "overall_accuracy_pct": _pct(result["overall"]["accuracy"]),
        "overall_metric_pct": _pct(result["overall_rate"]),
        "worst_single_attribute_group": grp(ws),
        "worst_intersectional_group": grp(wi),
        "intersection_vs_single_gap_ratio": round(ratio, 1) if ratio else None,
        "largest_size_weighted_gap_pct_points": round(100 * result["max_weighted_gap"], 2),
        "top_intersectional_groups": [grp(r) for r in result["top_raw_gap"] if r["order"] >= 2][:3],
    }


def _numbers_in(obj) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _numbers_in(v)
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (int, float)):
        out |= {float(obj), abs(float(obj))}
    elif isinstance(obj, str):
        out |= {abs(float(m.replace(",", ""))) for m in _NUM.findall(obj)}
    return out


def verify_numbers(text: str, facts: dict) -> list[str]:
    """Return every number in ``text`` that does not appear in ``facts`` (empty = verified)."""
    allowed = _numbers_in(facts)
    bad = []
    for m in _NUM.findall(text):
        x = abs(float(m.replace(",", "")))
        if not any(abs(x - a) < 1e-9 for a in allowed):
            bad.append(m)
    return bad


def template_summary(facts: dict) -> str:
    ws, wi = facts["worst_single_attribute_group"], facts["worst_intersectional_group"]
    parts = [f"Across {facts['rows']:,} rows the model's overall {facts['metric']} is "
             f"{facts['overall_metric_pct']}% and its accuracy is {facts['overall_accuracy_pct']}%."]
    if ws:
        parts.append(f"Looking one attribute at a time, the largest difference is for {ws['subgroup']}: "
                     f"{ws['rate_pct']}%, a gap of {ws['gap_pct_points']} percentage points.")
    if wi:
        parts.append(f"Looking at combinations of attributes, {wi['subgroup']} has a {facts['metric']} of "
                     f"{wi['rate_pct']}%, a gap of {wi['gap_pct_points']} percentage points "
                     f"(based on {wi['rows_in_metric_base']:,} rows).")
    if facts["intersection_vs_single_gap_ratio"] and facts["intersection_vs_single_gap_ratio"] > 1.5:
        parts.append(f"That is {facts['intersection_vs_single_gap_ratio']} times the largest single-attribute "
                     "gap, so checking attributes one at a time would understate the problem.")
    parts.append(f"Only groups with at least {facts['min_rows_for_reliable_gap']} rows are reported; "
                 "gaps in small groups are noisy, so check the confidence intervals before acting.")
    return " ".join(parts)


SYSTEM_PROMPT = """You explain fairness audit results to non-technical readers (managers, clinicians, policy staff).
You receive a JSON object of results that were already computed. Write 120-180 words of plain English:
what the model's overall error looks like, which single-attribute group fares worst, which combination of
attributes fares worst, and whether looking only at single attributes would hide the problem.

Rules:
- Use only numbers that appear in the JSON, copied exactly as written (same rounding). Do not compute,
  convert or round any number yourself, and write no other digits (spell out small counts in words).
- Do not speculate about causes, intent or legal compliance.
- Mention that small groups give noisy estimates.
- No headings, bullet points or markdown."""


def llm_summary(facts: dict, model: str = DEFAULT_MODEL) -> str:
    import json

    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": json.dumps(facts, indent=1)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model declined to summarise")
    return "".join(b.text for b in response.content if b.type == "text").strip()


def summarize(result: dict, use_llm: bool | None = None, model: str = DEFAULT_MODEL) -> dict:
    """Summary text plus provenance: ``source`` is "llm" or "template"."""
    facts = facts_from_audit(result)
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    out = {"facts": facts, "source": "template", "verified": True, "unverified_numbers": [], "note": None}
    if use_llm:
        try:
            text = llm_summary(facts, model)
            bad = verify_numbers(text, facts)
            if not bad:
                return {**out, "text": text, "source": "llm"}
            out["unverified_numbers"] = bad
            out["note"] = "LLM summary rejected: it contained numbers not in the audit results."
        except Exception as exc:  # network, auth, refusal: fall back, never fail the audit
            out["note"] = f"LLM unavailable ({type(exc).__name__}); showing template summary."
    out["text"] = template_summary(facts)
    out["verified"] = not verify_numbers(out["text"], facts)
    return out


def pdf_report(result: dict, summary: dict, title: str = "Subgroup fairness audit") -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, title, fontsize=18, fontweight="bold", color=plots.INK)
        fig.text(0.07, 0.925, f"Metric: {result['metric_label']}  ·  {result['n_rows']:,} rows  ·  "
                 f"{result['n_subgroups_checked']} subgroups checked", fontsize=9.5, color=plots.INK2)
        src = "AI-written, every number verified against the audit" if summary["source"] == "llm" \
            else "template summary"
        fig.text(0.07, 0.89, f"Summary ({src})", fontsize=12, fontweight="bold", color=plots.INK)
        body = "\n".join(textwrap.wrap(summary["text"], 95))
        fig.text(0.07, 0.875, body, fontsize=9.5, va="top", color=plots.INK, linespacing=1.5)

        rows = [["Subgroup", "Rate", "Gap (pp)", "95% CI ±", "Rows", "Weighted gap"]]
        for r in result["top_raw_gap"][:12]:
            rows.append([r["subgroup"][:48], f"{100 * r['rate']:.1f}%", f"{100 * r['gap']:+.1f}",
                         f"{196 * r['std_err']:.1f}", f"{r['base_size']:,}", f"{r['weighted_gap']:.4f}"])
        fig.text(0.07, 0.55, "Largest reliable gaps (computed figures)", fontsize=12,
                 fontweight="bold", color=plots.INK)
        ax = fig.add_axes([0.07, 0.18, 0.86, 0.36])
        ax.axis("off")
        tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper left", cellLoc="left",
                       colWidths=[0.44, 0.1, 0.1, 0.1, 0.1, 0.13])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (i, _), cell in tbl.get_celld().items():
            cell.set_edgecolor(plots.GRID)
            if i == 0:
                cell.set_text_props(fontweight="bold")
        fig.text(0.07, 0.1, "Gap = subgroup rate minus overall rate. Weighted gap = gap × share of rows in the "
                 "metric's base population\n(Kearns et al., 2018). Groups smaller than the reliability threshold "
                 "are excluded from the table.", fontsize=8, color=plots.MUTED)
        pdf.savefig(fig)
        plt.close(fig)
        fig = plots.gap_panels(result, "Single attributes vs. intersections")
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()
