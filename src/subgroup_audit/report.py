"""Plain-English summaries and PDF reports for an audit result.

Trust model for the LLM summary:

1. the model only ever sees ``facts`` -- a small dict of numbers already computed
   by the audit (no raw data, no rows);
2. every number in its answer is extracted and checked against ``facts``, and
   must sit in the same or previous sentence as the group it belongs to (so a
   real number attached to the wrong group is caught too); one failure and the
   LLM text is rejected in favour of a deterministic template;
3. the UI and PDF always print the computed figures next to the text.

Providers: OpenRouter (default when ``OPENROUTER_API_KEY`` is set; restricted to
free ``:free`` models unless ``ALLOW_PAID_MODELS=1``) or Anthropic
(``ANTHROPIC_API_KEY``).
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

ANTHROPIC_MODEL = os.environ.get("AUDIT_LLM_MODEL", "claude-opus-5-5")
# Free models tried in order (a free model's provider is often rate-limited, so we fall back).
OPENROUTER_MODELS = os.environ.get(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free",
).split(",")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 4000  # reasoning models spend part of this budget thinking before they answer
_NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")


def _pct(x: float) -> float:
    return round(100 * x, 1) + 0.0  # + 0.0 turns -0.0 into 0.0


def facts_from_audit(result: dict) -> dict:
    """The only information the LLM receives. Rates are percentages, 1 decimal."""
    ws, wi = result.get("worst_single"), result.get("worst_intersection")

    def grp(r):
        if not r:
            return None
        return {"subgroup": r["subgroup"], "description": describe(r["subgroup"]),
                "rate_pct": _pct(r["rate"]), "gap_pct_points": _pct(r["gap"]),
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
        "flagged_top_pct": result.get("flag_top_pct"),
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


def _parse(subgroup: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in subgroup.split(" & ") if "=" in part)


def describe(subgroup: str) -> str:
    """Readable name: ``sex=Female & race=Black & age_band=<40`` -> "Black women under 40"."""
    v = _parse(subgroup)
    if not v or not set(v) <= {"sex", "race", "age_band"}:
        return subgroup
    race = v.get("race", "")
    noun = {"Female": "women", "Male": "men"}.get(v.get("sex", ""), "patients")
    words = [noun] if race in ("", "Other", "Unknown") else [race, noun]
    if race in ("Other", "Unknown"):
        words.append(f"of {race.lower()} race")
    if age := v.get("age_band"):
        m = re.fullmatch(r"(\d+)-(\d+)", age)
        words.append(f"aged {m[1]}–{m[2]}" if m else f"under {age[1:]}" if age.startswith("<")
                     else f"aged {age[:-1]} and over" if age.endswith("+") else f"aged {age}")
    return " ".join(words)


def _value_pattern(attr: str, value: str) -> re.Pattern | None:
    """Regex that recognises a subgroup value however a writer phrases it."""
    if attr == "sex":
        alts = {"Female": r"female|women|woman", "Male": r"(?<!fe)male|(?<!wo)men|(?<!wo)man"}.get(value)
        return re.compile(rf"\b(?:{alts})\b", re.I) if alts else None
    if m := re.fullmatch(r"(\d+)-(\d+)", value):
        return re.compile(rf"\b{m[1]}\s*(?:-|–|—|to)\s*{m[2]}\b")
    if m := re.fullmatch(r"<(\d+)", value):
        return re.compile(rf"(?:under|below|younger than|less than|<)\s*(?:age\s*)?{m[1]}\b", re.I)
    if m := re.fullmatch(r"(\d+)\+", value):
        return re.compile(rf"\b{m[1]}\s*(?:\+|and (?:over|older|above)|or (?:over|older))"
                          rf"|(?:over|older than|aged)\s*{m[1]}\b", re.I)
    if re.fullmatch(r"[A-Za-z][\w\- ]*", value):
        return re.compile(rf"\b{re.escape(value)}\b", re.I)
    return None  # e.g. quartile bins: cannot be phrased reliably, so not enforced


def _fact_entries(facts: dict) -> list[tuple[float, str, list[re.Pattern], bool]]:
    """(value, path, patterns the surrounding text must match, is_label) for every number in facts."""
    entries = []

    def walk(obj, path, reqs):
        if isinstance(obj, dict):
            if isinstance(obj.get("subgroup"), str):
                pats = [_value_pattern(a, v) for a, v in _parse(obj["subgroup"]).items()]
                reqs = [p for p in pats if p is not None]
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else k, reqs)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", reqs)
        elif isinstance(obj, bool) or obj is None:
            return
        elif isinstance(obj, (int, float)):
            entries.append((abs(float(obj)), path, reqs, False))
        elif isinstance(obj, str):  # numbers inside labels, e.g. "18-29": no binding needed
            entries.extend((abs(float(m.replace(",", ""))), path, [], True) for m in _NUM.findall(obj))

    walk(facts, "", [])
    return entries


def check_summary(text: str, facts: dict) -> dict:
    """Verify every number in ``text``.

    Returns ``matches`` (each number with its character span and the fact it was
    matched to), ``unverified_numbers`` (not in facts at all) and ``unbound_numbers``
    (in facts, but not next to the group they belong to).
    """
    entries = _fact_entries(facts)
    sentences, pos = [], 0
    for part in _SENTENCE.split(text):
        start = text.index(part, pos)
        sentences.append((start, start + len(part)))
        pos = start + len(part)
    out = {"matches": [], "unverified_numbers": [], "unbound_numbers": []}
    for m in _NUM.finditer(text):
        x = abs(float(m.group().replace(",", "")))
        cands = [e for e in entries if abs(e[0] - x) < 1e-9]
        if not cands:
            out["unverified_numbers"].append(m.group())
            continue
        i = next(k for k, (a, b) in enumerate(sentences) if a <= m.start() < b)
        context = text[sentences[max(i - 1, 0)][0]:sentences[i][1]]
        bound = next((e for e in cands if all(p.search(context) for p in e[2])), None)
        if bound is None:
            out["unbound_numbers"].append(m.group())
            continue
        if not bound[3]:  # digits inside a group's name ("aged 18–29") are not reported as figures
            out["matches"].append({"text": m.group(), "start": m.start(), "end": m.end(), "fact": bound[1]})
    out["verified"] = not out["unverified_numbers"] and not out["unbound_numbers"]
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
    flagged = (f" when the top {facts['flagged_top_pct']:g}% of patients by risk are flagged"
               if facts.get("flagged_top_pct") else "")
    parts = [f"Across {facts['rows']:,} patients the model's overall {facts['metric']} is "
             f"{facts['overall_metric_pct']}%{flagged}, and its accuracy is {facts['overall_accuracy_pct']}%."]
    if ws:
        parts.append(f"Looking one attribute at a time, the largest difference is for {ws['description']}: "
                     f"{ws['rate_pct']}%, a gap of {ws['gap_pct_points']} percentage points.")
    if wi:
        parts.append(f"Looking at combinations of attributes, {wi['description']} have a {facts['metric']} of "
                     f"{wi['rate_pct']}%, a gap of {wi['gap_pct_points']} percentage points "
                     f"(based on {wi['rows_in_metric_base']:,} patients).")
    if facts["intersection_vs_single_gap_ratio"] and facts["intersection_vs_single_gap_ratio"] > 1.5:
        parts.append(f"That is {facts['intersection_vs_single_gap_ratio']} times the largest single-attribute "
                     "gap, so checking attributes one at a time would understate the problem.")
    parts.append(f"Only groups with at least {facts['min_rows_for_reliable_gap']} patients are reported; "
                 "gaps in small groups are noisy, so check the confidence intervals before acting.")
    return " ".join(parts)


SYSTEM_PROMPT = """You explain fairness audits of clinical risk models to non-technical readers (clinicians,
care-management leads, hospital quality and equity staff). You receive a JSON object of results that were already
computed. Write 120-180 words of plain English: what the model's overall error looks like (and, if
"flagged_top_pct" is set, that patients are flagged by taking the highest-risk share), which single-attribute
group fares worst, which combination of attributes fares worst, and whether looking only at single attributes
would hide the problem. A false negative is a patient who had the outcome but was not flagged.

Rules:
- Refer to groups by their "description" field.
- Use only numbers that appear in the JSON, copied exactly as written (same rounding). Do not compute,
  convert or round any number yourself, and write no other digits (spell out small counts in words).
- Put each group's numbers in the same sentence as the group's description.
- Do not speculate about causes, intent or legal compliance.
- Write for the reader: never mention the JSON, field names, or information that is missing.
- Mention that small groups give noisy estimates.
- No headings, bullet points or markdown."""


def provider() -> str | None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def _openrouter(facts: dict, model: str) -> str:
    import json

    import requests

    if not model.endswith(":free") and os.environ.get("ALLOW_PAID_MODELS") != "1":
        raise ValueError(f"refusing paid model {model!r}; use a ':free' model or set ALLOW_PAID_MODELS=1")
    r = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "Clinical Fairness Audit"},
        json={"model": model, "max_tokens": MAX_TOKENS, "temperature": 0.2,
              # reasoning models: think briefly and keep the reasoning out of the answer text
              "reasoning": {"effort": "low", "exclude": True},
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": json.dumps(facts, indent=1)}]},
        timeout=180,
    )
    body = r.json()
    if r.status_code != 200 or "error" in body:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {body.get('error', {}).get('message', '')[:200]}")
    choice = body["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    # A summary cut off by the token limit can still contain only correct numbers, so the
    # number check would pass it; reject it explicitly.
    if not text or choice.get("finish_reason") == "length":
        raise RuntimeError(f"{model} returned {'a truncated' if text else 'no'} summary "
                           f"(finish_reason={choice.get('finish_reason')})")
    return text


def _anthropic(facts: dict, model: str) -> str:
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


def llm_summary(facts: dict, model: str | None = None) -> tuple[str, str]:
    """Return (text, model used)."""
    which = provider()
    if which == "openrouter":
        import requests

        errors = []
        for m in [model] if model else OPENROUTER_MODELS:
            try:
                return _openrouter(facts, m.strip()), m.strip()
            except (RuntimeError, requests.RequestException) as exc:  # rate limit, truncation, timeout
                errors.append(f"{m.strip()}: {exc}")
        raise RuntimeError("; ".join(errors))
    if which == "anthropic":
        return _anthropic(facts, model or ANTHROPIC_MODEL), model or ANTHROPIC_MODEL
    raise RuntimeError("no LLM API key configured")


def summarize(result: dict, use_llm: bool | None = None, model: str | None = None) -> dict:
    """Summary text plus provenance: ``source`` is "llm" or "template"."""
    facts = facts_from_audit(result)
    if use_llm is None:
        use_llm = provider() is not None
    out = {"facts": facts, "source": "template", "verified": True, "unverified_numbers": [],
           "unbound_numbers": [], "note": None, "model": None}
    if use_llm:
        try:
            text, used = llm_summary(facts, model)
            out["model"] = used
            check = check_summary(text, facts)
            if check["verified"]:
                return {**out, "text": text, "source": "llm", "matches": check["matches"]}
            out["unverified_numbers"] = check["unverified_numbers"]
            out["unbound_numbers"] = check["unbound_numbers"]
            out["note"] = ("AI summary rejected: it contained numbers that are not in the audit results"
                           if check["unverified_numbers"] else
                           "AI summary rejected: it attached numbers to the wrong group")
        except Exception as exc:  # network, auth, rate limit, refusal: fall back, never fail the audit
            out["note"] = f"AI summary unavailable ({type(exc).__name__}); showing template summary."
    out["text"] = template_summary(facts)
    check = check_summary(out["text"], facts)
    out["verified"], out["matches"] = check["verified"], check["matches"]
    return out


def pdf_report(result: dict, summary: dict, title: str = "Clinical fairness audit") -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, title, fontsize=18, fontweight="bold", color=plots.INK)
        fig.text(0.07, 0.925, f"Metric: {result['metric_label']}  ·  {result['n_rows']:,} patients  ·  "
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
