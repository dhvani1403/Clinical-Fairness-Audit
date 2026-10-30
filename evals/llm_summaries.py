"""Eval 2B: how good are the LLM's plain-English summaries?

Runs the summary model on a fixed set of audits and grades each summary with code
(no LLM judge). Responses are cached in results/evals/llm_cache.json, so re-running
does not spend requests; free OpenRouter models allow only a few dozen per day.

    python evals/llm_summaries.py                          # default free model chain
    python evals/llm_summaries.py --model google/gemma-4-31b-it:free
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from subgroup_audit.env import load_dotenv  # noqa: E402

load_dotenv()

from audit_detection import simulate  # noqa: E402
from subgroup_audit import report  # noqa: E402
from subgroup_audit.audit import audit  # noqa: E402
from subgroup_audit.tool import run_audit  # noqa: E402

CACHE = ROOT / "results" / "evals" / "llm_cache.json"
OUT = ROOT / "results" / "evals" / "llm_summaries.json"
ATTRS = ["sex", "race", "age_band"]
CAVEAT = re.compile(r"small|noisy|noise|uncertain|few patients|caution|confidence|not reliable|less reliable", re.I)
FORBIDDEN = re.compile(r"discriminat|racis|illegal|unlawful|biased against|because of (?:their|race)|"
                       r"due to (?:their )?(?:race|ethnicity)|intentional", re.I)
MARKDOWN = re.compile(r"^\s*(?:#|[-*•]\s|\d+\.\s)|\*\*|__", re.M)
HIDES = re.compile(r"\bhid(?:e|es|den|ing)\b|understat|mask|miss(?:es)? the (?:real|full)", re.I)
HEDGE = re.compile(r"slight|little|modest|somewhat|part of|only|small", re.I)


def cases() -> list[tuple[str, dict]]:
    out = []
    for name, settings in (("meps", [("fnr", 20), ("fnr", 10), ("calibration", None), ("fpr", 20)]),
                           ("diabetes", [("fnr", 20), ("fnr", 10), ("calibration", None)])):
        df = pd.read_csv(ROOT / "examples" / f"{name}_test_predictions.csv")
        for metric, cap in settings:
            out.append((f"{name}/{metric}" + (f"/top{cap}" if cap else ""),
                        run_audit(df, "label", "risk_score", ATTRS, metric, 3, 30, cap)))
    rng = np.random.default_rng(1)
    for design, delta in (("gerrymandered", 0.15), ("intersection-3", 0.20), ("null", 0.0)):
        y, p, g = simulate(20_000, design, delta, rng)
        res = audit(y, p, g, "fnr", 3, 30)
        res["flag_top_pct"] = None
        out.append((f"synthetic/{design}", res))
    return out


def mentions(text: str, subgroup: str | None) -> bool:
    if not subgroup:
        return True
    pats = [report._value_pattern(a, v) for a, v in report._parse(subgroup).items()]
    return all(p.search(text) for p in pats if p is not None)


def grade(text: str, facts: dict) -> dict:
    check = report.check_summary(text, facts)
    words = len(text.split())
    ratio = facts.get("intersection_vs_single_gap_ratio")
    ws, wi = facts.get("worst_single_attribute_group"), facts.get("worst_intersectional_group")
    overclaim = bool(ratio is not None and ratio < 1.5 and HIDES.search(text) and not HEDGE.search(
        " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if HIDES.search(s))))
    return {
        "verified": check["verified"],
        "unverified_numbers": check["unverified_numbers"],
        "unbound_numbers": check["unbound_numbers"],
        "names_worst_single": mentions(text, ws and ws["subgroup"]),
        "names_worst_intersection": mentions(text, wi and wi["subgroup"]),
        "small_group_caveat": bool(CAVEAT.search(text)),
        "no_causal_or_legal_claims": not FORBIDDEN.search(text),
        "plain_text": not MARKDOWN.search(text),
        "complete": text.rstrip().endswith((".", "!", "?")),
        "length_120_180": 120 <= words <= 180,
        "words": words,
        "no_overclaim": not overclaim,
    }


CRITERIA = ["verified", "complete", "names_worst_single", "names_worst_intersection", "small_group_caveat",
            "no_causal_or_legal_claims", "plain_text", "length_120_180", "no_overclaim"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="a single ':free' model (default: the fallback chain)")
    args = ap.parse_args()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    previous = json.loads(OUT.read_text())["rows"] if OUT.exists() else []
    rows = [r for r in previous if args.model is None or r["requested_model"] != (args.model or "default")]
    for case, res in cases():
        facts = report.facts_from_audit(res)
        key = hashlib.sha256(json.dumps([args.model, report.SYSTEM_PROMPT, report.MAX_TOKENS, facts],
                                        sort_keys=True).encode()).hexdigest()
        if key not in cache:
            t = time.time()
            try:
                text, used = report.llm_summary(facts, args.model)
                cache[key] = {"text": text, "model": used, "seconds": round(time.time() - t, 1)}
            except Exception as exc:
                print(f"  {case}: {str(exc)[:150]}")
                cache[key] = {"text": None, "model": args.model, "error": str(exc)[:300]}
            CACHE.write_text(json.dumps(cache, indent=1))
        hit = cache[key]
        row = {"case": case, "requested_model": args.model or "default", "model": hit.get("model"),
               "facts": facts, "text": hit.get("text"), "error": hit.get("error"), "seconds": hit.get("seconds")}
        row["grades"] = grade(hit["text"], facts) if hit.get("text") else None
        rows = [r for r in rows if not (r["case"] == case and r["requested_model"] == row["requested_model"])]
        rows.append(row)
        g = row["grades"]
        print(f"{case:28s} {row['model'] or '-':42s} " +
              (" ".join(f"{c[:12]}={'Y' if g[c] else 'n'}" for c in CRITERIA) if g else "ERROR"))
    graded = [r for r in rows if r["grades"]]
    summary = {}
    for model in sorted({r["model"] for r in graded}):
        rs = [r for r in graded if r["model"] == model]
        summary[model] = {c: float(np.mean([r["grades"][c] for r in rs])) for c in CRITERIA}
        summary[model]["all_criteria"] = float(np.mean([all(r["grades"][c] for c in CRITERIA) for r in rs]))
        summary[model]["n"] = len(rs)
        summary[model]["median_seconds"] = float(np.median([r["seconds"] or 0 for r in rs]))
    summary["_errors"] = sum(1 for r in rows if not r["grades"])
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
