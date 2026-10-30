"""Eval 2A: does the summary checker catch errors an LLM might make?

Start from correct summaries of real audits, inject one error of a known type, and see
whether the checker rejects it. Compares the original numbers-only check with the
current check (numbers + which group each number belongs to). No LLM is called.

    python evals/verifier_robustness.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subgroup_audit.report import (  # noqa: E402
    _NUM, check_summary, facts_from_audit, template_summary, verify_numbers)
from subgroup_audit.tool import run_audit  # noqa: E402

ATTRS = ["sex", "race", "age_band"]
SETTINGS = [("fnr", 10), ("fnr", 20), ("fnr", 30), ("fpr", 20), ("calibration", None),
            ("positive_rate", 20), ("error_rate", 20)]
RACES = ["White", "Black", "Hispanic", "Asian"]


def base_cases() -> list[tuple[str, dict, str]]:
    """(case id, facts, correct summary) for every dataset x setting."""
    cases = []
    for name in ("meps", "diabetes"):
        df = pd.read_csv(ROOT / "examples" / f"{name}_test_predictions.csv")
        for metric, cap in SETTINGS:
            for order in (2, 3):
                facts = facts_from_audit(run_audit(df, "label", "risk_score", ATTRS, metric, order, 30, cap))
                cases.append((f"{name}/{metric}/top{cap}/order{order}", facts, template_summary(facts)))
    llm_cache = ROOT / "results" / "evals" / "llm_summaries.json"
    if llm_cache.exists():  # also perturb real LLM summaries that passed the checker
        for r in json.loads(llm_cache.read_text())["rows"]:
            if r.get("text") and r["grades"]["verified"]:
                cases.append((f"llm/{r['case']}/{r['model']}", r["facts"], r["text"]))
    return cases


def _numbers(text):
    return [m for m in _NUM.finditer(text)]


def perturb(text: str, facts: dict, kind: str, rng: random.Random) -> str | None:
    # decimals that actually change when rounded (67.0 -> 67 is the same value, not an error)
    nums = [m for m in _numbers(text) if "." in m.group() and float(m.group()) != round(float(m.group()))]
    if kind == "changed_number" and nums:
        m = rng.choice(nums)
        x = float(m.group()) + rng.choice([-1, 1]) * rng.choice([0.1, 0.3, 1.0, 2.5, 5.0])
        return text[:m.start()] + f"{x:.1f}" + text[m.end():]
    if kind == "rounded" and nums:
        m = rng.choice(nums)
        return text[:m.start()] + str(round(float(m.group()))) + text[m.end():]
    if kind == "invented_statistic":
        return text + f" About {rng.choice([12.5, 18.3, 27.0, 64.2])}% of these patients were also re-admitted."
    if kind == "wrong_group":
        present = [r for r in RACES if re.search(rf"\b{r}\b", text)]
        if present:
            r = rng.choice(present)
            other = rng.choice([x for x in RACES if x != r and not re.search(rf"\b{x}\b", text)] or
                               [x for x in RACES if x != r])
            return re.sub(rf"\b{r}\b", other, text, count=1)
        swap = {"women": "men", "men": "women"}
        m = re.search(r"\b(women|men)\b", text)
        if m:
            return text[:m.start()] + swap[m.group()] + text[m.end():]
        m = re.search(r"aged (\d+)–(\d+)", text)
        if m:
            return text[:m.start()] + "aged 65 and over" + text[m.end():]
    if kind == "unchanged":
        return text
    return None


KINDS = ["unchanged", "changed_number", "rounded", "invented_statistic", "wrong_group"]


def main() -> None:
    rng = random.Random(0)
    rows = []
    for case, facts, text in base_cases():
        for kind in KINDS:
            for rep in range(3 if kind not in ("unchanged", "invented_statistic") else 1):
                bad = perturb(text, facts, kind, rng)
                if bad is None or (kind != "unchanged" and bad == text):
                    continue
                rows.append({
                    "case": case, "kind": kind, "rep": rep,
                    "numbers_only_rejects": bool(verify_numbers(bad, facts)),
                    "binding_rejects": not check_summary(bad, facts)["verified"],
                })
    df = pd.DataFrame(rows)
    table = df.groupby("kind")[["numbers_only_rejects", "binding_rejects"]].mean().reindex(KINDS)
    table["n"] = df.groupby("kind").size().reindex(KINDS)
    print(table.round(3).to_string())
    out = ROOT / "results" / "evals" / "verifier_robustness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": table.reset_index().to_dict("records"), "rows": rows}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
