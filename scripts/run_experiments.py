"""Run the full replication and write results/*.json and figures/*.png.

    python scripts/run_experiments.py                 # all datasets
    python scripts/run_experiments.py --datasets adult --iters 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subgroup_audit import data, plots  # noqa: E402
from subgroup_audit.experiments import run  # noqa: E402

PRETTY = {"communities": "Communities & Crime", "adult": "Adult income", "diabetes": "Diabetes readmission"}


def make_figures(res: dict, fig_dir: Path) -> None:
    ds, metric = res["dataset"], res["metric"].upper()
    m = res["models"]

    fig = plots.gap_panels(m["marginal_fair"], f"{PRETTY[ds]}: a model made fair on single attributes "
                                               f"({metric}, test set)")
    fig.savefig(fig_dir / f"{ds}_gerrymandering.png", dpi=150)

    fig = plots.gap_panels(m["logistic"], f"{PRETTY[ds]}: baseline logistic regression ({metric}, test set)")
    fig.savefig(fig_dir / f"{ds}_baseline_gaps.png", dpi=150)

    rows = [{"model": label, "single": abs(m[k]["worst_single"]["gap"]),
             "intersection": abs(m[k]["worst_intersection"]["gap"])}
            for k, label in (("logistic", "Logistic\nbaseline"), ("boosting", "Boosting\nbaseline"),
                             ("marginal_fair", "Fair on single\nattributes"),
                             ("subgroup_fair", "Fair on\nsubgroups"))]
    fig = plots.worst_gap_bars(rows, f"{PRETTY[ds]}: worst {metric} gaps, single attribute vs. intersection")
    fig.savefig(fig_dir / f"{ds}_worst_gaps.png", dpi=150)

    acc_key = "balanced_accuracy" if ds == "diabetes" else "accuracy"
    curves = [{"label": lbl, "x": [p["max_weighted_gap"] for p in res["tradeoff"][k]],
               "y": [p[acc_key] for p in res["tradeoff"][k]]}
              for k, lbl in (("marginal_fair", "fair on single attributes"), ("subgroup_fair", "fair on subgroups"))]
    base = {"x": m["logistic"]["max_weighted_gap"], "y": m["logistic"]["overall"][acc_key], "label": "baseline"}
    fig = plots.tradeoff(curves, base, f"{PRETTY[ds]}: accuracy vs. worst subgroup unfairness",
                         xlabel=f"worst size-weighted {metric} disparity over subgroups (test)",
                         ylabel="test balanced accuracy" if acc_key == "balanced_accuracy" else "test accuracy")
    fig.savefig(fig_dir / f"{ds}_tradeoff.png", dpi=150)

    if "paper" in res:
        p = res["paper"]
        curves = [{"label": "train", "x": [s["train_unfairness"] for s in p["sweep"]],
                   "y": [1 - s["train_error"] for s in p["sweep"]]},
                  {"label": "test", "x": [s["test_unfairness"] for s in p["sweep"]],
                   "y": [1 - s["test_error"] for s in p["sweep"]]}]
        base = {"x": m["logistic"]["rich"]["weighted_gap"], "y": m["logistic"]["overall"]["accuracy"],
                "label": "baseline (test)"}
        fig = plots.tradeoff(curves, base, "Paper setting: linear-threshold subgroups, 18 protected attributes",
                             xlabel="γ-unfairness (FPR, linear-threshold auditor)")
        fig.savefig(fig_dir / "communities_paper_tradeoff.png", dpi=150)
        fx, bg = p["bug"]["fixed"], p["bug"]["with_bug"]
        fig = plots.convergence({f"correct cost update (train error {100 * fx['train_error']:.1f}%)": fx["history"],
                                 f"GerryFair indexing issue (train error {100 * bg['train_error']:.1f}%)": bg["history"]},
                                0.005, "Fictitious play on Communities & Crime (γ = 0.005)")
        fig.savefig(fig_dir / "communities_convergence.png", dpi=150)
    plots.plt.close("all")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["communities", "adult", "diabetes"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--gbm-iters", type=int, default=25)
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from results/*.json")
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    for name in args.datasets:
        if args.figures_only:
            make_figures(json.loads((ROOT / "results" / f"{name}.json").read_text()), ROOT / "figures")
            continue
        res = run(data.load(name), iters=args.iters, gbm_iters=args.gbm_iters)
        (ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1))
        make_figures(res, ROOT / "figures")
        print(f"[{name}] wrote results/{name}.json and figures/{name}_*.png")


if __name__ == "__main__":
    main()
