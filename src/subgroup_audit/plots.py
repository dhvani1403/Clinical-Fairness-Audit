"""Matplotlib figures for the report and the audit tool (light surface, fixed palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order: blue, orange, aqua, violet.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})


def _pp(x):
    return 100 * x


def gap_panels(result: dict, title: str, top_k: int = 8):
    """Single-attribute gaps (left) next to the worst intersectional gaps (right).

    Bars are gaps in percentage points vs. the overall rate; whiskers are 95% CIs.
    """
    single = [r for r in result["single_attribute"] if r["reliable"]]
    inter = [r for r in result["top_raw_gap"] if r["order"] >= 2][:top_k]
    lim = max([abs(r["gap"]) + 1.96 * r["std_err"] for r in single + inter] + [0.01])
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * max(len(single), len(inter)) + 1.8),
                             sharex=True, layout="constrained")
    for ax, rows, color, name in ((axes[0], single, BLUE, "Single attributes"),
                                  (axes[1], inter, ORANGE, "Worst intersectional subgroups")):
        rows = rows[::-1]
        ypos = range(len(rows))
        ax.barh(ypos, [_pp(r["gap"]) for r in rows], color=color, height=0.62,
                xerr=[_pp(1.96 * r["std_err"]) for r in rows],
                error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        ax.set_yticks(list(ypos), [r["subgroup"].replace(" & ", " ∧ ") for r in rows], fontsize=8.5)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_xlim(-_pp(lim) * 1.1, _pp(lim) * 1.1)
        ax.grid(axis="y", visible=False)
        ax.set_title(name, loc="left")
        ax.set_xlabel(f"gap in {result['metric_label']} vs. overall (percentage points)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return fig


def worst_gap_bars(rows: list[dict], title: str):
    """Grouped bars per model: worst single-attribute gap vs. worst intersectional gap.

    rows: [{"model": str, "single": float, "intersection": float}] (absolute gaps, 0-1).
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = range(len(rows))
    w = 0.36
    for off, key, color, label in ((-w / 2, "single", BLUE, "worst single attribute"),
                                   (w / 2, "intersection", ORANGE, "worst intersection")):
        vals = [_pp(r[key]) for r in rows]
        bars = ax.bar([i + off for i in x], vals, width=w - 0.03, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(list(x), [r["model"] for r in rows])
    ax.set_ylabel("largest |gap| (percentage points)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def tradeoff(curves: list[dict], baseline: dict, title: str, xlabel: str, ylabel: str = "test accuracy"):
    """Accuracy vs. unfairness across fairness strengths.

    curves: [{"label": str, "x": [...], "y": [...], "gammas": [...]}]; baseline: {"x", "y", "label"}.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for curve, color in zip(curves, (ORANGE, AQUA, VIOLET)):
        order = sorted(range(len(curve["x"])), key=lambda i: curve["x"][i])
        xs = [curve["x"][i] for i in order]
        ys = [curve["y"][i] for i in order]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, label=curve["label"],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot([baseline["x"]], [baseline["y"]], marker="D", ms=8, color=BLUE, ls="none",
            label=baseline["label"], markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def convergence(histories: dict[str, list[dict]], gamma: float, title: str):
    """Worst size-weighted disparity found by the auditor per iteration."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (label, hist), color in zip(histories.items(), (AQUA, ORANGE, VIOLET)):
        ax.plot([h["iteration"] for h in hist], [h["worst_weighted_gap"] for h in hist],
                color=color, lw=2, label=label)
    ax.axhline(gamma, color=MUTED, lw=1, ls="--")
    last = max(h["iteration"] for hist in histories.values() for h in hist)
    ax.annotate(f"γ = {gamma}", (last, gamma), xytext=(0, -12), ha="right",
                textcoords="offset points",
                color=INK2, fontsize=8.5)
    ax.set_xlabel("fictitious-play iteration")
    ax.set_ylabel("worst size-weighted disparity (train)")
    ax.legend(fontsize=9)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig
