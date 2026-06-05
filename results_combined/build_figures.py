"""Generate all PNG figures into results_combined/figures/."""
from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _data import (
    CELLS, CELL_LABELS, CLASSIFIER, CF,
    is_degenerate, sig_stars,
)

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 180

# Stable color scheme across all figures.
COLOR_WACHTER = "#3B7DD8"   # blue
COLOR_GLANCE  = "#E07A5F"   # warm orange
COLOR_DEGEN_EDGE = "#C0392B"  # red border for degenerate bars
HATCH = "///"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _bar(ax, x, height, color, degen=False, label=None):
    edgecolor = COLOR_DEGEN_EDGE if degen else "black"
    lw = 2.0 if degen else 0.5
    hatch = HATCH if degen else None
    return ax.bar(
        x, height,
        color=color if not degen else "white",
        edgecolor=edgecolor,
        linewidth=lw,
        hatch=hatch,
        label=label,
        width=0.38,
    )


def _label_bar(ax, bars, labels, fontsize=8, dy=0):
    for b, t in zip(bars, labels):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + dy,
            t,
            ha="center", va="bottom", fontsize=fontsize,
        )


def _annotate_degen(ax, x, y):
    ax.annotate(
        "DEGENERATE",
        xy=(x, y),
        xytext=(x, y + 0.04),
        ha="center", va="bottom",
        fontsize=7, color=COLOR_DEGEN_EDGE, fontweight="bold",
        rotation=90,
    )


def _grid_2x2(figsize=(11, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=False)
    return fig, axes


def _grid_cell(dataset_idx, classifier_idx, axes):
    return axes[dataset_idx][classifier_idx]


# ----------------------------------------------------------------------
# Fig 1: Classifier accuracy & AUC
# ----------------------------------------------------------------------

def fig1():
    fig, ax = plt.subplots(figsize=(9, 5))
    cells = CELLS
    x = np.arange(len(cells))
    w = 0.38
    acc = [CLASSIFIER[c]["accuracy"] for c in cells]
    auc = [CLASSIFIER[c]["auc"] for c in cells]
    b1 = ax.bar(x - w/2, acc, width=w, color="#4C72B0", label="Accuracy",
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, auc, width=w, color="#55A868", label="AUC",
                edgecolor="black", linewidth=0.5)
    for bars, vals in [(b1, acc), (b2, auc)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Classifier performance across the 4 cells (held-out test split)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_classifier_performance.png", dpi=DPI)
    plt.close(fig)
    print("wrote fig1_classifier_performance.png")


# ----------------------------------------------------------------------
# Helper: standard 2x2 method-bar plot
# ----------------------------------------------------------------------

def _method_bars_2x2(metric_key, ylabel, title, outname,
                     value_format=lambda v: f"{v:.3f}",
                     y_is_percent=False, ymax_pad=0.15):
    """Build a 2x2 grid (rows=datasets, cols=classifiers) of Wachter-vs-GLANCE
    bar charts for one metric."""
    fig, axes = _grid_2x2(figsize=(11, 8))
    layout = {
        # (row, col) -> cell
        (0, 0): "adult_lr", (0, 1): "adult_rf",
        (1, 0): "acs_lr",   (1, 1): "acs_rf",
    }

    # global ymax for consistent comparison
    all_vals = [CF[(c, m)][metric_key] for c in CELLS for m in ("wachter", "glance")]
    all_vals = [v for v in all_vals if v == v]  # drop NaN
    ymax = max(all_vals) * (1 + ymax_pad) if all_vals else 1.0

    for (r, c), cell in layout.items():
        ax = axes[r][c]
        w_val = CF[(cell, "wachter")][metric_key]
        g_val = CF[(cell, "glance")][metric_key]
        w_degen = is_degenerate(cell, "wachter")
        g_degen = is_degenerate(cell, "glance")

        bw = _bar(ax, 0, w_val, COLOR_WACHTER, degen=w_degen, label="Wachter")
        bg = _bar(ax, 1, g_val, COLOR_GLANCE,  degen=g_degen, label="GLANCE")

        _label_bar(ax, bw, [value_format(w_val)], dy=ymax * 0.01)
        _label_bar(ax, bg, [value_format(g_val)], dy=ymax * 0.01)

        if w_degen:
            _annotate_degen(ax, 0, max(w_val, ymax * 0.05))
        if g_degen:
            _annotate_degen(ax, 1, max(g_val, ymax * 0.05))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Wachter", "GLANCE"])
        ax.set_ylim(0, ymax)
        if y_is_percent:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{100*v:.0f}%"
            ))
        ax.set_ylabel(ylabel)
        ax.set_title(CELL_LABELS[cell])
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / outname, dpi=DPI)
    plt.close(fig)
    print(f"wrote {outname}")


# ----------------------------------------------------------------------
# Fig 2: Validity 2x2
# ----------------------------------------------------------------------

def fig2():
    _method_bars_2x2(
        "validity",
        ylabel="Validity rate",
        title="Validity: Wachter vs GLANCE (2x2 design)",
        outname="fig2_validity_4way.png",
        value_format=lambda v: f"{100*v:.1f}%",
        y_is_percent=True,
    )


# ----------------------------------------------------------------------
# Fig 3: L2 distance 2x2
# ----------------------------------------------------------------------

def fig3():
    _method_bars_2x2(
        "l2",
        ylabel="Mean L2 distance",
        title="L2 distance (proximity) across the 4 cells",
        outname="fig3_l2_distance_4way.png",
        value_format=lambda v: f"{v:.3f}",
        y_is_percent=False,
    )


# ----------------------------------------------------------------------
# Fig 4: DP gap with 95% CI error bars
# ----------------------------------------------------------------------

def fig4():
    fig, axes = _grid_2x2(figsize=(11, 8))
    layout = {
        (0, 0): "adult_lr", (0, 1): "adult_rf",
        (1, 0): "acs_lr",   (1, 1): "acs_rf",
    }
    # global ymax
    ymax = 0.50

    for (r, c), cell in layout.items():
        ax = axes[r][c]
        vals = []
        cis_low = []
        cis_high = []
        degens = []
        ps = []
        labels = ["Wachter", "GLANCE"]
        for m in ("wachter", "glance"):
            d = CF[(cell, m)]
            vals.append(d["dp_gap"])
            lo, hi = d["dp_gap_ci"]
            cis_low.append(d["dp_gap"] - lo)
            cis_high.append(hi - d["dp_gap"])
            degens.append(is_degenerate(cell, m))
            ps.append(d["dp_p"])

        x = np.arange(2)
        for i, (v, deg) in enumerate(zip(vals, degens)):
            color = COLOR_WACHTER if i == 0 else COLOR_GLANCE
            _bar(ax, i, v, color, degen=deg)
        ax.errorbar(
            x, vals,
            yerr=[cis_low, cis_high],
            fmt="none", ecolor="black", capsize=6, linewidth=1.2,
        )
        for i, (v, p, deg) in enumerate(zip(vals, ps, degens)):
            txt = f"{100*v:.1f}% {sig_stars(p)}"
            if deg:
                txt = "DEGEN"
            ax.text(i, v + 0.02, txt, ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{100*v:.0f}%"
        ))
        ax.set_ylabel("Sex DP gap (male - female)")
        ax.set_title(CELL_LABELS[cell])
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Demographic-parity gap on sex (95% CI, chi-square significance)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig4_dp_gap_4way.png", dpi=DPI)
    plt.close(fig)
    print("wrote fig4_dp_gap_4way.png")


# ----------------------------------------------------------------------
# Fig 5: FACTS Equal-Effectiveness gap
# ----------------------------------------------------------------------

def fig5():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cells = CELLS
    x = np.arange(len(cells))
    w = 0.38
    w_vals = [CF[(c, "wachter")]["facts_ee_gap"] for c in cells]
    g_vals = [CF[(c, "glance")]["facts_ee_gap"] for c in cells]
    w_ps   = [CF[(c, "wachter")]["facts_ee_p"]  for c in cells]
    g_ps   = [CF[(c, "glance")]["facts_ee_p"]  for c in cells]
    w_deg  = [is_degenerate(c, "wachter") for c in cells]
    g_deg  = [is_degenerate(c, "glance")  for c in cells]

    # bars one-by-one so we can mark degeneracy
    for i, (v, deg) in enumerate(zip(w_vals, w_deg)):
        _bar(ax, i - w/2, v, COLOR_WACHTER, degen=deg,
             label="Wachter" if i == 0 else None)
    for i, (v, deg) in enumerate(zip(g_vals, g_deg)):
        _bar(ax, i + w/2, v, COLOR_GLANCE, degen=deg,
             label="GLANCE" if i == 0 else None)

    # labels above
    for i, (v, p, deg) in enumerate(zip(w_vals, w_ps, w_deg)):
        txt = "DEGEN" if deg else f"{100*v:.1f}% {sig_stars(p)}"
        ax.text(i - w/2, v + 0.015, txt, ha="center", va="bottom", fontsize=8)
    for i, (v, p, deg) in enumerate(zip(g_vals, g_ps, g_deg)):
        txt = "DEGEN" if deg else f"{100*v:.1f}% {sig_stars(p)}"
        ax.text(i + w/2, v + 0.015, txt, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylim(0, 0.50)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{100*v:.0f}%"
    ))
    ax.set_ylabel("FACTS Equal-Effectiveness gap (sex)")
    ax.set_title("FACTS Equal-Effectiveness: success-rate parity gap across the 4 cells\n"
                 "(*** p<0.001, ** p<0.01, * p<0.05, ns not significant)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_facts_equal_effectiveness.png", dpi=DPI)
    plt.close(fig)
    print("wrote fig5_facts_equal_effectiveness.png")


# ----------------------------------------------------------------------
# Fig 6: Wachter collapse on tree ensembles (featured)
# ----------------------------------------------------------------------

def fig6():
    """Side-by-side bar chart showing the dramatic sparsity jump on RF
    cells, with validity overlay as a secondary panel."""
    fig, (ax_sp, ax_val) = plt.subplots(1, 2, figsize=(13, 5.5))

    cells = CELLS
    x = np.arange(len(cells))
    sp = [CF[(c, "wachter")]["sparsity"] for c in cells]
    val = [CF[(c, "wachter")]["validity"] for c in cells]
    degs = [is_degenerate(c, "wachter") for c in cells]

    # --- sparsity panel ---
    for i, (v, deg) in enumerate(zip(sp, degs)):
        _bar(ax_sp, i, v, COLOR_WACHTER, degen=deg)
        ax_sp.text(i, v + 0.02, f"{100*v:.2f}%", ha="center", va="bottom", fontsize=9)
        if deg:
            ax_sp.text(i, 0.55, "DEGENERATE", ha="center", va="center",
                       fontsize=10, color=COLOR_DEGEN_EDGE, fontweight="bold",
                       rotation=90)
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax_sp.set_ylim(0, 1.10)
    ax_sp.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{100*v:.0f}%"
    ))
    ax_sp.set_ylabel("Sparsity (fraction of features UNCHANGED)")
    ax_sp.set_title("Wachter sparsity collapses on tree ensembles")
    ax_sp.grid(axis="y", linestyle=":", alpha=0.5)
    ax_sp.axhline(0.95, color="grey", linestyle="--", linewidth=0.8)
    ax_sp.text(0.05, 0.965, "no-movement threshold (95%)", color="grey",
               fontsize=8, transform=ax_sp.get_yaxis_transform())

    # --- validity panel ---
    for i, (v, deg) in enumerate(zip(val, degs)):
        _bar(ax_val, i, v, COLOR_WACHTER, degen=deg)
        ax_val.text(i, v + 0.01, f"{100*v:.2f}%", ha="center", va="bottom", fontsize=9)
    ax_val.set_xticks(x)
    ax_val.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax_val.set_ylim(0, 0.65)
    ax_val.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{100*v:.0f}%"
    ))
    ax_val.set_ylabel("Wachter validity rate")
    ax_val.set_title("...and validity drops to ~0%")
    ax_val.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Finite-difference Wachter collapses on tree-ensemble decision surfaces\n"
        "(zero local gradient inside RF leaves => optimiser cannot move)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "fig6_wachter_collapse.png", dpi=DPI)
    plt.close(fig)
    print("wrote fig6_wachter_collapse.png")


# ----------------------------------------------------------------------
# Fig 7: FACTS Equal Cost of Effectiveness (4 cells x 2 methods)
# ----------------------------------------------------------------------

def fig7():
    """Mirror fig5's layout for the FACTS 4th dimension.

    At target_effectiveness=0.80, the target is unreachable in every cell
    (max observed group validity is 71.2%), so every per-group cost is NaN
    and every disparity is 0. The figure shows the empty bars explicitly
    and annotates each cell with 'TARGET UNREACHABLE' alongside the same
    'DEGEN' marker on Wachter-RF cells used in the other figures.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cells = CELLS
    x = np.arange(len(cells))
    w = 0.38
    w_vals = [CF[(c, "wachter")]["facts_coe_disparity"] for c in cells]
    g_vals = [CF[(c, "glance")]["facts_coe_disparity"] for c in cells]
    w_deg  = [is_degenerate(c, "wachter") for c in cells]
    g_deg  = [is_degenerate(c, "glance")  for c in cells]

    for i, (v, deg) in enumerate(zip(w_vals, w_deg)):
        _bar(ax, i - w/2, v, COLOR_WACHTER, degen=deg,
             label="Wachter" if i == 0 else None)
    for i, (v, deg) in enumerate(zip(g_vals, g_deg)):
        _bar(ax, i + w/2, v, COLOR_GLANCE, degen=deg,
             label="GLANCE" if i == 0 else None)

    # Labels: every cell is unreachable at 80% target; mark DEGEN on RF Wachter
    for i, (v, deg) in enumerate(zip(w_vals, w_deg)):
        txt = "DEGEN" if deg else "N/A"
        ax.text(i - w/2, max(v, 0.001) + 0.005, txt,
                ha="center", va="bottom", fontsize=8)
    for i, (v, deg) in enumerate(zip(g_vals, g_deg)):
        txt = "DEGEN" if deg else "N/A"
        ax.text(i + w/2, max(v, 0.001) + 0.005, txt,
                ha="center", va="bottom", fontsize=8)

    # Single explanatory band across all cells
    ax.text(
        0.5, 0.55,
        "TARGET UNREACHABLE in every cell\n"
        "(max observed group validity = 71.2% < 80% target;\n"
        "no cell can supply enough successful CFs for 80% of subjects)",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=11, color="#555555", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8DC",
                  edgecolor="#999999"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells])
    ax.set_ylim(0, 0.50)
    ax.set_ylabel("Cost disparity at 80% effectiveness target (L2 units)")
    ax.set_title("FACTS Equal Cost of Effectiveness (target=80%): cost disparity by sex\n"
                 "Disparity = 0 everywhere because the 80% target is not reachable")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_equal_cost_of_effectiveness.png", dpi=DPI)
    plt.close(fig)
    print("wrote fig7_equal_cost_of_effectiveness.png")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
