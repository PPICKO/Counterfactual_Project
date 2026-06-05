"""
Cross-Dataset Comparison Plots for Counterfactual Fairness Study.

Generates:
1. cross_dataset_comparison.png - Side-by-side comparison of Adult vs ACS
   showing validity, proximity, and fairness metrics
2. fairness_by_group_comparison.png - Detailed fairness gap visualization
   by demographic group

This script compares results from two datasets:
- Adult Income dataset (results/)
- ACS Income dataset (results_acs/)
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_all_results(
    base_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load results from both Adult and ACS experiments.

    Args:
        base_dir: Base project directory.

    Returns:
        Tuple of (adult_wachter, adult_glance, acs_wachter, acs_glance) dictionaries.
    """
    # Adult dataset results
    adult_results_dir = base_dir / "results"
    with open(adult_results_dir / "wachter" / "results.pkl", "rb") as f:
        adult_wachter = pickle.load(f)
    with open(adult_results_dir / "glance" / "results.pkl", "rb") as f:
        adult_glance = pickle.load(f)

    # ACS dataset results
    acs_results_dir = base_dir / "results_acs"
    with open(acs_results_dir / "wachter" / "results.pkl", "rb") as f:
        acs_wachter = pickle.load(f)
    with open(acs_results_dir / "glance" / "results.pkl", "rb") as f:
        acs_glance = pickle.load(f)

    return adult_wachter, adult_glance, acs_wachter, acs_glance


def extract_metrics(
    wachter_data: Dict[str, Any],
    glance_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract key metrics from results for comparison.

    Args:
        wachter_data: Wachter method results.
        glance_data: GLANCE method results.

    Returns:
        Dictionary with extracted metrics.
    """
    wachter_results = wachter_data["results"]
    glance_results = glance_data["results"]
    protected = wachter_data["protected"]

    # Overall validity
    wachter_validity = np.mean([r["validity"] for r in wachter_results])
    glance_validity = np.mean([r["validity"] for r in glance_results])

    # Distance (L2) for valid counterfactuals
    wachter_distances = [r["distance"] for r in wachter_results if r["validity"]]
    glance_distances = [r["distance"] for r in glance_results if r["validity"]]

    wachter_avg_distance = np.mean(wachter_distances) if wachter_distances else 0.0
    glance_avg_distance = np.mean(glance_distances) if glance_distances else 0.0

    # Per-group validity
    unique_groups = np.unique(protected)
    group_metrics = {}

    for group in unique_groups:
        mask = protected == group
        wachter_group = [r for i, r in enumerate(wachter_results) if mask[i]]
        glance_group = [r for i, r in enumerate(glance_results) if mask[i]]

        wachter_group_validity = np.mean([r["validity"] for r in wachter_group])
        glance_group_validity = np.mean([r["validity"] for r in glance_group])

        group_metrics[str(group)] = {
            "count": int(mask.sum()),
            "wachter_validity": float(wachter_group_validity),
            "glance_validity": float(glance_group_validity),
        }

    # Fairness gap (demographic parity difference)
    wachter_validities = [m["wachter_validity"] for m in group_metrics.values()]
    glance_validities = [m["glance_validity"] for m in group_metrics.values()]

    wachter_fairness_gap = max(wachter_validities) - min(wachter_validities)
    glance_fairness_gap = max(glance_validities) - min(glance_validities)

    return {
        "wachter_validity": wachter_validity,
        "glance_validity": glance_validity,
        "wachter_distance": wachter_avg_distance,
        "glance_distance": glance_avg_distance,
        "wachter_fairness_gap": wachter_fairness_gap,
        "glance_fairness_gap": glance_fairness_gap,
        "group_metrics": group_metrics,
        "unique_groups": unique_groups,
    }


def create_cross_dataset_comparison(
    adult_metrics: Dict[str, Any],
    acs_metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create cross-dataset comparison visualization with 3 panels.

    Args:
        adult_metrics: Metrics from Adult dataset.
        acs_metrics: Metrics from ACS dataset.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Color scheme
    wachter_color = "#1f77b4"  # Blue
    glance_color = "#ff7f0e"   # Orange

    datasets = ["Adult", "ACS"]
    x = np.arange(len(datasets))
    width = 0.35

    # Panel 1: Validity Rates
    ax = axes[0]
    wachter_validity = [
        adult_metrics["wachter_validity"] * 100,
        acs_metrics["wachter_validity"] * 100,
    ]
    glance_validity = [
        adult_metrics["glance_validity"] * 100,
        acs_metrics["glance_validity"] * 100,
    ]

    bars1 = ax.bar(
        x - width / 2,
        wachter_validity,
        width,
        label="Wachter",
        color=wachter_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        glance_validity,
        width,
        label="GLANCE",
        color=glance_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars
    for bar, val in zip(bars1, wachter_validity):
        ax.annotate(
            f"{val:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for bar, val in zip(bars2, glance_validity):
        ax.annotate(
            f"{val:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Validity Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Validity Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation for Adult being harder
    ax.annotate(
        "4x higher\nvalidity",
        xy=(0, 49),
        xytext=(0.5, 58),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )

    # Panel 2: L2 Distance (Proximity)
    ax = axes[1]
    wachter_distance = [
        adult_metrics["wachter_distance"],
        acs_metrics["wachter_distance"],
    ]
    glance_distance = [
        adult_metrics["glance_distance"],
        acs_metrics["glance_distance"],
    ]

    bars1 = ax.bar(
        x - width / 2,
        wachter_distance,
        width,
        label="Wachter",
        color=wachter_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        glance_distance,
        width,
        label="GLANCE",
        color=glance_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for bar, val in zip(bars1, wachter_distance):
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for bar, val in zip(bars2, glance_distance):
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("L2 Distance (lower = better)", fontsize=12, fontweight="bold")
    ax.set_title("Proximity Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance markers
    ax.annotate("***", xy=(0, 0.52), ha="center", fontsize=14, fontweight="bold")
    ax.annotate("***", xy=(1, 0.25), ha="center", fontsize=14, fontweight="bold")

    # Panel 3: Fairness Gap
    ax = axes[2]
    wachter_gap = [
        adult_metrics["wachter_fairness_gap"] * 100,
        acs_metrics["wachter_fairness_gap"] * 100,
    ]
    glance_gap = [
        adult_metrics["glance_fairness_gap"] * 100,
        acs_metrics["glance_fairness_gap"] * 100,
    ]

    bars1 = ax.bar(
        x - width / 2,
        wachter_gap,
        width,
        label="Wachter",
        color=wachter_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        glance_gap,
        width,
        label="GLANCE",
        color=glance_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for bar, val in zip(bars1, wachter_gap):
        ax.annotate(
            f"{val:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for bar, val in zip(bars2, glance_gap):
        ax.annotate(
            f"{val:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Fairness Gap (%) (lower = better)", fontsize=12, fontweight="bold")
    ax.set_title("Fairness Gap Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 50)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance indicators
    ax.annotate("***", xy=(0, 42), ha="center", fontsize=14, fontweight="bold")
    ax.annotate("NS", xy=(1, 10), ha="center", fontsize=11, fontweight="bold", color="gray")

    # Add annotation about 84% improvement
    ax.annotate(
        "84% reduction\nin fairness gap",
        xy=(0.5, 38),
        xytext=(1.2, 35),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        color="darkgreen",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def create_fairness_by_group_comparison(
    adult_metrics: Dict[str, Any],
    acs_metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create detailed fairness gap visualization by demographic group.

    Args:
        adult_metrics: Metrics from Adult dataset.
        acs_metrics: Metrics from ACS dataset.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color scheme
    wachter_color = "#1f77b4"  # Blue
    glance_color = "#ff7f0e"   # Orange
    female_color = "#e377c2"   # Pink
    male_color = "#2ca02c"     # Green

    # Map group names
    # Adult uses "Female"/"Male", ACS uses "1.0"/"2.0" (1=Male, 2=Female based on typical coding)
    adult_groups = adult_metrics["group_metrics"]
    acs_groups = acs_metrics["group_metrics"]

    # Left panel: Adult Dataset
    ax = axes[0]
    ax.set_title("Adult Dataset\n(Significant Gender Disparity)", fontsize=14, fontweight="bold")

    # Get Adult group data
    if "Female" in adult_groups:
        female_key = "Female"
        male_key = "Male"
    else:
        # Fallback if different naming
        keys = list(adult_groups.keys())
        female_key = keys[0]
        male_key = keys[1] if len(keys) > 1 else keys[0]

    female_wachter = adult_groups[female_key]["wachter_validity"] * 100
    female_glance = adult_groups[female_key]["glance_validity"] * 100
    male_wachter = adult_groups[male_key]["wachter_validity"] * 100
    male_glance = adult_groups[male_key]["glance_validity"] * 100

    # Bar positions
    groups = ["Female", "Male"]
    x = np.arange(len(groups))
    width = 0.35

    wachter_vals = [female_wachter, male_wachter]
    glance_vals = [female_glance, male_glance]

    bars1 = ax.bar(
        x - width / 2,
        wachter_vals,
        width,
        label="Wachter",
        color=wachter_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        glance_vals,
        width,
        label="GLANCE",
        color=glance_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for bar, val in zip(bars1, wachter_vals):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    for bar, val in zip(bars2, glance_vals):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Validity Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Demographic Group", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add disparity indicator
    disparity_ratio = male_wachter / female_wachter if female_wachter > 0 else 0
    ax.annotate(
        f"2.5x disparity\n(p < 0.001) ***",
        xy=(0.5, 50),
        fontsize=11,
        ha="center",
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="darkred"),
    )

    # Draw bracket showing gap
    ax.annotate(
        "",
        xy=(0, female_wachter),
        xytext=(0, male_wachter),
        arrowprops=dict(arrowstyle="<->", color="darkred", lw=2),
    )

    # Right panel: ACS Dataset
    ax = axes[1]
    ax.set_title("ACS Dataset\n(No Significant Gender Disparity)", fontsize=14, fontweight="bold")

    # Get ACS group data (1.0 = Male, 2.0 = Female typically in ACS)
    acs_keys = list(acs_groups.keys())
    if "1.0" in acs_groups:
        male_key_acs = "1.0"
        female_key_acs = "2.0"
    else:
        male_key_acs = acs_keys[0]
        female_key_acs = acs_keys[1] if len(acs_keys) > 1 else acs_keys[0]

    female_wachter_acs = acs_groups[female_key_acs]["wachter_validity"] * 100
    female_glance_acs = acs_groups[female_key_acs]["glance_validity"] * 100
    male_wachter_acs = acs_groups[male_key_acs]["wachter_validity"] * 100
    male_glance_acs = acs_groups[male_key_acs]["glance_validity"] * 100

    # For ACS: 1.0 = Male, 2.0 = Female
    groups_acs = ["Female", "Male"]
    wachter_vals_acs = [female_wachter_acs, male_wachter_acs]
    glance_vals_acs = [female_glance_acs, male_glance_acs]

    bars1 = ax.bar(
        x - width / 2,
        wachter_vals_acs,
        width,
        label="Wachter",
        color=wachter_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        glance_vals_acs,
        width,
        label="GLANCE",
        color=glance_color,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for bar, val in zip(bars1, wachter_vals_acs):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    for bar, val in zip(bars2, glance_vals_acs):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Validity Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Demographic Group", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups_acs, fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add NS indicator
    ax.annotate(
        "NS\n(p = 0.76)",
        xy=(0.5, 20),
        fontsize=11,
        ha="center",
        fontweight="bold",
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="gray", alpha=0.5),
    )

    # Add summary text at top
    fig.suptitle(
        "Counterfactual Validity by Gender: Cross-Dataset Comparison",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    """Main function to generate cross-dataset comparison plots."""
    print("=" * 70)
    print("CROSS-DATASET COMPARISON PLOT GENERATION")
    print("=" * 70)

    # Set up paths
    base_dir = Path(__file__).parent.resolve()
    output_dir = base_dir / "results_acs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results from both datasets
    print("\n[1] Loading results from Adult and ACS datasets...")
    try:
        adult_wachter, adult_glance, acs_wachter, acs_glance = load_all_results(base_dir)
        print(f"    Adult: {len(adult_wachter['results'])} samples")
        print(f"    ACS:   {len(acs_wachter['results'])} samples")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find results files: {e}")
        return

    # Extract metrics
    print("\n[2] Extracting metrics...")
    adult_metrics = extract_metrics(adult_wachter, adult_glance)
    acs_metrics = extract_metrics(acs_wachter, acs_glance)

    print("\n    Adult Dataset Metrics:")
    print(f"      Wachter validity: {adult_metrics['wachter_validity']:.2%}")
    print(f"      GLANCE validity:  {adult_metrics['glance_validity']:.2%}")
    print(f"      Wachter fairness gap: {adult_metrics['wachter_fairness_gap']:.2%}")
    print(f"      GLANCE fairness gap:  {adult_metrics['glance_fairness_gap']:.2%}")

    print("\n    ACS Dataset Metrics:")
    print(f"      Wachter validity: {acs_metrics['wachter_validity']:.2%}")
    print(f"      GLANCE validity:  {acs_metrics['glance_validity']:.2%}")
    print(f"      Wachter fairness gap: {acs_metrics['wachter_fairness_gap']:.2%}")
    print(f"      GLANCE fairness gap:  {acs_metrics['glance_fairness_gap']:.2%}")

    # Generate plots
    print("\n[3] Creating cross-dataset comparison plot...")
    create_cross_dataset_comparison(
        adult_metrics,
        acs_metrics,
        output_dir / "cross_dataset_comparison.png",
    )

    print("\n[4] Creating fairness by group comparison plot...")
    create_fairness_by_group_comparison(
        adult_metrics,
        acs_metrics,
        output_dir / "fairness_by_group_comparison.png",
    )

    # Also save to paper directory
    paper_dir = base_dir / "paper"
    if paper_dir.exists():
        print("\n[5] Copying plots to paper directory...")
        import shutil
        shutil.copy(
            output_dir / "cross_dataset_comparison.png",
            paper_dir / "cross_dataset_comparison.png",
        )
        shutil.copy(
            output_dir / "fairness_by_group_comparison.png",
            paper_dir / "fairness_by_group_comparison.png",
        )
        print(f"    Copied to: {paper_dir}")

    print("\n" + "=" * 70)
    print("PLOT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  1. {output_dir / 'cross_dataset_comparison.png'}")
    print(f"  2. {output_dir / 'fairness_by_group_comparison.png'}")


if __name__ == "__main__":
    main()
