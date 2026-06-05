"""
Comprehensive comparison analysis between Wachter and GLANCE methods.

Compares:
1. Validity rates
2. Proximity (distance metrics)
3. Sparsity (feature changes)
4. Fairness metrics (demographic parity)
5. Cost disparity across protected groups
6. Computational efficiency
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from src.classifiers import BaselineClassifier
from src.data_loader import AdultDataLoader
from src.facts_fairness import FACTSEvaluator, group_results_by_attribute
from src.glance_method import GLANCECounterfactual, evaluate_glance
from src.wachter_method import WachterCounterfactual, evaluate_counterfactuals


def load_results(
    results_dir: Union[str, Path] = "../results",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load saved results from both methods.

    Args:
        results_dir: Directory containing results.

    Returns:
        Tuple of (wachter_data, glance_data) dictionaries.
    """
    results_dir = Path(results_dir)

    with open(results_dir / "wachter" / "results.pkl", "rb") as f:
        wachter_data = pickle.load(f)

    with open(results_dir / "glance" / "results.pkl", "rb") as f:
        glance_data = pickle.load(f)

    return wachter_data, glance_data


def statistical_comparison(
    wachter_results: List[Dict[str, Any]],
    glance_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Perform statistical tests comparing methods.

    Args:
        wachter_results: List of Wachter result dictionaries.
        glance_results: List of GLANCE result dictionaries.

    Returns:
        Dictionary with statistical test results for distance, sparsity, validity.
    """
    # Extract metrics
    wachter_distances = np.array([r["distance"] for r in wachter_results])
    glance_distances = np.array([r["distance"] for r in glance_results])

    wachter_sparsity = np.array([r["sparsity"] for r in wachter_results])
    glance_sparsity = np.array([r["sparsity"] for r in glance_results])

    # Convert boolean validity to integer array for proper handling
    wachter_validity = np.array([int(r["validity"]) for r in wachter_results])
    glance_validity = np.array([int(r["validity"]) for r in glance_results])

    # T-tests
    distance_ttest = stats.ttest_ind(wachter_distances, glance_distances)
    sparsity_ttest = stats.ttest_ind(wachter_sparsity, glance_sparsity)

    # Chi-square test for validity
    # Use integer sums for contingency table
    wachter_valid_count = int(wachter_validity.sum())
    wachter_invalid_count = len(wachter_validity) - wachter_valid_count
    glance_valid_count = int(glance_validity.sum())
    glance_invalid_count = len(glance_validity) - glance_valid_count

    contingency = np.array([
        [wachter_valid_count, wachter_invalid_count],
        [glance_valid_count, glance_invalid_count],
    ])
    validity_chi2 = stats.chi2_contingency(contingency)

    results: Dict[str, Any] = {
        "distance": {
            "wachter_mean": float(wachter_distances.mean()),
            "glance_mean": float(glance_distances.mean()),
            "t_statistic": float(distance_ttest.statistic),
            "p_value": float(distance_ttest.pvalue),
            "significant": distance_ttest.pvalue < 0.05,
        },
        "sparsity": {
            "wachter_mean": float(wachter_sparsity.mean()),
            "glance_mean": float(glance_sparsity.mean()),
            "t_statistic": float(sparsity_ttest.statistic),
            "p_value": float(sparsity_ttest.pvalue),
            "significant": sparsity_ttest.pvalue < 0.05,
        },
        "validity": {
            "wachter_rate": float(wachter_validity.mean()),
            "glance_rate": float(glance_validity.mean()),
            "chi2_statistic": float(validity_chi2[0]),
            "p_value": float(validity_chi2[1]),
            "significant": validity_chi2[1] < 0.05,
        },
    }

    return results


def fairness_comparison(
    wachter_data: Dict[str, Any],
    glance_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare fairness metrics between methods.

    Args:
        wachter_data: Wachter results with protected attributes.
        glance_data: GLANCE results with protected attributes.

    Returns:
        Dictionary with fairness comparison including per-group metrics and summary.
    """
    wachter_results = wachter_data["results"]
    glance_results = glance_data["results"]

    protected = wachter_data["protected"]

    # Group by protected attribute
    unique_groups = np.unique(protected)

    fairness_metrics: Dict[str, Any] = {}

    for group in unique_groups:
        mask = protected == group

        # Wachter metrics for this group
        wachter_group = [r for i, r in enumerate(wachter_results) if mask[i]]
        wachter_validity = (
            float(np.mean([r["validity"] for r in wachter_group]))
            if wachter_group
            else 0.0
        )

        # Handle empty valid results for distance calculation
        wachter_valid_group = [r for r in wachter_group if r["validity"]]
        wachter_distance = (
            float(np.mean([r["distance"] for r in wachter_valid_group]))
            if wachter_valid_group
            else np.nan
        )

        # GLANCE metrics for this group
        glance_group = [r for i, r in enumerate(glance_results) if mask[i]]
        glance_validity = (
            float(np.mean([r["validity"] for r in glance_group]))
            if glance_group
            else 0.0
        )

        # Handle empty valid results for distance calculation
        glance_valid_group = [r for r in glance_group if r["validity"]]
        glance_distance = (
            float(np.mean([r["distance"] for r in glance_valid_group]))
            if glance_valid_group
            else np.nan
        )

        fairness_metrics[str(group)] = {
            "count": int(mask.sum()),
            "wachter": {
                "validity": wachter_validity,
                "distance": wachter_distance,
            },
            "glance": {
                "validity": glance_validity,
                "distance": glance_distance,
            },
        }

    # Compute demographic parity difference
    wachter_validities = [
        m["wachter"]["validity"] for m in fairness_metrics.values()
        if isinstance(m, dict) and "wachter" in m
    ]
    glance_validities = [
        m["glance"]["validity"] for m in fairness_metrics.values()
        if isinstance(m, dict) and "glance" in m
    ]

    wachter_dp_diff = (
        max(wachter_validities) - min(wachter_validities)
        if wachter_validities
        else 0.0
    )
    glance_dp_diff = (
        max(glance_validities) - min(glance_validities)
        if glance_validities
        else 0.0
    )

    # Compute cost disparity (distance disparity)
    wachter_distances = [
        m["wachter"]["distance"] for m in fairness_metrics.values()
        if isinstance(m, dict) and "wachter" in m and not np.isnan(m["wachter"]["distance"])
    ]
    glance_distances = [
        m["glance"]["distance"] for m in fairness_metrics.values()
        if isinstance(m, dict) and "glance" in m and not np.isnan(m["glance"]["distance"])
    ]

    wachter_cost_disparity = (
        max(wachter_distances) - min(wachter_distances) if wachter_distances else 0
    )
    glance_cost_disparity = (
        max(glance_distances) - min(glance_distances) if glance_distances else 0
    )

    # Statistical tests for validity differences between groups
    wachter_statistical_tests: Dict[str, Any] = {}
    glance_statistical_tests: Dict[str, Any] = {}

    if len(unique_groups) == 2:
        # For binary protected attribute (e.g., male/female), do pairwise test
        groups_list = list(unique_groups)
        group1, group2 = groups_list[0], groups_list[1]

        # Wachter contingency table
        mask1 = protected == group1
        mask2 = protected == group2

        wachter_group1_results = [r for i, r in enumerate(wachter_results) if mask1[i]]
        wachter_group2_results = [r for i, r in enumerate(wachter_results) if mask2[i]]

        wachter_g1_valid = sum([r["validity"] for r in wachter_group1_results])
        wachter_g1_invalid = len(wachter_group1_results) - wachter_g1_valid
        wachter_g2_valid = sum([r["validity"] for r in wachter_group2_results])
        wachter_g2_invalid = len(wachter_group2_results) - wachter_g2_valid

        wachter_contingency = np.array([
            [wachter_g1_valid, wachter_g1_invalid],
            [wachter_g2_valid, wachter_g2_invalid],
        ])

        # Chi-square or Fisher's exact (use Fisher's if any cell < 5)
        if np.min(wachter_contingency) < 5:
            wachter_test = stats.fisher_exact(wachter_contingency)
            wachter_statistical_tests = {
                "test_type": "fisher_exact",
                "statistic": float(wachter_test[0]),
                "p_value": float(wachter_test[1]),
                "significant": wachter_test[1] < 0.05,
                "group1": str(group1),
                "group2": str(group2),
            }
        else:
            wachter_test = stats.chi2_contingency(wachter_contingency)
            wachter_statistical_tests = {
                "test_type": "chi_square",
                "statistic": float(wachter_test[0]),
                "p_value": float(wachter_test[1]),
                "significant": wachter_test[1] < 0.05,
                "group1": str(group1),
                "group2": str(group2),
            }

        # Confidence interval for demographic parity gap (Wilson score interval)
        n1 = len(wachter_group1_results)
        n2 = len(wachter_group2_results)

        if n1 > 0 and n2 > 0:
            p1 = wachter_g1_valid / n1
            p2 = wachter_g2_valid / n2

            # Standard error for difference in proportions
            var1 = p1 * (1 - p1) / n1 if n1 > 0 else 0.0
            var2 = p2 * (1 - p2) / n2 if n2 > 0 else 0.0
            se = np.sqrt(var1 + var2)
            gap = abs(p1 - p2)
            ci_lower = gap - 1.96 * se
            ci_upper = gap + 1.96 * se

            wachter_statistical_tests["gap_estimate"] = gap
            wachter_statistical_tests["gap_ci_95"] = (
                max(0, ci_lower),
                min(1, ci_upper),
            )

        # Same for GLANCE
        glance_group1_results = [r for i, r in enumerate(glance_results) if mask1[i]]
        glance_group2_results = [r for i, r in enumerate(glance_results) if mask2[i]]

        glance_g1_valid = sum([r["validity"] for r in glance_group1_results])
        glance_g1_invalid = len(glance_group1_results) - glance_g1_valid
        glance_g2_valid = sum([r["validity"] for r in glance_group2_results])
        glance_g2_invalid = len(glance_group2_results) - glance_g2_valid

        glance_contingency = np.array([
            [glance_g1_valid, glance_g1_invalid],
            [glance_g2_valid, glance_g2_invalid],
        ])

        if np.min(glance_contingency) < 5:
            glance_test = stats.fisher_exact(glance_contingency)
            glance_statistical_tests = {
                "test_type": "fisher_exact",
                "statistic": float(glance_test[0]),
                "p_value": float(glance_test[1]),
                "significant": glance_test[1] < 0.05,
                "group1": str(group1),
                "group2": str(group2),
            }
        else:
            glance_test = stats.chi2_contingency(glance_contingency)
            glance_statistical_tests = {
                "test_type": "chi_square",
                "statistic": float(glance_test[0]),
                "p_value": float(glance_test[1]),
                "significant": glance_test[1] < 0.05,
                "group1": str(group1),
                "group2": str(group2),
            }

        n1 = len(glance_group1_results)
        n2 = len(glance_group2_results)

        if n1 > 0 and n2 > 0:
            p1 = glance_g1_valid / n1
            p2 = glance_g2_valid / n2

            var1 = p1 * (1 - p1) / n1 if n1 > 0 else 0.0
            var2 = p2 * (1 - p2) / n2 if n2 > 0 else 0.0
            se = np.sqrt(var1 + var2)
            gap = abs(p1 - p2)
            ci_lower = gap - 1.96 * se
            ci_upper = gap + 1.96 * se

            glance_statistical_tests["gap_estimate"] = gap
            glance_statistical_tests["gap_ci_95"] = (
                max(0, ci_lower),
                min(1, ci_upper),
            )

    fairness_metrics["summary"] = {
        "wachter_dp_diff": wachter_dp_diff,
        "glance_dp_diff": glance_dp_diff,
        "wachter_cost_disparity": wachter_cost_disparity,
        "glance_cost_disparity": glance_cost_disparity,
        "wachter_statistical_tests": wachter_statistical_tests,
        "glance_statistical_tests": glance_statistical_tests,
    }

    return fairness_metrics


def facts_evaluation(
    wachter_data: Dict[str, Any],
    glance_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate both methods using the FACTS fairness framework.

    Args:
        wachter_data: Wachter results with protected attributes.
        glance_data: GLANCE results with protected attributes.

    Returns:
        Dictionary with FACTS evaluation for both methods.
    """
    evaluator = FACTSEvaluator(alpha=0.05)

    # Group results by protected attribute (sex)
    wachter_grouped = group_results_by_attribute(
        wachter_data["results"],
        wachter_data["protected"],
    )

    glance_grouped = group_results_by_attribute(
        glance_data["results"],
        glance_data["protected"],
    )

    # Evaluate both methods
    wachter_facts = evaluator.evaluate_all(wachter_grouped)
    glance_facts = evaluator.evaluate_all(glance_grouped)

    return {
        "wachter": wachter_facts,
        "glance": glance_facts,
        "evaluator": evaluator,
    }


def create_visualizations(
    wachter_data: Dict[str, Any],
    glance_data: Dict[str, Any],
    output_dir: Union[str, Path] = "../results/comparison",
) -> None:
    """
    Create comparison visualizations.

    Args:
        wachter_data: Wachter results dictionary.
        glance_data: GLANCE results dictionary.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wachter_results = wachter_data["results"]
    glance_results = glance_data["results"]
    protected = wachter_data["protected"]

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # 1. Distance distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distance histogram
    ax = axes[0, 0]
    wachter_distances = [r["distance"] for r in wachter_results]
    glance_distances = [r["distance"] for r in glance_results]

    ax.hist(wachter_distances, bins=30, alpha=0.6, label="Wachter", color="blue")
    ax.hist(glance_distances, bins=30, alpha=0.6, label="GLANCE", color="orange")
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Frequency")
    ax.set_title("Proximity Comparison")
    ax.legend()

    # Sparsity comparison
    ax = axes[0, 1]
    wachter_sparsity = [r["sparsity"] * 100 for r in wachter_results]
    glance_sparsity = [r["sparsity"] * 100 for r in glance_results]

    ax.hist(wachter_sparsity, bins=20, alpha=0.6, label="Wachter", color="blue")
    ax.hist(glance_sparsity, bins=20, alpha=0.6, label="GLANCE", color="orange")
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Sparsity Comparison")
    ax.legend()

    # Validity by group
    ax = axes[1, 0]
    groups = np.unique(protected)

    wachter_group_validity: List[float] = []
    glance_group_validity: List[float] = []

    for group in groups:
        mask = protected == group
        wachter_group = [r for i, r in enumerate(wachter_results) if mask[i]]
        glance_group = [r for i, r in enumerate(glance_results) if mask[i]]

        wachter_group_validity.append(
            float(np.mean([r["validity"] for r in wachter_group]))
            if wachter_group
            else 0.0
        )
        glance_group_validity.append(
            float(np.mean([r["validity"] for r in glance_group]))
            if glance_group
            else 0.0
        )

    x = np.arange(len(groups))
    width = 0.35

    ax.bar(x - width / 2, wachter_group_validity, width, label="Wachter", color="blue", alpha=0.8)
    ax.bar(x + width / 2, glance_group_validity, width, label="GLANCE", color="orange", alpha=0.8)
    ax.set_xlabel("Protected Group")
    ax.set_ylabel("Validity Rate")
    ax.set_title("Validity by Protected Attribute (Sex)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.legend()
    ax.set_ylim(0, 1)

    # Distance by group (for valid counterfactuals)
    ax = axes[1, 1]

    wachter_group_distance: List[float] = []
    glance_group_distance: List[float] = []

    for group in groups:
        mask = protected == group
        wachter_group = [r for i, r in enumerate(wachter_results) if mask[i] and r["validity"]]
        glance_group = [r for i, r in enumerate(glance_results) if mask[i] and r["validity"]]

        if wachter_group:
            wachter_group_distance.append(float(np.mean([r["distance"] for r in wachter_group])))
        else:
            wachter_group_distance.append(0)

        if glance_group:
            glance_group_distance.append(float(np.mean([r["distance"] for r in glance_group])))
        else:
            glance_group_distance.append(0)

    ax.bar(x - width / 2, wachter_group_distance, width, label="Wachter", color="blue", alpha=0.8)
    ax.bar(x + width / 2, glance_group_distance, width, label="GLANCE", color="orange", alpha=0.8)
    ax.set_xlabel("Protected Group")
    ax.set_ylabel("Avg Distance (L2)")
    ax.set_title("Cost Disparity by Protected Attribute (Sex)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_overview.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved comparison_overview.png")

    # 2. Box plots for detailed distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distance box plot
    ax = axes[0]
    data_to_plot = [wachter_distances, glance_distances]
    bp = ax.boxplot(data_to_plot, labels=["Wachter", "GLANCE"], patch_artist=True)

    for patch, color in zip(bp["boxes"], ["lightblue", "orange"]):
        patch.set_facecolor(color)

    ax.set_ylabel("L2 Distance")
    ax.set_title("Distance Distribution")
    ax.grid(axis="y", alpha=0.3)

    # Sparsity box plot
    ax = axes[1]
    data_to_plot = [wachter_sparsity, glance_sparsity]
    bp = ax.boxplot(data_to_plot, labels=["Wachter", "GLANCE"], patch_artist=True)

    for patch, color in zip(bp["boxes"], ["lightblue", "orange"]):
        patch.set_facecolor(color)

    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Sparsity Distribution")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved distribution_boxplots.png")


def generate_comparison_report(
    wachter_data: Dict[str, Any],
    glance_data: Dict[str, Any],
    stats_results: Dict[str, Any],
    fairness_results: Dict[str, Any],
    facts_results: Optional[Dict[str, Any]] = None,
    output_file: Union[str, Path] = "../results/comparison/comparison_report.txt",
) -> None:
    """
    Generate comprehensive text report of comparison.

    Args:
        wachter_data: Wachter results dictionary.
        glance_data: GLANCE results dictionary.
        stats_results: Statistical test results.
        fairness_results: Fairness comparison results.
        facts_results: Optional FACTS evaluation results.
        output_file: Path to output report.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE COMPARISON: WACHTER VS GLANCE\n")
        f.write("=" * 80 + "\n\n")

        # 1. Overall metrics
        f.write("1. OVERALL METRICS\n")
        f.write("-" * 80 + "\n")

        wachter_metrics = wachter_data["metrics"]
        glance_metrics = glance_data["metrics"]

        f.write(f"{'Metric':<30} {'Wachter':<20} {'GLANCE':<20}\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Validity Rate':<30} {wachter_metrics['validity_rate']:<20.2%} "
            f"{glance_metrics['validity_rate']:<20.2%}\n"
        )
        f.write(
            f"{'Avg Distance (L2)':<30} {wachter_metrics['avg_distance']:<20.4f} "
            f"{glance_metrics['avg_distance']:<20.4f}\n"
        )
        f.write(
            f"{'Avg Sparsity':<30} {wachter_metrics['avg_sparsity']:<20.2%} "
            f"{glance_metrics['avg_sparsity']:<20.2%}\n"
        )
        f.write(
            f"{'Avg Features Changed':<30} {wachter_metrics['avg_num_changes']:<20.2f} "
            f"{glance_metrics['avg_num_changes']:<20.2f}\n"
        )
        f.write(
            f"{'Convergence Rate':<30} {wachter_metrics['convergence_rate']:<20.2%} "
            f"{glance_metrics.get('convergence_rate', 1.0):<20.2%}\n"
        )
        f.write("\n")

        # 2. Statistical significance
        f.write("2. STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("-" * 80 + "\n")

        f.write(f"\nDistance (L2):\n")
        f.write(f"  Wachter mean: {stats_results['distance']['wachter_mean']:.4f}\n")
        f.write(f"  GLANCE mean:  {stats_results['distance']['glance_mean']:.4f}\n")
        f.write(f"  t-statistic:  {stats_results['distance']['t_statistic']:.4f}\n")
        f.write(f"  p-value:      {stats_results['distance']['p_value']:.6f}\n")
        f.write(
            f"  Significant:  {'YES' if stats_results['distance']['significant'] else 'NO'}\n"
        )

        f.write(f"\nSparsity:\n")
        f.write(f"  Wachter mean: {stats_results['sparsity']['wachter_mean']:.4f}\n")
        f.write(f"  GLANCE mean:  {stats_results['sparsity']['glance_mean']:.4f}\n")
        f.write(f"  t-statistic:  {stats_results['sparsity']['t_statistic']:.4f}\n")
        f.write(f"  p-value:      {stats_results['sparsity']['p_value']:.6f}\n")
        f.write(
            f"  Significant:  {'YES' if stats_results['sparsity']['significant'] else 'NO'}\n"
        )

        f.write(f"\nValidity:\n")
        f.write(f"  Wachter rate: {stats_results['validity']['wachter_rate']:.2%}\n")
        f.write(f"  GLANCE rate:  {stats_results['validity']['glance_rate']:.2%}\n")
        f.write(f"  Chi-square:   {stats_results['validity']['chi2_statistic']:.4f}\n")
        f.write(f"  p-value:      {stats_results['validity']['p_value']:.6f}\n")
        f.write(
            f"  Significant:  {'YES' if stats_results['validity']['significant'] else 'NO'}\n"
        )
        f.write("\n")

        # 3. Fairness analysis
        f.write("3. FAIRNESS ANALYSIS (SEX)\n")
        f.write("-" * 80 + "\n")

        summary = fairness_results["summary"]

        f.write(f"\nDemographic Parity Difference:\n")
        f.write(f"  Wachter: {summary['wachter_dp_diff']:.2%}\n")
        f.write(f"  GLANCE:  {summary['glance_dp_diff']:.2%}\n")

        # Statistical significance tests
        if summary.get("wachter_statistical_tests"):
            wachter_tests = summary["wachter_statistical_tests"]
            f.write(
                f"\nWachter Statistical Test ({wachter_tests.get('group1', 'G1')} "
                f"vs {wachter_tests.get('group2', 'G2')}):\n"
            )
            f.write(f"  Test: {wachter_tests.get('test_type', 'N/A')}\n")
            f.write(f"  Statistic: {wachter_tests.get('statistic', 0):.4f}\n")
            f.write(f"  p-value: {wachter_tests.get('p_value', 1):.6f}\n")
            f.write(
                f"  Significant: {'YES' if wachter_tests.get('significant', False) else 'NO'} (a=0.05)\n"
            )
            if "gap_estimate" in wachter_tests:
                f.write(f"  Gap estimate: {wachter_tests['gap_estimate']:.2%}\n")
            if "gap_ci_95" in wachter_tests:
                ci = wachter_tests["gap_ci_95"]
                f.write(f"  95% CI: [{ci[0]:.2%}, {ci[1]:.2%}]\n")

        if summary.get("glance_statistical_tests"):
            glance_tests = summary["glance_statistical_tests"]
            f.write(
                f"\nGLANCE Statistical Test ({glance_tests.get('group1', 'G1')} "
                f"vs {glance_tests.get('group2', 'G2')}):\n"
            )
            f.write(f"  Test: {glance_tests.get('test_type', 'N/A')}\n")
            f.write(f"  Statistic: {glance_tests.get('statistic', 0):.4f}\n")
            f.write(f"  p-value: {glance_tests.get('p_value', 1):.6f}\n")
            f.write(
                f"  Significant: {'YES' if glance_tests.get('significant', False) else 'NO'} (a=0.05)\n"
            )
            if "gap_estimate" in glance_tests:
                f.write(f"  Gap estimate: {glance_tests['gap_estimate']:.2%}\n")
            if "gap_ci_95" in glance_tests:
                ci = glance_tests["gap_ci_95"]
                f.write(f"  95% CI: [{ci[0]:.2%}, {ci[1]:.2%}]\n")

        f.write(f"\nCost Disparity (Distance):\n")
        f.write(f"  Wachter: {summary['wachter_cost_disparity']:.4f}\n")
        f.write(f"  GLANCE:  {summary['glance_cost_disparity']:.4f}\n")

        f.write(f"\nPer-Group Analysis:\n")
        for group, metrics in fairness_results.items():
            if group == "summary":
                continue
            f.write(f"\n  {group}:\n")
            f.write(f"    Count: {metrics['count']}\n")
            f.write(f"    Wachter validity: {metrics['wachter']['validity']:.2%}\n")
            f.write(f"    GLANCE validity:  {metrics['glance']['validity']:.2%}\n")
            if not np.isnan(metrics["wachter"]["distance"]):
                f.write(f"    Wachter distance: {metrics['wachter']['distance']:.4f}\n")
            if not np.isnan(metrics["glance"]["distance"]):
                f.write(f"    GLANCE distance:  {metrics['glance']['distance']:.4f}\n")

        f.write("\n")

        # 4. Key findings
        f.write("4. KEY FINDINGS\n")
        f.write("-" * 80 + "\n\n")

        # Validity comparison
        if stats_results["validity"]["wachter_rate"] > stats_results["validity"]["glance_rate"]:
            validity_winner = "Wachter"
            validity_diff = (
                stats_results["validity"]["wachter_rate"]
                - stats_results["validity"]["glance_rate"]
            )
        else:
            validity_winner = "GLANCE"
            validity_diff = (
                stats_results["validity"]["glance_rate"]
                - stats_results["validity"]["wachter_rate"]
            )

        f.write(
            f"- Validity: {validity_winner} achieves {validity_diff:.1%} higher validity rate "
        )
        f.write(
            f"({'statistically significant' if stats_results['validity']['significant'] else 'not significant'})\n"
        )

        # Distance comparison
        if stats_results["distance"]["wachter_mean"] < stats_results["distance"]["glance_mean"]:
            distance_winner = "Wachter"
            distance_diff = (
                stats_results["distance"]["glance_mean"]
                - stats_results["distance"]["wachter_mean"]
            )
        else:
            distance_winner = "GLANCE"
            distance_diff = (
                stats_results["distance"]["wachter_mean"]
                - stats_results["distance"]["glance_mean"]
            )

        f.write(
            f"- Proximity: {distance_winner} generates counterfactuals {distance_diff:.4f} closer to originals "
        )
        f.write(
            f"({'statistically significant' if stats_results['distance']['significant'] else 'not significant'})\n"
        )

        # Fairness comparison
        if summary["wachter_dp_diff"] < summary["glance_dp_diff"]:
            fairness_winner = "Wachter"
            fairness_diff = summary["glance_dp_diff"] - summary["wachter_dp_diff"]
        else:
            fairness_winner = "GLANCE"
            fairness_diff = summary["wachter_dp_diff"] - summary["glance_dp_diff"]

        f.write(f"- Fairness: {fairness_winner} shows {fairness_diff:.1%} better demographic parity\n")

        # Cost disparity
        if summary["wachter_cost_disparity"] < summary["glance_cost_disparity"]:
            cost_winner = "Wachter"
        else:
            cost_winner = "GLANCE"

        f.write(
            f"- Cost Disparity: {cost_winner} has lower distance disparity across protected groups\n"
        )

        f.write("\n")

        # 5. GLANCE-specific insights
        if "rule_usage" in glance_metrics:
            f.write("5. GLANCE GLOBAL RULES\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Number of rules discovered: {len(glance_metrics['rule_usage'])}\n\n")

            f.write("Rule usage distribution:\n")
            for rule_id, count in sorted(glance_metrics["rule_usage"].items()):
                percentage = count / len(glance_data["results"]) * 100
                f.write(f"  Rule {rule_id}: {count} instances ({percentage:.1f}%)\n")

            f.write("\n")

        # 6. FACTS Fairness Evaluation
        if facts_results:
            f.write("6. FACTS FAIRNESS EVALUATION\n")
            f.write("-" * 80 + "\n\n")

            f.write(
                "FACTS (Fairness-Aware Counterfactuals for Subgroups) evaluates fairness\n"
            )
            f.write(
                "across 4 dimensions: Equal Burden, Equal Effectiveness, Equal Choice,\n"
            )
            f.write("and Equal Cost of Effectiveness.\n\n")

            # Wachter FACTS
            f.write("Wachter Method:\n")
            f.write("-" * 40 + "\n")

            wachter_facts = facts_results["wachter"]

            f.write(f"  Equal Burden (Cost Similarity):\n")
            for group, cost in wachter_facts["equal_burden"]["avg_cost_by_group"].items():
                if not np.isnan(cost):
                    f.write(f"    {group}: {cost:.4f}\n")
            f.write(
                f"    Disparity: {wachter_facts['equal_burden']['burden_disparity']:.4f}\n"
            )

            f.write(f"\n  Equal Effectiveness (Success Rate Parity):\n")
            for group, rate in wachter_facts["equal_effectiveness"][
                "success_rate_by_group"
            ].items():
                f.write(f"    {group}: {rate:.2%}\n")
            f.write(
                f"    Gap: {wachter_facts['equal_effectiveness']['effectiveness_gap']:.2%}\n"
            )
            if wachter_facts["equal_effectiveness"]["statistical_test"]:
                test = wachter_facts["equal_effectiveness"]["statistical_test"]
                f.write(
                    f"    p-value: {test['p_value']:.6f} "
                    f"({'significant' if test['significant'] else 'not significant'})\n"
                )

            f.write(f"\n  Equal Choice (Option Availability):\n")
            for group, rate in wachter_facts["equal_choice"][
                "choice_availability_by_group"
            ].items():
                f.write(f"    {group}: {rate:.2%}\n")
            f.write(
                f"    Disparity: {wachter_facts['equal_choice']['choice_disparity']:.2%}\n"
            )

            wachter_cost_eff = wachter_facts["equal_cost_of_effectiveness"]
            f.write(
                f"\n  Equal Cost of Effectiveness "
                f"(Cost at {wachter_cost_eff['target_effectiveness']:.0%} target):\n"
            )
            for group, cost in wachter_cost_eff[
                "cost_at_target_effectiveness"
            ].items():
                if not np.isnan(cost):
                    f.write(f"    {group}: {cost:.4f}\n")
                else:
                    f.write(f"    {group}: N/A (target not achievable)\n")
            f.write(
                f"    Disparity: {wachter_cost_eff['cost_disparity']:.4f}\n"
            )

            # GLANCE FACTS
            f.write("\n\nGLANCE Method:\n")
            f.write("-" * 40 + "\n")

            glance_facts = facts_results["glance"]

            f.write(f"  Equal Burden (Cost Similarity):\n")
            for group, cost in glance_facts["equal_burden"]["avg_cost_by_group"].items():
                if not np.isnan(cost):
                    f.write(f"    {group}: {cost:.4f}\n")
            f.write(
                f"    Disparity: {glance_facts['equal_burden']['burden_disparity']:.4f}\n"
            )

            f.write(f"\n  Equal Effectiveness (Success Rate Parity):\n")
            for group, rate in glance_facts["equal_effectiveness"][
                "success_rate_by_group"
            ].items():
                f.write(f"    {group}: {rate:.2%}\n")
            f.write(
                f"    Gap: {glance_facts['equal_effectiveness']['effectiveness_gap']:.2%}\n"
            )
            if glance_facts["equal_effectiveness"]["statistical_test"]:
                test = glance_facts["equal_effectiveness"]["statistical_test"]
                f.write(
                    f"    p-value: {test['p_value']:.6f} "
                    f"({'significant' if test['significant'] else 'not significant'})\n"
                )

            f.write(f"\n  Equal Choice (Option Availability):\n")
            for group, rate in glance_facts["equal_choice"][
                "choice_availability_by_group"
            ].items():
                f.write(f"    {group}: {rate:.2%}\n")
            f.write(
                f"    Disparity: {glance_facts['equal_choice']['choice_disparity']:.2%}\n"
            )

            if "rule_diversity_by_group" in glance_facts["equal_choice"]:
                f.write(f"    Rule Diversity:\n")
                for group, n_rules in glance_facts["equal_choice"][
                    "rule_diversity_by_group"
                ].items():
                    f.write(f"      {group}: {n_rules} unique rules\n")

            glance_cost_eff = glance_facts["equal_cost_of_effectiveness"]
            f.write(
                f"\n  Equal Cost of Effectiveness "
                f"(Cost at {glance_cost_eff['target_effectiveness']:.0%} target):\n"
            )
            for group, cost in glance_cost_eff[
                "cost_at_target_effectiveness"
            ].items():
                if not np.isnan(cost):
                    f.write(f"    {group}: {cost:.4f}\n")
                else:
                    f.write(f"    {group}: N/A (target not achievable)\n")
            f.write(
                f"    Disparity: {glance_cost_eff['cost_disparity']:.4f}\n"
            )

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Saved comparison report to {output_file}")


def main() -> None:
    """Run comprehensive comparison analysis."""
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON ANALYSIS: WACHTER VS GLANCE")
    print("=" * 80)

    # Load results
    print("\n1. Loading saved results...")
    wachter_data, glance_data = load_results()

    print(f"   Wachter: {len(wachter_data['results'])} samples")
    print(f"   GLANCE:  {len(glance_data['results'])} samples")

    # Statistical comparison
    print("\n2. Running statistical tests...")
    stats_results = statistical_comparison(wachter_data["results"], glance_data["results"])

    print(
        f"   Distance difference: "
        f"{abs(stats_results['distance']['wachter_mean'] - stats_results['distance']['glance_mean']):.4f}"
    )
    print(f"   Significant: {stats_results['distance']['significant']}")

    # Fairness comparison
    print("\n3. Analyzing fairness metrics...")
    fairness_results = fairness_comparison(wachter_data, glance_data)

    print(f"   Wachter DP difference: {fairness_results['summary']['wachter_dp_diff']:.2%}")
    print(f"   GLANCE DP difference:  {fairness_results['summary']['glance_dp_diff']:.2%}")

    # FACTS fairness evaluation
    print("\n4. Running FACTS fairness evaluation...")
    facts_results = facts_evaluation(wachter_data, glance_data)

    print("\n   FACTS Report - Wachter:")
    facts_results["evaluator"].print_report(facts_results["wachter"], "Wachter")

    print("\n   FACTS Report - GLANCE:")
    facts_results["evaluator"].print_report(facts_results["glance"], "GLANCE")

    # Create visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(wachter_data, glance_data)

    # Generate report
    print("\n6. Generating comparison report...")
    generate_comparison_report(
        wachter_data, glance_data, stats_results, fairness_results, facts_results
    )

    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nResults saved to: results/comparison/")
    print("  - comparison_overview.png")
    print("  - distribution_boxplots.png")
    print("  - comparison_report.txt")


if __name__ == "__main__":
    main()
