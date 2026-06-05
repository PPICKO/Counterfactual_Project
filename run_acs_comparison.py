"""
Run comprehensive comparison analysis with FACTS framework on ACS results.

This script loads the saved ACS results and performs:
- Statistical comparison tests
- Fairness analysis with demographic parity
- FACTS framework evaluation (Equal Burden, Equal Effectiveness, Equal Choice)
- Visualization generation

Uses shared utilities from comparison_analysis module to avoid code duplication.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.comparison_analysis import (
    create_visualizations,
    facts_evaluation,
    fairness_comparison,
    generate_comparison_report,
    statistical_comparison,
)
from src.config import Config
from src.facts_fairness import FACTSEvaluator, group_results_by_attribute


def load_acs_results(
    results_dir: str = "results_acs",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load saved ACS results from both methods.

    Args:
        results_dir: Directory containing ACS results.

    Returns:
        Tuple of (wachter_data, glance_data) dictionaries.

    Raises:
        FileNotFoundError: If result files don't exist.
    """
    results_path = Path(results_dir)

    wachter_path = results_path / "wachter" / "results.pkl"
    glance_path = results_path / "glance" / "results.pkl"

    if not wachter_path.exists():
        raise FileNotFoundError(
            f"Wachter results not found at {wachter_path}. "
            "Run run_acs_experiment.py first."
        )

    if not glance_path.exists():
        raise FileNotFoundError(
            f"GLANCE results not found at {glance_path}. "
            "Run run_acs_experiment.py first."
        )

    with open(wachter_path, "rb") as f:
        wachter_data = pickle.load(f)

    with open(glance_path, "rb") as f:
        glance_data = pickle.load(f)

    return wachter_data, glance_data


def print_quick_summary(
    wachter_data: Dict[str, Any],
    glance_data: Dict[str, Any],
) -> None:
    """
    Print a quick summary of the results.

    Args:
        wachter_data: Wachter results dictionary.
        glance_data: GLANCE results dictionary.
    """
    wachter_metrics = wachter_data["metrics"]
    glance_metrics = glance_data["metrics"]

    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Wachter':<20} {'GLANCE':<20}")
    print("-" * 70)
    print(
        f"{'Validity Rate':<30} {wachter_metrics['validity_rate']:<20.2%} "
        f"{glance_metrics['validity_rate']:<20.2%}"
    )
    print(
        f"{'Avg Distance (L2)':<30} {wachter_metrics['avg_distance']:<20.4f} "
        f"{glance_metrics['avg_distance']:<20.4f}"
    )
    print(
        f"{'Avg Sparsity':<30} {wachter_metrics['avg_sparsity']:<20.2%} "
        f"{glance_metrics['avg_sparsity']:<20.2%}"
    )
    print(
        f"{'Avg Features Changed':<30} {wachter_metrics['avg_num_changes']:<20.2f} "
        f"{glance_metrics['avg_num_changes']:<20.2f}"
    )


def run_acs_comparison(
    results_dir: str = "results_acs",
) -> Optional[Dict[str, Any]]:
    """
    Run full comparison analysis on ACS results.

    Uses shared utilities from comparison_analysis module to avoid code duplication.

    Args:
        results_dir: Directory containing ACS results.

    Returns:
        Dictionary containing all comparison results, or None if results don't exist.
    """
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON ANALYSIS: WACHTER VS GLANCE (ACS DATASET)")
    print("=" * 80)

    # Load results
    print("\n1. Loading saved ACS results...")
    try:
        wachter_data, glance_data = load_acs_results(results_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the ACS experiment first:")
        print("  python run_acs_experiment.py")
        return None

    print(f"   Wachter: {len(wachter_data['results'])} samples")
    print(f"   GLANCE:  {len(glance_data['results'])} samples")

    # Print quick summary
    print_quick_summary(wachter_data, glance_data)

    # Statistical comparison (using shared utility from comparison_analysis)
    print("\n2. Running statistical tests...")
    stats_results = statistical_comparison(
        wachter_data["results"],
        glance_data["results"],
    )

    print(
        f"   Distance difference: "
        f"{abs(stats_results['distance']['wachter_mean'] - stats_results['distance']['glance_mean']):.4f}"
    )
    print(f"   Distance significant: {stats_results['distance']['significant']}")
    print(
        f"   Validity difference: "
        f"{abs(stats_results['validity']['wachter_rate'] - stats_results['validity']['glance_rate']):.2%}"
    )
    print(f"   Validity significant: {stats_results['validity']['significant']}")

    # Fairness comparison (using shared utility from comparison_analysis)
    print("\n3. Analyzing fairness metrics...")
    fairness_results = fairness_comparison(wachter_data, glance_data)

    print(f"   Wachter DP difference: {fairness_results['summary']['wachter_dp_diff']:.2%}")
    print(f"   GLANCE DP difference:  {fairness_results['summary']['glance_dp_diff']:.2%}")

    # FACTS fairness evaluation (using shared utility from comparison_analysis)
    print("\n4. Running FACTS fairness evaluation...")
    facts_results = facts_evaluation(wachter_data, glance_data)

    print("\n   FACTS Report - Wachter:")
    facts_results["evaluator"].print_report(facts_results["wachter"], "Wachter")

    print("\n   FACTS Report - GLANCE:")
    facts_results["evaluator"].print_report(facts_results["glance"], "GLANCE")

    # Create visualizations (using shared utility from comparison_analysis)
    print("\n5. Creating visualizations...")
    output_dir = Path(results_dir) / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    create_visualizations(wachter_data, glance_data, output_dir=output_dir)

    # Generate comparison report (using shared utility from comparison_analysis)
    print("\n6. Generating comparison report...")
    report_path = output_dir / "comparison_report.txt"
    generate_comparison_report(
        wachter_data,
        glance_data,
        stats_results,
        fairness_results,
        facts_results,
        output_file=report_path,
    )

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Validity winner
    wachter_validity = stats_results["validity"]["wachter_rate"]
    glance_validity = stats_results["validity"]["glance_rate"]

    if wachter_validity > glance_validity:
        validity_winner = "Wachter"
        validity_diff = wachter_validity - glance_validity
    else:
        validity_winner = "GLANCE"
        validity_diff = glance_validity - wachter_validity

    print(
        f"\n1. Validity: {validity_winner} achieves {validity_diff:.1%} higher validity rate"
    )
    sig_text = (
        "(statistically significant)"
        if stats_results["validity"]["significant"]
        else "(not significant)"
    )
    print(f"   {sig_text}")

    # Proximity winner
    wachter_dist = stats_results["distance"]["wachter_mean"]
    glance_dist = stats_results["distance"]["glance_mean"]

    if wachter_dist < glance_dist:
        dist_winner = "Wachter"
        dist_diff = glance_dist - wachter_dist
    else:
        dist_winner = "GLANCE"
        dist_diff = wachter_dist - glance_dist

    print(
        f"\n2. Proximity: {dist_winner} generates counterfactuals {dist_diff:.4f} closer"
    )
    sig_text = (
        "(statistically significant)"
        if stats_results["distance"]["significant"]
        else "(not significant)"
    )
    print(f"   {sig_text}")

    # Fairness winner
    wachter_dp = fairness_results["summary"]["wachter_dp_diff"]
    glance_dp = fairness_results["summary"]["glance_dp_diff"]

    if wachter_dp < glance_dp:
        fairness_winner = "Wachter"
        fairness_diff = glance_dp - wachter_dp
    else:
        fairness_winner = "GLANCE"
        fairness_diff = wachter_dp - glance_dp

    print(
        f"\n3. Fairness: {fairness_winner} shows {fairness_diff:.1%} better demographic parity"
    )

    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("  - comparison_overview.png")
    print("  - distribution_boxplots.png")
    print("  - comparison_report.txt")

    return {
        "wachter_data": wachter_data,
        "glance_data": glance_data,
        "stats_results": stats_results,
        "fairness_results": fairness_results,
        "facts_results": facts_results,
    }


def main() -> None:
    """Run ACS comparison analysis."""
    run_acs_comparison()


if __name__ == "__main__":
    main()
