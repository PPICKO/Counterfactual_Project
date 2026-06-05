"""
Run the full Wachter-vs-GLANCE comparison analysis on Adult results.

Mirrors run_acs_comparison.py exactly but reads from results_adult/ and
writes a metrics_summary.json alongside the standard report/plots so the
artifact layout matches results_acs/.
"""

from __future__ import annotations

import json
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


def load_adult_results(
    results_dir: str = "results_adult",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load saved Adult results from both methods."""
    results_path = Path(results_dir)
    wachter_path = results_path / "wachter" / "results.pkl"
    glance_path = results_path / "glance" / "results.pkl"

    if not wachter_path.exists():
        raise FileNotFoundError(
            f"Wachter results not found at {wachter_path}. "
            "Run run_adult_experiment.py first."
        )
    if not glance_path.exists():
        raise FileNotFoundError(
            f"GLANCE results not found at {glance_path}. "
            "Run run_adult_experiment.py first."
        )

    with open(wachter_path, "rb") as f:
        wachter_data = pickle.load(f)
    with open(glance_path, "rb") as f:
        glance_data = pickle.load(f)

    return wachter_data, glance_data


def print_quick_summary(
    wachter_data: Dict[str, Any], glance_data: Dict[str, Any]
) -> None:
    w, g = wachter_data["metrics"], glance_data["metrics"]
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Wachter':<20} {'GLANCE':<20}")
    print("-" * 70)
    print(f"{'Validity Rate':<30} {w['validity_rate']:<20.2%} {g['validity_rate']:<20.2%}")
    print(f"{'Avg Distance (L2)':<30} {w['avg_distance']:<20.4f} {g['avg_distance']:<20.4f}")
    print(f"{'Avg Sparsity':<30} {w['avg_sparsity']:<20.2%} {g['avg_sparsity']:<20.2%}")
    print(
        f"{'Avg Features Changed':<30} {w['avg_num_changes']:<20.2f} "
        f"{g['avg_num_changes']:<20.2f}"
    )


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / non-serializable types for JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def write_metrics_summary(
    output_path: Path,
    wachter_data: Dict[str, Any],
    glance_data: Dict[str, Any],
    stats_results: Dict[str, Any],
    fairness_results: Dict[str, Any],
) -> None:
    payload = {
        "project": "Counterfactual_Project (Adult Dataset)",
        "wachter_metrics": wachter_data["metrics"],
        "glance_metrics": glance_data["metrics"],
        "stats_results": stats_results,
        "fairness_summary": {
            "wachter_dp_diff": fairness_results["summary"]["wachter_dp_diff"],
            "glance_dp_diff": fairness_results["summary"]["glance_dp_diff"],
            "wachter_cost_disparity": fairness_results["summary"].get(
                "wachter_cost_disparity"
            ),
            "glance_cost_disparity": fairness_results["summary"].get(
                "glance_cost_disparity"
            ),
            "wachter_statistical_tests": fairness_results.get(
                "wachter_statistical_tests"
            ),
            "glance_statistical_tests": fairness_results.get(
                "glance_statistical_tests"
            ),
        },
    }
    with open(output_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


def run_adult_comparison(
    results_dir: str = "results_adult",
) -> Optional[Dict[str, Any]]:
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON ANALYSIS: WACHTER VS GLANCE (ADULT DATASET)")
    print("=" * 80)

    print("\n1. Loading saved Adult results...")
    try:
        wachter_data, glance_data = load_adult_results(results_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return None

    print(f"   Wachter: {len(wachter_data['results'])} samples")
    print(f"   GLANCE:  {len(glance_data['results'])} samples")

    print_quick_summary(wachter_data, glance_data)

    print("\n2. Running statistical tests...")
    stats_results = statistical_comparison(
        wachter_data["results"], glance_data["results"]
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

    print("\n3. Analyzing fairness metrics...")
    fairness_results = fairness_comparison(wachter_data, glance_data)
    print(
        f"   Wachter DP difference: {fairness_results['summary']['wachter_dp_diff']:.2%}"
    )
    print(
        f"   GLANCE DP difference:  {fairness_results['summary']['glance_dp_diff']:.2%}"
    )

    print("\n4. Running FACTS fairness evaluation...")
    facts_results = facts_evaluation(wachter_data, glance_data)
    print("\n   FACTS Report - Wachter:")
    facts_results["evaluator"].print_report(facts_results["wachter"], "Wachter")
    print("\n   FACTS Report - GLANCE:")
    facts_results["evaluator"].print_report(facts_results["glance"], "GLANCE")

    print("\n5. Creating visualizations...")
    output_dir = Path(results_dir) / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    create_visualizations(wachter_data, glance_data, output_dir=output_dir)

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

    print("\n7. Writing metrics_summary.json...")
    write_metrics_summary(
        output_dir / "metrics_summary.json",
        wachter_data,
        glance_data,
        stats_results,
        fairness_results,
    )

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    w_val, g_val = (
        stats_results["validity"]["wachter_rate"],
        stats_results["validity"]["glance_rate"],
    )
    val_winner, val_diff = ("Wachter", w_val - g_val) if w_val > g_val else ("GLANCE", g_val - w_val)
    sig = (
        "(statistically significant)"
        if stats_results["validity"]["significant"]
        else "(not significant)"
    )
    print(f"\n1. Validity: {val_winner} achieves {val_diff:.1%} higher validity rate")
    print(f"   {sig}")

    w_d, g_d = (
        stats_results["distance"]["wachter_mean"],
        stats_results["distance"]["glance_mean"],
    )
    dist_winner, dist_diff = ("Wachter", g_d - w_d) if w_d < g_d else ("GLANCE", w_d - g_d)
    sig = (
        "(statistically significant)"
        if stats_results["distance"]["significant"]
        else "(not significant)"
    )
    print(f"\n2. Proximity: {dist_winner} generates counterfactuals {dist_diff:.4f} closer")
    print(f"   {sig}")

    w_dp, g_dp = (
        fairness_results["summary"]["wachter_dp_diff"],
        fairness_results["summary"]["glance_dp_diff"],
    )
    fair_winner, fair_diff = ("Wachter", g_dp - w_dp) if w_dp < g_dp else ("GLANCE", w_dp - g_dp)
    print(f"\n3. Fairness: {fair_winner} shows {fair_diff:.1%} better demographic parity")

    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("  - comparison_overview.png")
    print("  - distribution_boxplots.png")
    print("  - comparison_report.txt")
    print("  - metrics_summary.json")

    return {
        "wachter_data": wachter_data,
        "glance_data": glance_data,
        "stats_results": stats_results,
        "fairness_results": fairness_results,
        "facts_results": facts_results,
    }


def main() -> None:
    run_adult_comparison()


if __name__ == "__main__":
    main()
