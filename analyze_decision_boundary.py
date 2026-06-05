"""
Analyze decision boundary distances to understand ACS validity collapse.

This script investigates whether the 13% validity on ACS is due to:
1. Test instances being far from decision boundary
2. Sampling artifacts
3. Fundamental difference in decision boundary geometry
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def analyze_boundary_distances(
    model: Any,
    X_orig: np.ndarray,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze the relationship between boundary distance and counterfactual validity.

    Args:
        model: Trained classifier with predict_proba method.
        X_orig: Original instances.
        results: List of counterfactual result dictionaries.

    Returns:
        Dictionary with analysis results including distributions and statistics.
    """
    # Get prediction probabilities for original instances
    proba_orig = model.predict_proba(X_orig)[:, 1]

    # Distance from boundary
    boundary_dist = np.abs(proba_orig - 0.5)

    # Validity mask
    valid_mask = np.array([r["validity"] for r in results])

    analysis: Dict[str, Any] = {
        "n_samples": len(X_orig),
        "proba_stats": {
            "mean": float(proba_orig.mean()),
            "std": float(proba_orig.std()),
            "min": float(proba_orig.min()),
            "max": float(proba_orig.max()),
        },
        "boundary_distance_stats": {
            "mean": float(boundary_dist.mean()),
            "median": float(np.median(boundary_dist)),
        },
        "validity_rate": float(valid_mask.mean()),
    }

    return analysis


def main() -> None:
    """Analyze decision boundary for ACS dataset counterfactuals."""
    results_dir = Path("results_acs")

    # Load results
    with open(results_dir / "wachter" / "results.pkl", "rb") as f:
        wachter_data = pickle.load(f)

    with open(results_dir / "classifier.pkl", "rb") as f:
        model = pickle.load(f)

    X_orig = wachter_data["X_orig"]
    results = wachter_data["results"]

    # Analyze decision boundary distances
    print("=" * 80)
    print("DECISION BOUNDARY ANALYSIS - ACS Dataset")
    print("=" * 80)

    # Get prediction probabilities for original instances
    proba_orig = model.predict_proba(X_orig)[:, 1]

    print(f"\n1. Original Instance Characteristics (n={len(X_orig)})")
    print(f"   Avg probability of positive class: {proba_orig.mean():.4f}")
    print(f"   Std probability: {proba_orig.std():.4f}")
    print(f"   Min probability: {proba_orig.min():.4f}")
    print(f"   Max probability: {proba_orig.max():.4f}")
    print(f"   Distance from boundary (|P-0.5|):")
    print(f"     Mean: {np.abs(proba_orig - 0.5).mean():.4f}")
    print(f"     Median: {np.median(np.abs(proba_orig - 0.5)):.4f}")

    # Analyze by validity
    valid_mask = np.array([r["validity"] for r in results])

    print(f"\n2. Valid vs Invalid Counterfactuals")
    print(f"   Valid instances: {valid_mask.sum()} ({valid_mask.mean():.2%})")
    print(f"   Invalid instances: {(~valid_mask).sum()} ({(~valid_mask).mean():.2%})")

    if valid_mask.sum() > 0:
        print(f"\n   Valid instances - original probabilities:")
        print(f"     Mean: {proba_orig[valid_mask].mean():.4f}")
        print(
            f"     Distance from boundary: {np.abs(proba_orig[valid_mask] - 0.5).mean():.4f}"
        )

    if (~valid_mask).sum() > 0:
        print(f"\n   Invalid instances - original probabilities:")
        print(f"     Mean: {proba_orig[~valid_mask].mean():.4f}")
        print(
            f"     Distance from boundary: {np.abs(proba_orig[~valid_mask] - 0.5).mean():.4f}"
        )

    # Statistical test
    if valid_mask.sum() > 0 and (~valid_mask).sum() > 0:
        boundary_dist_valid = np.abs(proba_orig[valid_mask] - 0.5)
        boundary_dist_invalid = np.abs(proba_orig[~valid_mask] - 0.5)
        t_stat, p_val = stats.ttest_ind(boundary_dist_valid, boundary_dist_invalid)
        print(f"\n   t-test for boundary distance difference:")
        print(f"     t-statistic: {t_stat:.4f}")
        print(f"     p-value: {p_val:.6f}")
        print(f"     Significant: {'YES' if p_val < 0.05 else 'NO'}")

    # Analyze counterfactual distances
    print(f"\n3. Counterfactual Distance Analysis")
    valid_cfs = [r for r in results if r["validity"]]
    invalid_cfs = [r for r in results if not r["validity"]]

    if valid_cfs:
        valid_distances = [r["distance"] for r in valid_cfs]
        print(
            f"   Valid CF distances: {np.mean(valid_distances):.4f} +/- {np.std(valid_distances):.4f}"
        )

    if invalid_cfs:
        invalid_distances = [r["distance"] for r in invalid_cfs]
        print(
            f"   Invalid CF distances: {np.mean(invalid_distances):.4f} +/- {np.std(invalid_distances):.4f}"
        )

    # Analyze iterations
    valid_iters = [r["iterations"] for r in valid_cfs] if valid_cfs else []
    invalid_iters = [r["iterations"] for r in invalid_cfs] if invalid_cfs else []

    if valid_iters:
        print(f"\n4. Optimization Iterations")
        print(
            f"   Valid CF iterations: {np.mean(valid_iters):.1f} +/- {np.std(valid_iters):.1f}"
        )
    if invalid_iters:
        print(
            f"   Invalid CF iterations: {np.mean(invalid_iters):.1f} +/- {np.std(invalid_iters):.1f}"
        )

    # Check if instances are "impossible cases"
    print(f"\n5. Impossible Cases Analysis")
    very_far = np.abs(proba_orig - 0.5) > 0.4
    print(
        f"   Instances very far from boundary (|P-0.5| > 0.4): "
        f"{very_far.sum()} ({very_far.mean():.2%})"
    )
    if very_far.sum() > 0:
        print(f"   Validity rate for very far instances: {valid_mask[very_far].mean():.2%}")

    close = np.abs(proba_orig - 0.5) < 0.2
    print(
        f"   Instances close to boundary (|P-0.5| < 0.2): "
        f"{close.sum()} ({close.mean():.2%})"
    )
    if close.sum() > 0:
        print(f"   Validity rate for close instances: {valid_mask[close].mean():.2%}")

    # Distribution quartiles
    print(f"\n6. Validity by Boundary Distance Quartiles")
    boundary_dist = np.abs(proba_orig - 0.5)
    quartiles = np.percentile(boundary_dist, [25, 50, 75])

    for i, (low, high) in enumerate(
        [
            (0, quartiles[0]),
            (quartiles[0], quartiles[1]),
            (quartiles[1], quartiles[2]),
            (quartiles[2], 0.5),
        ]
    ):
        mask = (boundary_dist >= low) & (boundary_dist < high)
        if mask.sum() > 0:
            print(
                f"   Q{i+1} ({low:.3f} to {high:.3f}): "
                f"{mask.sum()} instances, {valid_mask[mask].mean():.2%} validity"
            )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
