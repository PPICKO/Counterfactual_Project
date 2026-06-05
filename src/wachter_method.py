"""
Wachter et al. (2017) Counterfactual Explanation Method.

Reference:
Wachter, S., Mittelstadt, B., & Russell, C. (2017).
Counterfactual explanations without opening the black box:
Automated decisions and the GDPR. arXiv:1711.00399

Generates instance-specific counterfactuals by solving:
    min ||x' - x||^2 + lambda * loss(f(x'), y')
where:
    x: original instance
    x': counterfactual instance
    f: classifier
    y': desired target class
    lambda: regularization parameter
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from src.classifiers import BaselineClassifier
from src.config import Config, DEFAULT_CONFIG
from src.counterfactual_base import CounterfactualGenerator
from src.data_loader import AdultDataLoader


class WachterCounterfactual(CounterfactualGenerator):
    """Wachter et al. (2017) counterfactual generator."""

    def __init__(
        self,
        classifier: Any,
        feature_names: List[str],
        continuous_features: List[str],
        categorical_features: List[str],
        lambda_param: Optional[float] = None,
        lr: Optional[float] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the Wachter counterfactual generator.

        Args:
            classifier: Trained classifier with predict_proba method.
            feature_names: List of feature names.
            continuous_features: List of continuous feature names.
            categorical_features: List of categorical feature names.
            lambda_param: Weight for proximity loss (default: from config).
            lr: Learning rate for optimization (default: from config).
            max_iter: Maximum iterations (default: from config).
            tolerance: Convergence tolerance (default: from config).
            config: Optional Config object for settings.
        """
        super().__init__(
            classifier=classifier,
            feature_names=feature_names,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
        )

        self.config = config or DEFAULT_CONFIG
        self.lambda_param = lambda_param or self.config.wachter_lambda
        self.lr = lr or self.config.wachter_lr
        self.max_iter = max_iter or self.config.wachter_max_iter
        self.tolerance = tolerance or self.config.wachter_tolerance
        self.epsilon = self.config.gradient_epsilon

    def _compute_gradient_batch(
        self,
        x_cf: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """
        Compute gradient using batch finite differences.

        This is much more efficient than the loop-based approach as it
        makes 2 predictions instead of 2*n_features predictions.

        Args:
            x_cf: Current counterfactual estimate.
            target_class: Desired target class.

        Returns:
            Gradient array with respect to prediction loss.
        """
        n_features = len(x_cf)

        # Create perturbation matrices for all features at once
        # x_plus[i] = x_cf with feature i increased by epsilon
        # x_minus[i] = x_cf with feature i decreased by epsilon
        x_plus = np.tile(x_cf, (n_features, 1))
        x_minus = np.tile(x_cf, (n_features, 1))

        # Apply perturbations along diagonal
        np.fill_diagonal(x_plus, x_cf + self.epsilon)
        np.fill_diagonal(x_minus, x_cf - self.epsilon)

        # Batch predict for all perturbations
        y_plus = self.classifier.predict_proba(x_plus)[:, target_class]
        y_minus = self.classifier.predict_proba(x_minus)[:, target_class]

        # Compute gradient using central differences
        # Negative because we want to maximize probability of target class
        grad = -(y_plus - y_minus) / (2 * self.epsilon)

        return grad

    def generate(
        self,
        x_orig: np.ndarray,
        target_class: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual for a single instance.

        Args:
            x_orig: Original instance (numpy array).
            target_class: Desired target class (0 or 1).
            verbose: Print optimization progress.

        Returns:
            Dictionary with:
                - counterfactual: Generated counterfactual instance (np.ndarray).
                - distance: L2 distance from original.
                - prediction: Predicted probability for target class.
                - prediction_proba: Full probability distribution.
                - predicted_class: Final predicted class.
                - validity: Whether prediction matches target.
                - iterations: Number of optimization iterations.
                - converged: Whether optimization converged.
                - sparsity: Proportion of features unchanged.
                - num_changes: Number of features changed.
        """
        # Initialize counterfactual as copy of original
        x_cf = x_orig.copy().astype(np.float64)
        x_orig_float = x_orig.astype(np.float64)

        # Track best counterfactual
        best_cf: Optional[np.ndarray] = None
        best_loss = float("inf")
        best_pred = 0.0

        converged = False
        prev_loss = float("inf")
        iteration = 0

        for iteration in range(self.max_iter):
            # Compute current prediction
            y_pred_proba = self.classifier.predict_proba(x_cf.reshape(1, -1))[0]
            y_pred_target = y_pred_proba[target_class]

            # Compute losses
            pred_loss = (1.0 - y_pred_target) ** 2
            proximity_loss = np.sum((x_cf - x_orig_float) ** 2)
            total_loss = pred_loss + self.lambda_param * proximity_loss

            # Track best
            if total_loss < best_loss:
                best_loss = total_loss
                best_cf = x_cf.copy()
                best_pred = y_pred_target

            # Check convergence
            if abs(prev_loss - total_loss) < self.tolerance:
                converged = True
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_loss = total_loss

            # Compute gradient using efficient batch method
            pred_grad = self._compute_gradient_batch(x_cf, target_class)

            # Add proximity gradient
            grad = pred_grad + 2 * self.lambda_param * (x_cf - x_orig_float)

            # Update
            x_cf = x_cf - self.lr * grad

            # Progress
            if verbose and iteration % 100 == 0:
                print(
                    f"Iter {iteration}: Loss={total_loss:.4f}, "
                    f"Pred={y_pred_target:.4f}, "
                    f"Dist={proximity_loss:.4f}"
                )

        # Use best counterfactual found
        if best_cf is None:
            best_cf = x_cf.copy()

        # Final prediction
        final_pred_proba = self.classifier.predict_proba(best_cf.reshape(1, -1))[0]
        final_pred_class = int(np.argmax(final_pred_proba))

        # Compute metrics
        distance = self.compute_distance(x_orig, best_cf)
        sparsity = self.compute_sparsity(x_orig, best_cf)
        num_changes = self.count_changes(x_orig, best_cf)

        result: Dict[str, Any] = {
            "counterfactual": best_cf,
            "distance": distance,
            "prediction": float(final_pred_proba[target_class]),
            "prediction_proba": final_pred_proba,
            "predicted_class": final_pred_class,
            "validity": final_pred_class == target_class,
            "iterations": iteration + 1,
            "converged": converged,
            "sparsity": sparsity,
            "num_changes": num_changes,
        }

        return result

    def generate_batch(
        self,
        X: np.ndarray,
        target_class: int = 1,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactuals for a batch of instances.

        Args:
            X: Array of instances (n_samples, n_features).
            target_class: Desired target class.
            max_samples: Maximum number of samples to process (None = all).
            verbose: Show progress bar.

        Returns:
            List of result dictionaries, one per instance.
        """
        if max_samples is not None:
            X = X[:max_samples]

        results: List[Dict[str, Any]] = []
        iterator = tqdm(X, desc="Generating counterfactuals") if verbose else X

        for x in iterator:
            result = self.generate(x, target_class=target_class, verbose=False)
            results.append(result)

        return results


def evaluate_counterfactuals(
    results: List[Dict[str, Any]],
    X_orig: np.ndarray,
    protected_attr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate counterfactual quality metrics.

    Args:
        results: List of result dictionaries from generate_batch.
        X_orig: Original instances.
        protected_attr: Protected attribute values (optional, for fairness analysis).

    Returns:
        Dictionary with aggregate metrics:
            - n_samples: Number of samples.
            - validity_rate: Percentage achieving target class.
            - avg_distance: Average L2 distance.
            - std_distance: Standard deviation of distances.
            - avg_sparsity: Average proportion of features unchanged.
            - avg_num_changes: Average number of features changed.
            - convergence_rate: Percentage that converged.
            - avg_iterations: Average optimization iterations.
            - group_metrics: Per-group metrics (if protected_attr provided).
    """
    n_samples = len(results)

    # Validity: percentage of counterfactuals that achieve target class
    validity_rate = float(np.mean([r["validity"] for r in results]))

    # Proximity: average L2 distance
    avg_distance = float(np.mean([r["distance"] for r in results]))
    std_distance = float(np.std([r["distance"] for r in results]))

    # Sparsity: average percentage of features unchanged
    avg_sparsity = float(np.mean([r["sparsity"] for r in results]))
    avg_num_changes = float(np.mean([r["num_changes"] for r in results]))

    # Convergence: percentage that converged
    convergence_rate = float(np.mean([r["converged"] for r in results]))

    # Average iterations
    avg_iterations = float(np.mean([r["iterations"] for r in results]))

    metrics: Dict[str, Any] = {
        "n_samples": n_samples,
        "validity_rate": validity_rate,
        "avg_distance": avg_distance,
        "std_distance": std_distance,
        "avg_sparsity": avg_sparsity,
        "avg_num_changes": avg_num_changes,
        "convergence_rate": convergence_rate,
        "avg_iterations": avg_iterations,
    }

    print("\n" + "=" * 60)
    print("WACHTER COUNTERFACTUAL EVALUATION")
    print("=" * 60)
    print(f"Samples:              {n_samples}")
    print(f"Validity Rate:        {validity_rate:.2%}")
    print(f"Avg Distance (L2):    {avg_distance:.4f} +/- {std_distance:.4f}")
    print(f"Avg Sparsity:         {avg_sparsity:.2%}")
    print(f"Avg Features Changed: {avg_num_changes:.2f}")
    print(f"Convergence Rate:     {convergence_rate:.2%}")
    print(f"Avg Iterations:       {avg_iterations:.1f}")

    # Fairness analysis if protected attribute provided
    if protected_attr is not None:
        print("\nFairness Analysis:")
        unique_groups = np.unique(protected_attr)

        group_metrics: Dict[str, Dict[str, Any]] = {}
        for group in unique_groups:
            mask = protected_attr == group
            group_results = [r for i, r in enumerate(results) if mask[i]]

            if len(group_results) == 0:
                continue

            group_validity = float(np.mean([r["validity"] for r in group_results]))

            # Handle empty valid results for distance calculation
            valid_group_results = [r for r in group_results if r["validity"]]
            group_distance = (
                float(np.mean([r["distance"] for r in valid_group_results]))
                if valid_group_results
                else np.nan
            )
            group_sparsity = float(np.mean([r["sparsity"] for r in group_results]))

            group_metrics[str(group)] = {
                "count": len(group_results),
                "validity": group_validity,
                "distance": group_distance,
                "sparsity": group_sparsity,
            }

            print(f"\n  {group}:")
            print(f"    Samples:      {len(group_results)}")
            print(f"    Validity:     {group_validity:.2%}")
            if not np.isnan(group_distance):
                print(f"    Avg Distance: {group_distance:.4f}")
            print(f"    Avg Sparsity: {group_sparsity:.2%}")

        metrics["group_metrics"] = group_metrics

    return metrics


def main() -> None:
    """Test Wachter counterfactual method."""
    print("Loading dataset and trained classifier...")
    loader = AdultDataLoader(data_dir="../data")
    dataset = loader.load_processed_data()

    # Load trained classifier
    clf = BaselineClassifier.load("../results/models/logistic_classifier.pkl")

    print("\n" + "=" * 60)
    print("WACHTER COUNTERFACTUAL GENERATION")
    print("=" * 60)

    # Select samples with negative prediction for counterfactual generation
    y_pred = clf.predict(dataset["X_test"])
    negative_mask = y_pred == 0
    X_negative = dataset["X_test"][negative_mask]
    protected_negative = dataset["protected_test"]["sex"][negative_mask]

    print(f"\nFound {len(X_negative)} samples with negative prediction")
    print(f"Generating counterfactuals for first 100 samples...")

    # Initialize generator
    generator = WachterCounterfactual(
        classifier=clf,
        feature_names=dataset["feature_names"],
        continuous_features=dataset["continuous_features"],
        categorical_features=dataset["categorical_features"],
        lambda_param=0.1,
        lr=0.01,
        max_iter=500,
    )

    # Generate counterfactuals
    results = generator.generate_batch(
        X_negative[:100],
        target_class=1,
        verbose=True,
    )

    # Evaluate
    metrics = evaluate_counterfactuals(
        results,
        X_negative[:100],
        protected_attr=protected_negative[:100],
    )

    # Save results
    output_dir = Path("../results/wachter")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": results,
                "metrics": metrics,
                "X_orig": X_negative[:100],
                "protected": protected_negative[:100],
            },
            f,
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
