"""
GLANCE (2026) Global Counterfactual Explanation Method.

Reference:
Kavouras, L., et al. (2026).
GLANCE: Global Actions in a Nutshell for Counterfactual Explainability.
arXiv:2405.18921

Discovers global actionable recourse strategies applicable across multiple instances,
rather than generating instance-specific counterfactuals. Identifies common patterns
in feature changes that lead to desired outcomes.
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from src.classifiers import BaselineClassifier
from src.config import Config, DEFAULT_CONFIG
from src.counterfactual_base import CounterfactualGenerator
from src.data_loader import AdultDataLoader
from src.wachter_method import WachterCounterfactual


class GLANCECounterfactual(CounterfactualGenerator):
    """GLANCE global counterfactual generator.

    Simplified implementation that:
    1. Generates instance-specific counterfactuals (using Wachter)
    2. Clusters them to find global patterns
    3. Extracts global rules applicable to multiple instances
    """

    def __init__(
        self,
        classifier: Any,
        feature_names: List[str],
        continuous_features: List[str],
        categorical_features: List[str],
        n_rules: Optional[int] = None,
        lambda_param: Optional[float] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the GLANCE counterfactual generator.

        Args:
            classifier: Trained classifier.
            feature_names: List of feature names.
            continuous_features: List of continuous feature names.
            categorical_features: List of categorical feature names.
            n_rules: Number of global rules to extract (default: from config).
            lambda_param: Regularization for Wachter method (default: from config).
            config: Optional Config object for settings.
        """
        super().__init__(
            classifier=classifier,
            feature_names=feature_names,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
        )

        self.config = config or DEFAULT_CONFIG
        self.n_rules = n_rules or self.config.glance_n_rules
        self.lambda_param = lambda_param or self.config.wachter_lambda

        # Wachter generator for initial counterfactuals
        self.wachter = WachterCounterfactual(
            classifier=classifier,
            feature_names=feature_names,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            lambda_param=self.lambda_param,
            max_iter=500,
            config=self.config,
        )

        self.global_rules: List[Dict[str, Any]] = []
        self.rule_tree: Optional[DecisionTreeClassifier] = None

    def discover_global_rules(
        self,
        X: np.ndarray,
        target_class: int = 1,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Discover global counterfactual rules from dataset.

        Args:
            X: Training instances (n_samples, n_features).
            target_class: Desired target class.
            max_samples: Maximum samples to use for rule discovery (default: from config).
            verbose: Show progress.

        Returns:
            List of global rule dictionaries containing:
                - cluster_id: Cluster identifier.
                - size: Number of instances in cluster.
                - avg_delta: Average feature change direction.
                - important_features: Indices of most important features.
                - important_feature_names: Names of important features.
                - feature_changes: Dictionary of feature changes.
                - cluster_center: Cluster center coordinates.
                - validity: Rule validity rate.
        """
        max_samples = max_samples or self.config.glance_max_samples
        print(f"\nPhase 1: Generating instance-specific counterfactuals...")

        # Generate counterfactuals for samples
        if max_samples is not None and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Generate counterfactuals
        cf_results = self.wachter.generate_batch(
            X_sample,
            target_class=target_class,
            verbose=verbose,
        )

        # Extract successful counterfactuals
        valid_cfs: List[np.ndarray] = []
        valid_originals: List[np.ndarray] = []

        for i, result in enumerate(cf_results):
            if result["validity"]:
                valid_cfs.append(result["counterfactual"])
                valid_originals.append(X_sample[i])

        if len(valid_cfs) == 0:
            print("Warning: No valid counterfactuals generated!")
            return []

        valid_cfs_arr = np.array(valid_cfs)
        valid_originals_arr = np.array(valid_originals)

        print(f"\n{len(valid_cfs)} valid counterfactuals generated")
        print(f"\nPhase 2: Extracting global patterns...")

        # Compute changes (delta)
        deltas = valid_cfs_arr - valid_originals_arr

        # Cluster the deltas to find common patterns
        n_clusters = min(self.n_rules, len(deltas))

        if n_clusters < 2:
            print("Warning: Too few samples for clustering")
            n_clusters = 1

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(deltas)

        # Extract rules from clusters
        global_rules: List[Dict[str, Any]] = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_deltas = deltas[cluster_mask]
            cluster_originals = valid_originals_arr[cluster_mask]

            # Compute average change direction
            avg_delta = np.mean(cluster_deltas, axis=0)

            # Identify most important features (largest average change)
            abs_delta = np.abs(avg_delta)
            important_features = np.argsort(abs_delta)[::-1][:5]  # Top 5 features

            # Create rule
            rule: Dict[str, Any] = {
                "cluster_id": cluster_id,
                "size": int(cluster_mask.sum()),
                "avg_delta": avg_delta,
                "important_features": important_features,
                "important_feature_names": [
                    self.feature_names[i] for i in important_features
                ],
                "feature_changes": {
                    self.feature_names[i]: float(avg_delta[i])
                    for i in important_features
                },
                "cluster_center": kmeans.cluster_centers_[cluster_id],
            }

            # Compute rule validity
            # Apply rule to cluster originals and check predictions
            applied_cfs = cluster_originals + avg_delta
            rule_predictions = self.classifier.predict(applied_cfs)
            rule_validity = float((rule_predictions == target_class).mean())

            rule["validity"] = rule_validity

            global_rules.append(rule)

        # Sort rules by size (coverage)
        global_rules = sorted(global_rules, key=lambda x: x["size"], reverse=True)

        self.global_rules = global_rules

        # Train decision tree to represent rules
        self._train_rule_tree(deltas, cluster_labels)

        return global_rules

    def _train_rule_tree(
        self,
        deltas: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> None:
        """
        Train decision tree to represent global rules.

        Args:
            deltas: Feature change vectors.
            cluster_labels: Cluster assignments.
        """
        self.rule_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        self.rule_tree.fit(deltas, cluster_labels)

    def apply_global_rule(
        self,
        X: np.ndarray,
        rule_id: int = 0,
    ) -> np.ndarray:
        """
        Apply a global rule to instances.

        Args:
            X: Instances to apply rule to (n_samples, n_features).
            rule_id: Which global rule to apply.

        Returns:
            Counterfactual instances.

        Raises:
            ValueError: If rule_id is out of range.
        """
        if rule_id >= len(self.global_rules):
            raise ValueError(
                f"Rule {rule_id} not found. Only {len(self.global_rules)} rules available."
            )

        rule = self.global_rules[rule_id]

        # Apply average delta
        X_cf = X + rule["avg_delta"]

        return X_cf

    def generate(
        self,
        x_orig: np.ndarray,
        target_class: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual for a single instance using global rules.

        Args:
            x_orig: Original instance (numpy array).
            target_class: Desired target class (0 or 1).
            verbose: Print progress (unused for single instance).

        Returns:
            Dictionary with counterfactual result.
        """
        if len(self.global_rules) == 0:
            raise ValueError(
                "No global rules discovered. Call discover_global_rules first."
            )

        # Try each rule and pick the best
        best_result: Optional[Dict[str, Any]] = None
        best_validity = False
        best_distance = float("inf")

        for rule_id, rule in enumerate(self.global_rules):
            # Apply rule
            x_cf = x_orig + rule["avg_delta"]

            # Predict
            pred_proba = self.classifier.predict_proba(x_cf.reshape(1, -1))[0]
            pred_class = int(np.argmax(pred_proba))

            # Compute metrics
            distance = self.compute_distance(x_orig, x_cf)
            validity = pred_class == target_class
            sparsity = self.compute_sparsity(x_orig, x_cf)
            num_changes = self.count_changes(x_orig, x_cf)

            # Track best
            if validity and distance < best_distance:
                best_validity = True
                best_distance = distance
                best_result = {
                    "counterfactual": x_cf,
                    "distance": distance,
                    "prediction": float(pred_proba[target_class]),
                    "prediction_proba": pred_proba,
                    "predicted_class": pred_class,
                    "validity": validity,
                    "rule_id": rule_id,
                    "rule_name": f"Rule {rule_id}",
                    "sparsity": sparsity,
                    "num_changes": num_changes,
                    "converged": True,
                    "iterations": 1,
                }
            elif not best_validity and distance < best_distance:
                # If no valid rule found yet, track closest
                best_distance = distance
                best_result = {
                    "counterfactual": x_cf,
                    "distance": distance,
                    "prediction": float(pred_proba[target_class]),
                    "prediction_proba": pred_proba,
                    "predicted_class": pred_class,
                    "validity": validity,
                    "rule_id": rule_id,
                    "rule_name": f"Rule {rule_id}",
                    "sparsity": sparsity,
                    "num_changes": num_changes,
                    "converged": True,
                    "iterations": 1,
                }

        return best_result

    def generate_batch(
        self,
        X: np.ndarray,
        target_class: int = 1,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactuals for a batch using global rules.

        Alias for generate_with_rules for API compatibility.

        Args:
            X: Array of instances (n_samples, n_features).
            target_class: Desired target class.
            max_samples: Maximum number of samples to process (None = all).
            verbose: Show progress bar.

        Returns:
            List of result dictionaries.
        """
        return self.generate_with_rules(X, target_class, verbose)

    def generate_with_rules(
        self,
        X: np.ndarray,
        target_class: int = 1,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactuals by applying global rules.

        Args:
            X: Instances (n_samples, n_features).
            target_class: Desired target class.
            verbose: Show progress.

        Returns:
            List of result dictionaries.

        Raises:
            ValueError: If no global rules have been discovered.
        """
        if len(self.global_rules) == 0:
            raise ValueError(
                "No global rules discovered. Call discover_global_rules first."
            )

        results: List[Dict[str, Any]] = []

        iterator = tqdm(X, desc="Applying global rules") if verbose else X

        for x in iterator:
            result = self.generate(x, target_class=target_class, verbose=False)
            results.append(result)

        return results

    def print_global_rules(self) -> None:
        """Print discovered global rules in human-readable format."""
        print("\n" + "=" * 60)
        print("DISCOVERED GLOBAL COUNTERFACTUAL RULES")
        print("=" * 60)

        for i, rule in enumerate(self.global_rules):
            print(
                f"\nRule {i} (covers {rule['size']} instances, {rule['validity']:.1%} valid):"
            )
            print(f"  Top feature changes:")

            for feature_name in rule["important_feature_names"]:
                change = rule["feature_changes"][feature_name]
                direction = "increase" if change > 0 else "decrease"
                print(f"    - {feature_name}: {direction} by {abs(change):.3f}")


def evaluate_glance(
    results: List[Dict[str, Any]],
    X_orig: np.ndarray,
    protected_attr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate GLANCE counterfactual quality.

    Args:
        results: List of result dictionaries from generate_with_rules.
        X_orig: Original instances.
        protected_attr: Protected attribute values (optional).

    Returns:
        Dictionary with evaluation metrics including rule_usage.
    """
    from src.wachter_method import evaluate_counterfactuals

    # Use same evaluation as Wachter for comparison
    metrics = evaluate_counterfactuals(results, X_orig, protected_attr)

    # Add GLANCE-specific metrics
    rule_usage: Dict[int, int] = defaultdict(int)
    for r in results:
        if "rule_id" in r:
            rule_usage[r["rule_id"]] += 1

    metrics["rule_usage"] = dict(rule_usage)

    print("\nRule Usage Distribution:")
    for rule_id, count in sorted(rule_usage.items()):
        print(f"  Rule {rule_id}: {count} instances ({count/len(results):.1%})")

    return metrics


def main() -> None:
    """Test GLANCE global counterfactual method."""
    print("Loading dataset and trained classifier...")
    loader = AdultDataLoader(data_dir="../data")
    dataset = loader.load_processed_data()

    # Load trained classifier
    clf = BaselineClassifier.load("../results/models/logistic_classifier.pkl")

    print("\n" + "=" * 60)
    print("GLANCE GLOBAL COUNTERFACTUAL GENERATION")
    print("=" * 60)

    # Select samples with negative prediction
    y_pred = clf.predict(dataset["X_test"])
    negative_mask = y_pred == 0
    X_negative = dataset["X_test"][negative_mask]
    protected_negative = dataset["protected_test"]["sex"][negative_mask]

    print(f"\nFound {len(X_negative)} samples with negative prediction")

    # Initialize GLANCE
    glance = GLANCECounterfactual(
        classifier=clf,
        feature_names=dataset["feature_names"],
        continuous_features=dataset["continuous_features"],
        categorical_features=dataset["categorical_features"],
        n_rules=5,
        lambda_param=0.1,
    )

    # Discover global rules
    print(f"\nDiscovering global rules from {min(200, len(X_negative))} samples...")
    global_rules = glance.discover_global_rules(
        X_negative,
        target_class=1,
        max_samples=200,
        verbose=True,
    )

    # Print rules
    glance.print_global_rules()

    # Apply rules to test set
    print(f"\n\nApplying global rules to 100 test samples...")
    results = glance.generate_with_rules(
        X_negative[:100],
        target_class=1,
        verbose=True,
    )

    # Evaluate
    metrics = evaluate_glance(
        results,
        X_negative[:100],
        protected_attr=protected_negative[:100],
    )

    # Save results
    output_dir = Path("../results/glance")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": results,
                "metrics": metrics,
                "global_rules": global_rules,
                "X_orig": X_negative[:100],
                "protected": protected_negative[:100],
            },
            f,
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
