"""
Abstract base class for counterfactual explanation generators.

Provides a common interface for all counterfactual generation methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class CounterfactualGenerator(ABC):
    """
    Abstract base class for counterfactual explanation generators.

    All counterfactual methods should inherit from this class and implement
    the required abstract methods.

    Attributes:
        classifier: Trained classifier with predict and predict_proba methods.
        feature_names: List of feature names.
        continuous_features: List of continuous feature names.
        categorical_features: List of categorical feature names.
    """

    def __init__(
        self,
        classifier: Any,
        feature_names: List[str],
        continuous_features: List[str],
        categorical_features: List[str],
    ) -> None:
        """
        Initialize the counterfactual generator.

        Args:
            classifier: Trained classifier with predict and predict_proba methods.
            feature_names: List of feature names.
            continuous_features: List of continuous feature names.
            categorical_features: List of categorical feature names.
        """
        self.classifier = classifier
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features

        # Compute feature indices
        self.continuous_indices = [
            i for i, f in enumerate(feature_names) if f in continuous_features
        ]
        self.categorical_indices = [
            i for i, f in enumerate(feature_names) if f in categorical_features
        ]

    @abstractmethod
    def generate(
        self,
        x_orig: np.ndarray,
        target_class: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a counterfactual for a single instance.

        Args:
            x_orig: Original instance (numpy array of shape (n_features,)).
            target_class: Desired target class (default: 1).
            verbose: Whether to print optimization progress.

        Returns:
            Dict containing:
                - counterfactual: Generated counterfactual instance (np.ndarray).
                - distance: L2 distance from original.
                - prediction: Predicted probability for target class.
                - prediction_proba: Full probability distribution.
                - predicted_class: Final predicted class.
                - validity: Whether prediction matches target class.
                - iterations: Number of optimization iterations.
                - converged: Whether optimization converged.
                - sparsity: Proportion of features unchanged.
                - num_changes: Number of features changed.
        """
        pass

    @abstractmethod
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
            target_class: Desired target class (default: 1).
            max_samples: Maximum number of samples to process (None = all).
            verbose: Whether to show progress bar.

        Returns:
            List of result dictionaries, one per instance.
        """
        pass

    def compute_distance(self, x_orig: np.ndarray, x_cf: np.ndarray) -> float:
        """
        Compute L2 distance between original and counterfactual.

        Args:
            x_orig: Original instance.
            x_cf: Counterfactual instance.

        Returns:
            L2 (Euclidean) distance.
        """
        return float(np.linalg.norm(x_cf - x_orig))

    def compute_sparsity(self, x_orig: np.ndarray, x_cf: np.ndarray) -> float:
        """
        Compute sparsity (proportion of features unchanged).

        Args:
            x_orig: Original instance.
            x_cf: Counterfactual instance.

        Returns:
            Sparsity value between 0 and 1 (1 = all features unchanged).
        """
        num_changes = np.sum(np.abs(x_cf - x_orig) > 1e-6)
        return 1.0 - (num_changes / len(x_orig))

    def count_changes(self, x_orig: np.ndarray, x_cf: np.ndarray) -> int:
        """
        Count the number of changed features.

        Args:
            x_orig: Original instance.
            x_cf: Counterfactual instance.

        Returns:
            Number of features that changed.
        """
        return int(np.sum(np.abs(x_cf - x_orig) > 1e-6))

    def check_validity(
        self, x_cf: np.ndarray, target_class: int
    ) -> tuple[bool, np.ndarray, int]:
        """
        Check if counterfactual achieves target class.

        Args:
            x_cf: Counterfactual instance.
            target_class: Desired target class.

        Returns:
            Tuple of (validity, prediction_proba, predicted_class).
        """
        pred_proba = self.classifier.predict_proba(x_cf.reshape(1, -1))[0]
        pred_class = int(np.argmax(pred_proba))
        validity = pred_class == target_class
        return validity, pred_proba, pred_class
