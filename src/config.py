"""
Configuration management for counterfactual experiments.

Provides a centralized Config dataclass for paths, hyperparameters, and random seeds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """
    Configuration settings for counterfactual experiments.

    Attributes:
        data_dir: Directory for storing datasets.
        results_dir: Directory for storing experiment results.
        random_seed: Random seed for reproducibility.
        test_size: Proportion of data for test set.
        val_size: Proportion of remaining data for validation set.
        wachter_lambda: Regularization parameter for Wachter method.
        wachter_lr: Learning rate for Wachter optimization.
        wachter_max_iter: Maximum iterations for Wachter method.
        wachter_tolerance: Convergence tolerance for Wachter method.
        glance_n_rules: Number of global rules to extract in GLANCE.
        glance_max_samples: Maximum samples for GLANCE rule discovery.
        classifier_max_iter: Maximum iterations for classifier training.
        n_estimators: Number of trees for Random Forest.
        max_depth: Maximum depth for Random Forest trees.
        alpha: Significance level for statistical tests.
        target_effectiveness: Target success rate for FACTS evaluation.
        gradient_epsilon: Epsilon for finite difference gradient approximation.
        num_test_samples: Number of test samples for counterfactual generation.
        states: List of US state codes for ACS data loading.
    """

    # Path configurations
    data_dir: str = "data"
    results_dir: str = "results"

    # Random seed for reproducibility
    random_seed: int = 42

    # Data split configurations
    test_size: float = 0.2
    val_size: float = 0.2

    # Wachter method hyperparameters
    wachter_lambda: float = 0.1
    wachter_lr: float = 0.01
    wachter_max_iter: int = 1000
    wachter_tolerance: float = 1e-3
    gradient_epsilon: float = 1e-5

    # GLANCE method hyperparameters
    glance_n_rules: int = 5
    glance_max_samples: int = 200

    # Classifier hyperparameters
    classifier_max_iter: int = 1000
    n_estimators: int = 100
    max_depth: int = 10

    # Statistical test configurations
    alpha: float = 0.05

    # FACTS evaluation configurations
    target_effectiveness: float = 0.8

    # Experiment configurations
    num_test_samples: int = 1000

    # ACS data configurations
    states: List[str] = field(default_factory=lambda: ["CA"])
    survey_year: str = "2018"
    horizon: str = "1-Year"
    survey: str = "person"

    def get_data_path(self) -> Path:
        """Get the data directory as a Path object.

        Returns:
            Path: The data directory path.
        """
        return Path(self.data_dir)

    def get_results_path(self) -> Path:
        """Get the results directory as a Path object.

        Returns:
            Path: The results directory path.
        """
        return Path(self.results_dir)

    def ensure_directories(self) -> None:
        """Create data and results directories if they don't exist."""
        self.get_data_path().mkdir(parents=True, exist_ok=True)
        self.get_results_path().mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = Config()
