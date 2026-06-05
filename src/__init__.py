"""
Counterfactual Explanations Comparison Package.

This package provides implementations of counterfactual explanation methods
and tools for comparing their fairness across protected groups.

Modules:
    - config: Configuration management
    - counterfactual_base: Abstract base class for counterfactual generators
    - data_loader: Adult Income dataset loader
    - acs_data_loader: ACS dataset loader using Folktables
    - classifiers: Baseline classifiers (Logistic Regression, Random Forest)
    - wachter_method: Wachter et al. (2017) counterfactual method
    - glance_method: GLANCE global counterfactual method
    - facts_fairness: FACTS fairness evaluation framework
    - comparison_analysis: Comprehensive comparison tools
"""

# Import config first as other modules depend on it
from src.config import Config, DEFAULT_CONFIG

# Import abstract base class
from src.counterfactual_base import CounterfactualGenerator

# Import data loaders
from src.data_loader import AdultDataLoader

# Import classifiers
from src.classifiers import BaselineClassifier, train_and_evaluate_models

# Import counterfactual methods
from src.wachter_method import WachterCounterfactual, evaluate_counterfactuals
from src.glance_method import GLANCECounterfactual, evaluate_glance

# Import fairness evaluation
from src.facts_fairness import FACTSEvaluator, group_results_by_attribute

# Import comparison analysis
from src.comparison_analysis import (
    load_results,
    statistical_comparison,
    fairness_comparison,
    facts_evaluation,
    create_visualizations,
    generate_comparison_report,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "Config",
    "DEFAULT_CONFIG",
    # Base class
    "CounterfactualGenerator",
    # Data loaders
    "AdultDataLoader",
    # Classifiers
    "BaselineClassifier",
    "train_and_evaluate_models",
    # Counterfactual methods
    "WachterCounterfactual",
    "evaluate_counterfactuals",
    "GLANCECounterfactual",
    "evaluate_glance",
    # Fairness
    "FACTSEvaluator",
    "group_results_by_attribute",
    # Comparison
    "load_results",
    "statistical_comparison",
    "fairness_comparison",
    "facts_evaluation",
    "create_visualizations",
    "generate_comparison_report",
]
