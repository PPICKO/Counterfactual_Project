"""Tests for Wachter counterfactual method."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.wachter_method import WachterCounterfactual, evaluate_counterfactuals


@pytest.fixture
def simple_classifier():
    """Create a simple trained classifier for testing."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    return clf, X, y


@pytest.fixture
def wachter_instance(simple_classifier):
    """Create a Wachter instance for testing."""
    clf, X, y = simple_classifier

    wachter = WachterCounterfactual(
        classifier=clf,
        feature_names=[f"feature_{i}" for i in range(5)],
        continuous_features=[f"feature_{i}" for i in range(5)],
        categorical_features=[],
        lambda_param=0.1,
        lr=0.01,
        max_iter=100,
    )

    return wachter, X, y


class TestWachterCounterfactual:
    """Test suite for WachterCounterfactual class."""

    def test_initialization(self, simple_classifier):
        """Test that Wachter can be initialized."""
        clf, _, _ = simple_classifier

        wachter = WachterCounterfactual(
            classifier=clf,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            continuous_features=["f1", "f2", "f3", "f4", "f5"],
            categorical_features=[],
        )

        assert wachter is not None
        assert wachter.classifier == clf

    def test_generate_single(self, wachter_instance):
        """Test generating a single counterfactual."""
        wachter, X, y = wachter_instance

        # Get a negative instance
        negative_idx = np.where(y == 0)[0][0]
        x_orig = X[negative_idx]

        result = wachter.generate(x_orig, target_class=1)

        assert "x_cf" in result
        assert "validity" in result
        assert "distance" in result
        assert "sparsity" in result
        assert "converged" in result
        assert isinstance(result["validity"], bool)
        assert result["distance"] >= 0

    def test_generate_batch(self, wachter_instance):
        """Test generating counterfactuals for multiple instances."""
        wachter, X, y = wachter_instance

        # Get negative instances
        negative_idx = np.where(y == 0)[0][:10]
        X_neg = X[negative_idx]

        results = wachter.generate_batch(X_neg, target_class=1)

        assert len(results) == 10
        for result in results:
            assert "x_cf" in result
            assert "validity" in result

    def test_counterfactual_closer_than_original(self, wachter_instance):
        """Test that counterfactuals are reasonably close to originals."""
        wachter, X, y = wachter_instance

        negative_idx = np.where(y == 0)[0][0]
        x_orig = X[negative_idx]

        result = wachter.generate(x_orig, target_class=1)

        # Distance should be finite
        assert np.isfinite(result["distance"])

        # If valid, counterfactual should exist
        if result["validity"]:
            assert result["x_cf"] is not None
            # Counterfactual should be different from original
            assert not np.allclose(result["x_cf"], x_orig)


class TestEvaluateCounterfactuals:
    """Test suite for evaluate_counterfactuals function."""

    def test_evaluate_with_results(self, wachter_instance):
        """Test evaluation of counterfactual results."""
        wachter, X, y = wachter_instance

        # Generate some results
        negative_idx = np.where(y == 0)[0][:20]
        X_neg = X[negative_idx]
        results = wachter.generate_batch(X_neg, target_class=1)

        # Evaluate
        metrics = evaluate_counterfactuals(results, X_neg)

        assert "n_samples" in metrics
        assert "validity_rate" in metrics
        assert "avg_distance" in metrics
        assert "avg_sparsity" in metrics
        assert "convergence_rate" in metrics

        assert metrics["n_samples"] == 20
        assert 0 <= metrics["validity_rate"] <= 1
        assert metrics["avg_distance"] >= 0

    def test_evaluate_with_protected_attribute(self, wachter_instance):
        """Test evaluation with protected attribute."""
        wachter, X, y = wachter_instance

        negative_idx = np.where(y == 0)[0][:20]
        X_neg = X[negative_idx]
        results = wachter.generate_batch(X_neg, target_class=1)

        # Create mock protected attribute
        protected = np.array([0, 1] * 10)

        metrics = evaluate_counterfactuals(results, X_neg, protected_attr=protected)

        assert "group_metrics" in metrics
        assert len(metrics["group_metrics"]) == 2  # Two groups
