"""Tests for FACTS fairness evaluation module."""

import numpy as np
import pytest

from src.facts_fairness import FACTSEvaluator, group_results_by_attribute


@pytest.fixture
def sample_results():
    """Create sample counterfactual results for testing."""
    results = []
    for i in range(50):
        results.append({
            "validity": i < 30,  # 60% validity
            "distance": np.random.uniform(0.1, 0.5),
            "sparsity": np.random.uniform(0, 0.3),
            "x_cf": np.random.randn(10),
        })
    return results


@pytest.fixture
def grouped_results(sample_results):
    """Create grouped results by protected attribute."""
    # Split results into two groups
    protected = np.array([0] * 25 + [1] * 25)
    return group_results_by_attribute(sample_results, protected)


class TestGroupResultsByAttribute:
    """Test suite for group_results_by_attribute function."""

    def test_basic_grouping(self, sample_results):
        """Test basic grouping functionality."""
        protected = np.array([0] * 25 + [1] * 25)
        grouped = group_results_by_attribute(sample_results, protected)

        assert len(grouped) == 2
        assert 0 in grouped
        assert 1 in grouped
        assert len(grouped[0]) == 25
        assert len(grouped[1]) == 25

    def test_unequal_groups(self, sample_results):
        """Test grouping with unequal group sizes."""
        protected = np.array([0] * 10 + [1] * 40)
        grouped = group_results_by_attribute(sample_results, protected)

        assert len(grouped[0]) == 10
        assert len(grouped[1]) == 40

    def test_multiple_groups(self, sample_results):
        """Test grouping with more than two groups."""
        protected = np.array([0] * 20 + [1] * 20 + [2] * 10)
        grouped = group_results_by_attribute(sample_results, protected)

        assert len(grouped) == 3
        assert len(grouped[0]) == 20
        assert len(grouped[1]) == 20
        assert len(grouped[2]) == 10


class TestFACTSEvaluator:
    """Test suite for FACTSEvaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = FACTSEvaluator(alpha=0.05)
        assert evaluator is not None
        assert evaluator.alpha == 0.05

    def test_equal_burden(self, grouped_results):
        """Test Equal Burden calculation."""
        evaluator = FACTSEvaluator()
        result = evaluator.equal_burden(grouped_results)

        assert "avg_cost_by_group" in result
        assert "burden_disparity" in result
        assert len(result["avg_cost_by_group"]) == 2
        assert result["burden_disparity"] >= 0

    def test_equal_effectiveness(self, grouped_results):
        """Test Equal Effectiveness calculation."""
        evaluator = FACTSEvaluator()
        result = evaluator.equal_effectiveness(grouped_results)

        assert "success_rate_by_group" in result
        assert "effectiveness_gap" in result
        assert "statistical_test" in result

        for rate in result["success_rate_by_group"].values():
            assert 0 <= rate <= 1

    def test_equal_choice(self, grouped_results):
        """Test Equal Choice calculation."""
        evaluator = FACTSEvaluator()
        result = evaluator.equal_choice(grouped_results)

        assert "choice_availability_by_group" in result
        assert "choice_disparity" in result

    def test_evaluate_all(self, grouped_results):
        """Test full FACTS evaluation."""
        evaluator = FACTSEvaluator()
        result = evaluator.evaluate_all(grouped_results)

        assert "equal_burden" in result
        assert "equal_effectiveness" in result
        assert "equal_choice" in result

    def test_empty_results(self):
        """Test handling of empty results."""
        evaluator = FACTSEvaluator()
        empty_grouped = {0: [], 1: []}

        result = evaluator.evaluate_all(empty_grouped)

        # Should handle gracefully without errors
        assert result is not None

    def test_single_group(self, sample_results):
        """Test handling of single group."""
        evaluator = FACTSEvaluator()
        protected = np.zeros(50, dtype=int)
        grouped = group_results_by_attribute(sample_results, protected)

        result = evaluator.evaluate_all(grouped)

        # Single group should have zero disparity
        assert result["equal_burden"]["burden_disparity"] == 0
        assert result["equal_effectiveness"]["effectiveness_gap"] == 0


class TestStatisticalTests:
    """Test suite for statistical test functionality."""

    def test_significant_difference(self):
        """Test detection of significant difference."""
        evaluator = FACTSEvaluator(alpha=0.05)

        # Create results with large difference
        group0 = [{"validity": True, "distance": 0.1} for _ in range(40)] + \
                 [{"validity": False, "distance": 0.1} for _ in range(10)]
        group1 = [{"validity": True, "distance": 0.1} for _ in range(10)] + \
                 [{"validity": False, "distance": 0.1} for _ in range(40)]

        grouped = {0: group0, 1: group1}
        result = evaluator.equal_effectiveness(grouped)

        # Should detect significant difference (80% vs 20%)
        assert result["statistical_test"]["significant"] is True

    def test_non_significant_difference(self):
        """Test handling of non-significant difference."""
        evaluator = FACTSEvaluator(alpha=0.05)

        # Create results with small difference
        group0 = [{"validity": True, "distance": 0.1} for _ in range(25)] + \
                 [{"validity": False, "distance": 0.1} for _ in range(25)]
        group1 = [{"validity": True, "distance": 0.1} for _ in range(24)] + \
                 [{"validity": False, "distance": 0.1} for _ in range(26)]

        grouped = {0: group0, 1: group1}
        result = evaluator.equal_effectiveness(grouped)

        # Should not detect significant difference (50% vs 48%)
        assert result["statistical_test"]["significant"] is False
