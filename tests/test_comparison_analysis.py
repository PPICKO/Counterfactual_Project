"""Tests for comparison analysis module."""

import numpy as np
import pytest

from src.comparison_analysis import statistical_comparison, fairness_comparison


@pytest.fixture
def wachter_results():
    """Create sample Wachter results."""
    results = []
    np.random.seed(42)
    for i in range(100):
        results.append({
            "validity": np.random.random() > 0.5,
            "distance": np.random.uniform(0.1, 0.8),
            "sparsity": np.random.uniform(0, 0.2),
            "x_cf": np.random.randn(10),
        })
    return results


@pytest.fixture
def glance_results():
    """Create sample GLANCE results."""
    results = []
    np.random.seed(43)
    for i in range(100):
        results.append({
            "validity": np.random.random() > 0.52,
            "distance": np.random.uniform(0.2, 0.9),
            "sparsity": 0.0,  # GLANCE typically has 0 sparsity
            "x_cf": np.random.randn(10),
            "rule_id": np.random.randint(0, 5),
        })
    return results


@pytest.fixture
def wachter_data(wachter_results):
    """Create Wachter data dict."""
    protected = np.array([0] * 50 + [1] * 50)
    return {
        "results": wachter_results,
        "protected": protected,
        "metrics": {
            "validity_rate": np.mean([r["validity"] for r in wachter_results]),
            "avg_distance": np.mean([r["distance"] for r in wachter_results]),
            "avg_sparsity": np.mean([r["sparsity"] for r in wachter_results]),
            "avg_num_changes": 10.0,
            "convergence_rate": 1.0,
        }
    }


@pytest.fixture
def glance_data(glance_results):
    """Create GLANCE data dict."""
    protected = np.array([0] * 50 + [1] * 50)
    return {
        "results": glance_results,
        "protected": protected,
        "metrics": {
            "validity_rate": np.mean([r["validity"] for r in glance_results]),
            "avg_distance": np.mean([r["distance"] for r in glance_results]),
            "avg_sparsity": np.mean([r["sparsity"] for r in glance_results]),
            "avg_num_changes": 10.0,
            "convergence_rate": 1.0,
        }
    }


class TestStatisticalComparison:
    """Test suite for statistical_comparison function."""

    def test_basic_comparison(self, wachter_results, glance_results):
        """Test basic statistical comparison."""
        result = statistical_comparison(wachter_results, glance_results)

        assert "distance" in result
        assert "sparsity" in result
        assert "validity" in result

    def test_distance_comparison(self, wachter_results, glance_results):
        """Test distance comparison output."""
        result = statistical_comparison(wachter_results, glance_results)

        dist = result["distance"]
        assert "wachter_mean" in dist
        assert "glance_mean" in dist
        assert "t_statistic" in dist
        assert "p_value" in dist
        assert "significant" in dist

        assert isinstance(dist["significant"], bool)
        assert 0 <= dist["p_value"] <= 1

    def test_validity_comparison(self, wachter_results, glance_results):
        """Test validity comparison output."""
        result = statistical_comparison(wachter_results, glance_results)

        val = result["validity"]
        assert "wachter_rate" in val
        assert "glance_rate" in val
        assert "chi2_statistic" in val
        assert "p_value" in val
        assert "significant" in val

        assert 0 <= val["wachter_rate"] <= 1
        assert 0 <= val["glance_rate"] <= 1

    def test_empty_results_raises_error(self):
        """Test that empty results raise ValueError."""
        with pytest.raises(ValueError):
            statistical_comparison([], [])

    def test_mismatched_lengths(self, wachter_results):
        """Test with different result lengths."""
        # Should still work with different lengths
        short_glance = wachter_results[:50]
        result = statistical_comparison(wachter_results, short_glance)

        assert result is not None


class TestFairnessComparison:
    """Test suite for fairness_comparison function."""

    def test_basic_fairness(self, wachter_data, glance_data):
        """Test basic fairness comparison."""
        result = fairness_comparison(wachter_data, glance_data)

        assert "summary" in result
        assert "wachter_dp_diff" in result["summary"]
        assert "glance_dp_diff" in result["summary"]

    def test_per_group_metrics(self, wachter_data, glance_data):
        """Test per-group metric computation."""
        result = fairness_comparison(wachter_data, glance_data)

        # Should have metrics for each group (0 and 1)
        assert "0" in result or 0 in result
        assert "1" in result or 1 in result

    def test_statistical_tests(self, wachter_data, glance_data):
        """Test that statistical tests are computed."""
        result = fairness_comparison(wachter_data, glance_data)

        summary = result["summary"]
        if "wachter_statistical_tests" in summary:
            tests = summary["wachter_statistical_tests"]
            assert "p_value" in tests
            assert "significant" in tests

    def test_cost_disparity(self, wachter_data, glance_data):
        """Test cost disparity calculation."""
        result = fairness_comparison(wachter_data, glance_data)

        summary = result["summary"]
        assert "wachter_cost_disparity" in summary
        assert "glance_cost_disparity" in summary

        assert summary["wachter_cost_disparity"] >= 0
        assert summary["glance_cost_disparity"] >= 0

    def test_demographic_parity_bounds(self, wachter_data, glance_data):
        """Test that DP differences are in valid range."""
        result = fairness_comparison(wachter_data, glance_data)

        summary = result["summary"]
        assert 0 <= summary["wachter_dp_diff"] <= 1
        assert 0 <= summary["glance_dp_diff"] <= 1
