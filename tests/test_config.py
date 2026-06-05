"""Tests for configuration module."""

import pytest
from pathlib import Path

from src.config import Config


class TestConfig:
    """Test suite for Config class."""

    def test_default_config_creation(self):
        """Test that default config can be created."""
        config = Config()
        assert config is not None

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()

        # Check Wachter defaults
        assert config.wachter_lambda == 0.1
        assert config.wachter_lr == 0.01
        assert config.wachter_max_iter == 1000

        # Check GLANCE defaults
        assert config.glance_n_rules == 5
        assert config.glance_max_samples == 200

        # Check random seed
        assert config.random_seed == 42

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = Config(
            wachter_lambda=0.5,
            wachter_max_iter=500,
            random_seed=123
        )

        assert config.wachter_lambda == 0.5
        assert config.wachter_max_iter == 500
        assert config.random_seed == 123

    def test_data_dir_path(self):
        """Test data directory path handling."""
        config = Config(data_dir="custom_data")
        assert "custom_data" in str(config.data_dir)

    def test_results_dir_path(self):
        """Test results directory path handling."""
        config = Config(results_dir="custom_results")
        assert "custom_results" in str(config.results_dir)
