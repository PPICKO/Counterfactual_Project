# Makefile for Counterfactual Fairness Comparison Project

.PHONY: help install install-dev test lint format clean run-experiment run-comparison docs

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install package dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with black and isort"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make run-experiment - Run ACS experiment"
	@echo "  make run-comparison - Run comparison analysis"
	@echo "  make all            - Run experiment and comparison"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Linting and Formatting
lint:
	flake8 src/ --max-line-length=88 --extend-ignore=E203
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

check-format:
	black --check src/ tests/
	isort --check-only src/ tests/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-results:
	rm -rf results_acs/wachter/*.pkl
	rm -rf results_acs/glance/*.pkl
	rm -rf results_acs/*.pkl

# Running experiments
run-experiment:
	python run_acs_experiment.py

run-comparison:
	python run_acs_comparison.py

all: run-experiment run-comparison

# Documentation
docs:
	@echo "Documentation not yet configured"

# Development workflow
dev-setup: install-dev
	@echo "Development environment ready!"

pre-commit:
	pre-commit run --all-files

# CI simulation
ci: check-format lint test
	@echo "CI checks passed!"
