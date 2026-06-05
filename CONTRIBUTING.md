# Contributing to Counterfactual Fairness Comparison

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/counterfactual-fairness-comparison/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Python version and OS
   - Relevant error messages or logs

### Suggesting Enhancements

1. Check existing issues for similar suggestions
2. Create a new issue with the "enhancement" label
3. Describe the enhancement and its benefits
4. Include any relevant examples or references

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes following the coding standards below
4. Add tests for new functionality
5. Run the test suite:
   ```bash
   pytest tests/
   ```
6. Commit your changes:
   ```bash
   git commit -m "Add: description of your changes"
   ```
7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Create a Pull Request

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 88 characters (Black default)

### Type Hints

- Use type hints for all function parameters and return values
- Use `typing` module for complex types:
  ```python
  from typing import Any, Dict, List, Optional, Tuple

  def process_data(
      data: np.ndarray,
      options: Optional[Dict[str, Any]] = None
  ) -> Tuple[np.ndarray, List[str]]:
      ...
  ```

### Docstrings

- Use Google-style docstrings:
  ```python
  def function_name(param1: int, param2: str) -> bool:
      """Short description of function.

      Longer description if needed.

      Args:
          param1: Description of param1.
          param2: Description of param2.

      Returns:
          Description of return value.

      Raises:
          ValueError: When param1 is negative.
      """
  ```

### Testing

- Write unit tests for all new functionality
- Use pytest for testing
- Aim for >80% code coverage
- Test edge cases and error conditions

### Commit Messages

Use conventional commit format:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for changes to existing features
- `Remove:` for removed features
- `Docs:` for documentation changes
- `Test:` for test changes
- `Refactor:` for code refactoring

Example:
```
Add: FACTS Equal Cost of Effectiveness metric

- Implement cost calculation for target success rates
- Add statistical tests for cost comparisons
- Update documentation with new metric
```

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/counterfactual-fairness-comparison.git
   cd counterfactual-fairness-comparison
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_wachter_method.py

# Run tests matching pattern
pytest -k "test_validity"
```

## Code Review Process

1. All PRs require at least one approval
2. CI checks must pass (tests, linting)
3. Code coverage should not decrease
4. Documentation must be updated if needed

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing!
