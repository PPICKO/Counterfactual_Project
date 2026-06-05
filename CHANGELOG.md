# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-01

### Added
- Initial release of counterfactual fairness comparison framework
- Wachter method implementation with vectorized gradient computation
- GLANCE method implementation for global counterfactual rules
- FACTS fairness evaluation framework with four metrics:
  - Equal Burden (cost similarity)
  - Equal Effectiveness (success rate parity)
  - Equal Choice (option availability)
  - Equal Cost of Effectiveness
- ACS dataset loader using Folktables
- Adult dataset loader with automatic download
- Comprehensive comparison analysis utilities
- Statistical tests (t-tests, chi-square, Fisher's exact)
- Visualization generation (histograms, boxplots, bar charts)
- Configuration management with dataclasses
- Abstract base class for counterfactual generators

### Experimental Results
- Adult Dataset (n=100):
  - Wachter: 49% validity, 0.358 distance, 38.29% fairness gap (p<0.001)
  - GLANCE: 48% validity, 0.494 distance, 36.65% fairness gap (p<0.001)
- ACS Dataset (n=100):
  - Wachter: 13% validity, 0.144 distance, 6% fairness gap (NS)
  - GLANCE: 7% validity, 0.222 distance, 6% fairness gap (NS)

### Key Findings
- Dataset selection matters more than method choice for fairness outcomes
- Validity drops 73-85% from Adult to ACS despite better classifiers
- Fairness gaps shrink 84% on modern ACS data (6% vs 38%)
- Wachter consistently produces closer counterfactuals (p<0.001)

## [Unreleased]

### Planned
- Support for additional counterfactual methods (DiCE, FACE)
- Nationwide ACS analysis to isolate geographic effects
- Deep learning classifier support
- Fairness-aware optimization constraints
- Interactive visualization dashboard
