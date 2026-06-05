"""
Run the full Wachter-vs-GLANCE comparison analysis on Adult RF results.

Reads from results_adult_rf/ and writes the comparison report/plots there.
Logic is identical to run_adult_comparison.py except for the results_dir.
"""

from __future__ import annotations

from run_adult_comparison import run_adult_comparison


def main() -> None:
    run_adult_comparison(results_dir="results_adult_rf")


if __name__ == "__main__":
    main()
