"""
Run the full Wachter-vs-GLANCE comparison analysis on ACS RF results.

Reads from results_acs_rf/ and writes the comparison report/plots there.
Logic is identical to run_acs_comparison.py except for the results_dir.
"""

from __future__ import annotations

from run_acs_comparison import run_acs_comparison


def main() -> None:
    run_acs_comparison(results_dir="results_acs_rf")


if __name__ == "__main__":
    main()
