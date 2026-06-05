"""
FACTS: Fairness-Aware Counterfactuals for Subgroups
Implementation of fairness evaluation framework from Kavouras et al. (2023)

Reference:
Kavouras, L., Sacharidis, D., Arvanitidis, G., & Palpanas, T. (2023).
FACTS: Fairness-Aware Counterfactuals for Subgroups.
arXiv:2306.14978
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.config import Config, DEFAULT_CONFIG


class FACTSEvaluator:
    """
    FACTS framework for evaluating fairness of recourse across subgroups.

    Evaluates counterfactual methods on 5 fairness criteria:
    1. Equal Burden: Average cost should be similar across groups
    2. Equal Effectiveness: Success rates should be comparable across groups
    3. Equal Effectiveness within Budget: Success at fixed cost should be equal
    4. Equal Choice: Number of available options should not differ
    5. Equal Cost of Effectiveness: Cost to reach fixed success rate should be similar
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the FACTS evaluator.

        Args:
            alpha: Significance level for statistical tests (default: from config).
            config: Optional Config object for settings.
        """
        self.config = config or DEFAULT_CONFIG
        self.alpha = alpha or self.config.alpha

    def evaluate_equal_burden(
        self,
        results_by_group: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Equal Burden: Average cost of recourse should be similar across groups.

        Args:
            results_by_group: Dict mapping group_id -> list of result dicts.

        Returns:
            Dictionary with burden analysis including:
                - avg_cost_by_group: Average distance per group.
                - burden_disparity: Max - min average cost.
                - statistical_test: t-test results comparing groups.
        """
        avg_costs: Dict[str, float] = {}
        valid_costs_by_group: Dict[str, List[float]] = {}

        for group, results in results_by_group.items():
            # Only consider valid counterfactuals
            valid_results = [r for r in results if r["validity"]]
            if valid_results:
                costs = [r["distance"] for r in valid_results]
                avg_costs[group] = float(np.mean(costs))
                valid_costs_by_group[group] = costs
            else:
                avg_costs[group] = np.nan
                valid_costs_by_group[group] = []

        # Compute burden disparity
        valid_avg_costs = [c for c in avg_costs.values() if not np.isnan(c)]
        if len(valid_avg_costs) >= 2:
            burden_disparity = max(valid_avg_costs) - min(valid_avg_costs)
        else:
            burden_disparity = 0.0

        # Statistical test (t-test for 2 groups, ANOVA for more)
        statistical_test: Optional[Dict[str, Any]] = None
        if len(valid_costs_by_group) == 2:
            groups_list = list(valid_costs_by_group.keys())
            costs1 = valid_costs_by_group[groups_list[0]]
            costs2 = valid_costs_by_group[groups_list[1]]

            if len(costs1) > 0 and len(costs2) > 0:
                t_stat, p_value = stats.ttest_ind(costs1, costs2)
                statistical_test = {
                    "test": "t-test",
                    "statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.alpha,
                }

        return {
            "avg_cost_by_group": avg_costs,
            "burden_disparity": burden_disparity,
            "statistical_test": statistical_test,
        }

    def evaluate_equal_effectiveness(
        self,
        results_by_group: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Equal Effectiveness: Success rates should be comparable across groups.

        Args:
            results_by_group: Dict mapping group_id -> list of result dicts.

        Returns:
            Dictionary with effectiveness analysis including:
                - success_rate_by_group: Validity rate per group.
                - effectiveness_gap: Demographic parity difference.
                - statistical_test: Chi-square or Fisher's exact test.
        """
        success_rates: Dict[str, float] = {}
        contingency_data: Dict[str, Dict[str, int]] = {}

        for group, results in results_by_group.items():
            n_total = len(results)
            n_success = sum([r["validity"] for r in results])
            success_rates[group] = n_success / n_total if n_total > 0 else 0.0
            contingency_data[group] = {
                "success": n_success,
                "failure": n_total - n_success,
            }

        # Compute effectiveness gap (demographic parity)
        rates = list(success_rates.values())
        effectiveness_gap = max(rates) - min(rates) if len(rates) >= 2 else 0.0

        # Statistical test
        statistical_test: Optional[Dict[str, Any]] = None
        if len(contingency_data) == 2:
            groups_list = list(contingency_data.keys())
            g1 = contingency_data[groups_list[0]]
            g2 = contingency_data[groups_list[1]]

            contingency_table = np.array([
                [g1["success"], g1["failure"]],
                [g2["success"], g2["failure"]],
            ])

            # Use Fisher's exact if any cell < 5, otherwise chi-square
            if np.min(contingency_table) < 5:
                odds_ratio, p_value = stats.fisher_exact(contingency_table)
                statistical_test = {
                    "test": "fisher_exact",
                    "statistic": float(odds_ratio),
                    "p_value": float(p_value),
                    "significant": p_value < self.alpha,
                }
            else:
                chi2, p_value, dof, expected = stats.chi2_contingency(
                    contingency_table
                )
                statistical_test = {
                    "test": "chi_square",
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                    "significant": p_value < self.alpha,
                }

            # Confidence interval for the gap
            p1 = success_rates[groups_list[0]]
            p2 = success_rates[groups_list[1]]
            n1 = g1["success"] + g1["failure"]
            n2 = g2["success"] + g2["failure"]

            # Guard against division by zero
            if n1 > 0 and n2 > 0:
                # Standard error calculation with safeguards
                var1 = p1 * (1 - p1) / n1 if n1 > 0 else 0.0
                var2 = p2 * (1 - p2) / n2 if n2 > 0 else 0.0
                se = np.sqrt(var1 + var2)
                gap = abs(p1 - p2)
                ci_lower = gap - 1.96 * se
                ci_upper = gap + 1.96 * se
                statistical_test["gap_ci_95"] = (
                    max(0, ci_lower),
                    min(1, ci_upper),
                )
            else:
                statistical_test["gap_ci_95"] = (0.0, 1.0)

        return {
            "success_rate_by_group": success_rates,
            "effectiveness_gap": effectiveness_gap,
            "statistical_test": statistical_test,
        }

    def evaluate_equal_choice(
        self,
        results_by_group: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Equal Choice: Number of available counterfactual options should not differ.

        For global methods, this measures how many distinct rules apply.
        For instance-specific methods, this measures valid CF availability.

        Args:
            results_by_group: Dict mapping group_id -> list of result dicts.

        Returns:
            Dictionary with choice analysis including:
                - choice_availability_by_group: Proportion with valid CFs.
                - choice_disparity: Max - min availability.
                - rule_diversity_by_group: Unique rules per group (if applicable).
        """
        choice_availability: Dict[str, float] = {}

        for group, results in results_by_group.items():
            # Handle empty results list
            if not results:
                choice_availability[group] = 0.0
                continue

            # Count how many instances have at least one valid counterfactual
            has_choice = sum([r["validity"] for r in results])
            total = len(results)
            choice_availability[group] = has_choice / total if total > 0 else 0.0

        # For GLANCE specifically, also count rule diversity
        rule_diversity: Dict[str, int] = {}
        for group, results in results_by_group.items():
            # Check for empty results
            if results and "rule_id" in results[0]:
                unique_rules = set([r["rule_id"] for r in results if r["validity"]])
                rule_diversity[group] = len(unique_rules)

        # Choice disparity
        rates = list(choice_availability.values())
        choice_disparity = max(rates) - min(rates) if len(rates) >= 2 else 0.0

        output: Dict[str, Any] = {
            "choice_availability_by_group": choice_availability,
            "choice_disparity": choice_disparity,
        }

        if rule_diversity:
            output["rule_diversity_by_group"] = rule_diversity

        return output

    def evaluate_equal_cost_of_effectiveness(
        self,
        results_by_group: Dict[str, List[Dict[str, Any]]],
        target_effectiveness: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Equal Cost of Effectiveness: Cost required to achieve a target success rate
        should be similar across groups.

        Args:
            results_by_group: Dict mapping group_id -> list of result dicts.
            target_effectiveness: Target success rate (default: from config).

        Returns:
            Dictionary with cost-effectiveness analysis including:
                - cost_at_target_effectiveness: Cost at target per group.
                - target_effectiveness: The target effectiveness used.
                - cost_disparity: Max - min cost.
        """
        target_effectiveness = target_effectiveness or self.config.target_effectiveness
        cost_at_target: Dict[str, float] = {}

        for group, results in results_by_group.items():
            valid_results = [r for r in results if r["validity"]]

            if len(valid_results) == 0:
                cost_at_target[group] = np.nan
                continue

            # Sort by cost
            sorted_results = sorted(valid_results, key=lambda x: x["distance"])

            # Find cost at which we achieve target effectiveness
            n_needed = int(target_effectiveness * len(results))

            if n_needed > 0 and len(sorted_results) >= n_needed:
                cost_at_target[group] = float(sorted_results[n_needed - 1]["distance"])
            else:
                # Can't achieve target effectiveness
                cost_at_target[group] = np.nan

        # Cost disparity
        valid_costs = [c for c in cost_at_target.values() if not np.isnan(c)]
        if len(valid_costs) >= 2:
            cost_disparity = max(valid_costs) - min(valid_costs)
        else:
            cost_disparity = 0.0

        return {
            "cost_at_target_effectiveness": cost_at_target,
            "target_effectiveness": target_effectiveness,
            "cost_disparity": cost_disparity,
        }

    def evaluate_all(
        self,
        results_by_group: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Run all FACTS fairness evaluations.

        Args:
            results_by_group: Dict mapping group_id -> list of result dicts.

        Returns:
            Comprehensive FACTS fairness report with all evaluations.
        """
        return {
            "equal_burden": self.evaluate_equal_burden(results_by_group),
            "equal_effectiveness": self.evaluate_equal_effectiveness(results_by_group),
            "equal_choice": self.evaluate_equal_choice(results_by_group),
            "equal_cost_of_effectiveness": self.evaluate_equal_cost_of_effectiveness(
                results_by_group
            ),
        }

    def print_report(
        self,
        facts_results: Dict[str, Any],
        method_name: str = "Method",
    ) -> None:
        """
        Print a human-readable FACTS fairness report.

        Args:
            facts_results: Output from evaluate_all().
            method_name: Name of the method being evaluated.
        """
        print("=" * 80)
        print(f"FACTS FAIRNESS REPORT: {method_name}")
        print("=" * 80)

        # Equal Burden
        print("\n1. EQUAL BURDEN (Average Cost Similarity)")
        print("-" * 80)
        burden = facts_results["equal_burden"]
        for group, cost in burden["avg_cost_by_group"].items():
            if not np.isnan(cost):
                print(f"  {group}: {cost:.4f}")
            else:
                print(f"  {group}: N/A (no valid CFs)")
        print(f"  Burden Disparity: {burden['burden_disparity']:.4f}")
        if burden["statistical_test"]:
            test = burden["statistical_test"]
            print(f"  Statistical Test: {test['test']}")
            print(f"    p-value: {test['p_value']:.6f}")
            print(f"    Significant: {'YES' if test['significant'] else 'NO'}")

        # Equal Effectiveness
        print("\n2. EQUAL EFFECTIVENESS (Success Rate Parity)")
        print("-" * 80)
        effectiveness = facts_results["equal_effectiveness"]
        for group, rate in effectiveness["success_rate_by_group"].items():
            print(f"  {group}: {rate:.2%}")
        print(f"  Effectiveness Gap: {effectiveness['effectiveness_gap']:.2%}")
        if effectiveness["statistical_test"]:
            test = effectiveness["statistical_test"]
            print(f"  Statistical Test: {test['test']}")
            print(f"    p-value: {test['p_value']:.6f}")
            print(f"    Significant: {'YES' if test['significant'] else 'NO'}")
            if "gap_ci_95" in test:
                ci = test["gap_ci_95"]
                print(f"    95% CI: [{ci[0]:.2%}, {ci[1]:.2%}]")

        # Equal Choice
        print("\n3. EQUAL CHOICE (Option Availability)")
        print("-" * 80)
        choice = facts_results["equal_choice"]
        for group, rate in choice["choice_availability_by_group"].items():
            print(f"  {group}: {rate:.2%}")
        print(f"  Choice Disparity: {choice['choice_disparity']:.2%}")
        if "rule_diversity_by_group" in choice:
            print("  Rule Diversity:")
            for group, n_rules in choice["rule_diversity_by_group"].items():
                print(f"    {group}: {n_rules} unique rules")

        # Equal Cost of Effectiveness
        print("\n4. EQUAL COST OF EFFECTIVENESS")
        print("-" * 80)
        cost_eff = facts_results["equal_cost_of_effectiveness"]
        print(f"  Target Effectiveness: {cost_eff['target_effectiveness']:.0%}")
        for group, cost in cost_eff["cost_at_target_effectiveness"].items():
            if not np.isnan(cost):
                print(f"  {group}: {cost:.4f}")
            else:
                print(f"  {group}: N/A (target not achievable)")
        print(f"  Cost Disparity: {cost_eff['cost_disparity']:.4f}")

        print("\n" + "=" * 80)


def group_results_by_attribute(
    results: List[Dict[str, Any]],
    protected_attr: np.ndarray,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Helper function to group results by protected attribute.

    Args:
        results: List of counterfactual result dictionaries.
        protected_attr: Array of protected attribute values (same length as results).

    Returns:
        Dictionary mapping group_id -> list of results for that group.
    """
    # Handle empty results
    if not results or len(protected_attr) == 0:
        return {}

    unique_groups = np.unique(protected_attr)
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for group in unique_groups:
        mask = protected_attr == group
        grouped[str(group)] = [r for i, r in enumerate(results) if mask[i]]

    return grouped
