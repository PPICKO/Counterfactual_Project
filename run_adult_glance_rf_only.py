"""
Re-run ONLY the GLANCE step for the Adult Random Forest experiment.

The original run_adult_experiment_rf.py finished the Wachter step but the
GLANCE rule-discovery (which internally runs Wachter on a smaller sample)
produced zero valid counterfactuals — the finite-difference gradient is
flat almost everywhere on a tree-based decision surface, so Wachter
cannot move instances across the boundary on Adult+RF. As a result
GLANCE raised "No global rules discovered" before saving anything.

This script:
1. Loads the saved RF classifier from results_adult_rf/classifier.pkl
2. Reloads the Adult dataset
3. Reconstructs the exact 1000-instance balanced negative-sample
4. Runs GLANCE and writes results_adult_rf/glance/results.pkl
5. If rule discovery still yields zero rules, writes a stub pkl with
   empty results and a clear failure flag so downstream comparison
   reporting can still produce a row.

This avoids re-running the 34-minute Wachter step and does NOT touch
any file under src/ or modify the LR results.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.config import Config
from src.data_loader import AdultDataLoader
from src.glance_method import GLANCECounterfactual, evaluate_glance


def _balanced_negative_sample(
    neg_indices: np.ndarray,
    protected_values: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = np.unique(protected_values[neg_indices])
    per_group = n_total // len(groups)
    picked: list[int] = []
    for g in groups:
        g_pool = neg_indices[protected_values[neg_indices] == g]
        take = min(per_group, len(g_pool))
        picked.extend(rng.choice(g_pool, size=take, replace=False).tolist())
    return np.array(picked, dtype=int)


def main() -> None:
    config = Config()
    results_dir = Path("results_adult_rf")
    t_total = time.time()

    print("=" * 80)
    print("ADULT RF: GLANCE-only rerun (Wachter already saved)")
    print("=" * 80)

    # Load classifier
    with open(results_dir / "classifier.pkl", "rb") as f:
        model = pickle.load(f)
    print(f"Loaded classifier: {type(model).__name__}")

    # Reload data and re-derive the same 1000-instance sample
    loader = AdultDataLoader(data_dir="data", config=config)
    dataset = loader.load_processed_data()

    y_test_pred = model.predict(dataset["X_test"])
    negative_mask = y_test_pred == 0
    negative_indices = np.where(negative_mask)[0]

    protected_sex_test = dataset["protected_test"]["sex"]
    rng = np.random.default_rng(config.random_seed)
    sample_indices = _balanced_negative_sample(
        negative_indices,
        protected_sex_test,
        n_total=config.num_test_samples,
        rng=rng,
    )
    X_sample = dataset["X_test"][sample_indices]
    protected_sample_sex = protected_sex_test[sample_indices]
    print(f"Reconstructed sample: {len(X_sample)} instances")

    # GLANCE
    glance = GLANCECounterfactual(
        classifier=model,
        feature_names=dataset["feature_names"],
        continuous_features=dataset["continuous_features"],
        categorical_features=dataset["categorical_features"],
        n_rules=config.glance_n_rules,
        lambda_param=config.wachter_lambda,
        config=config,
    )

    X_negative = dataset["X_test"][negative_mask]
    print(
        f"\nDiscovering global rules from "
        f"{min(config.glance_max_samples, len(X_negative))} samples..."
    )
    t0 = time.time()
    try:
        global_rules = glance.discover_global_rules(
            X_negative,
            target_class=1,
            max_samples=config.glance_max_samples,
            verbose=True,
        )
    except Exception as e:
        print(f"Rule discovery raised: {e}")
        global_rules = []

    glance.print_global_rules()
    rule_discovery_time = time.time() - t0
    print(f"\nRule discovery took {rule_discovery_time:.1f}s; discovered {len(global_rules)} rules")

    # Phase 2: apply rules if any were found
    if len(global_rules) > 0:
        print(f"\nApplying global rules to {len(X_sample)} test samples...")
        t0 = time.time()
        glance_results = glance.generate_with_rules(X_sample, target_class=1, verbose=True)
        apply_time = time.time() - t0
        print(f"Rule application took {apply_time:.1f}s")

        glance_metrics = evaluate_glance(
            glance_results, X_sample, protected_attr=protected_sample_sex
        )
        rules_payload = global_rules
        failure = False
    else:
        # No rules — produce empty stub so downstream comparison can run.
        print("\nNo rules to apply; writing empty GLANCE results stub.")
        apply_time = 0.0
        glance_results = []
        glance_metrics = {
            "validity_rate": 0.0,
            "avg_distance": float("nan"),
            "avg_sparsity": float("nan"),
            "avg_num_changes": float("nan"),
            "convergence_rate": 0.0,
            "n_samples": len(X_sample),
            "note": (
                "GLANCE rule discovery returned 0 rules under Random Forest. "
                "Finite-difference gradients are flat on the tree-ensemble "
                "decision surface, so the Wachter Phase-1 step inside GLANCE "
                "found no valid seed counterfactuals to cluster into rules."
            ),
        }
        rules_payload = []
        failure = True

    total = time.time() - t_total
    with open(results_dir / "glance" / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": glance_results,
                "metrics": glance_metrics,
                "rules": rules_payload,
                "X_orig": X_sample,
                "protected": protected_sample_sex,
                "wall_clock_seconds": total,
                "rule_discovery_failed": failure,
            },
            f,
        )

    print(f"\nTotal wall-clock: {total:.1f}s ({total/60:.2f} min)")
    print(f"Saved: {results_dir / 'glance' / 'results.pkl'} (failure={failure})")


if __name__ == "__main__":
    main()
