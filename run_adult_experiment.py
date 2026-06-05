"""
Run complete counterfactual experiment on the UCI Adult dataset.

Mirrors run_acs_experiment.py but loads the locally-provided Adult files
(data/adult.data and data/adult.test). Reuses every downstream module:
classifier training, Wachter, GLANCE, evaluation.

Output layout (parallel to results_acs/):
    results_adult/
        dataset_info.txt
        classifier.pkl
        wachter/results.pkl
        glance/results.pkl
        comparison/   (populated by run_adult_comparison.py)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import Config
from src.data_loader import AdultDataLoader
from src.glance_method import GLANCECounterfactual, evaluate_glance
from src.wachter_method import WachterCounterfactual, evaluate_counterfactuals


# Adult's `sex` column is label-encoded by AdultDataLoader. To stay
# robust against alphabetical encoding order ("Female"=0, "Male"=1 in
# practice), we keep the protected attribute as the *raw* string values
# returned by the loader. The downstream FACTS / fairness code treats
# the values opaquely (groups by unique value), so any hashable label
# works. We balance the sample 50/50 by these raw labels.


def _balanced_negative_sample(
    neg_indices: np.ndarray,
    protected_values: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick up to n_total indices from `neg_indices`, balanced across the
    unique values in `protected_values[neg_indices]` (50/50 when binary).

    Falls back to whatever is available per group; total may be < n_total
    if one group is small.
    """
    groups = np.unique(protected_values[neg_indices])
    per_group = n_total // len(groups)

    picked: list[int] = []
    for g in groups:
        g_pool = neg_indices[protected_values[neg_indices] == g]
        take = min(per_group, len(g_pool))
        picked.extend(rng.choice(g_pool, size=take, replace=False).tolist())

    return np.array(picked, dtype=int)


def main() -> Dict[str, Any]:
    """Run the complete Adult counterfactual experiment."""
    config = Config()

    print("=" * 80)
    print("COUNTERFACTUAL EXPERIMENT ON ADULT DATASET")
    print(f"  num_test_samples = {config.num_test_samples}")
    print("=" * 80)

    # Output directories
    results_dir = Path("results_adult")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "wachter").mkdir(exist_ok=True)
    (results_dir / "glance").mkdir(exist_ok=True)
    (results_dir / "comparison").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: load Adult (local files at data/adult.data, data/adult.test)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: Loading Adult Dataset (local files)")
    print("=" * 80)

    # AdultDataLoader.download_data() short-circuits when the files
    # already exist, so pointing at data/ uses the user-provided copies.
    loader = AdultDataLoader(data_dir="data", config=config)
    dataset = loader.load_processed_data()

    # Save dataset info
    with open(results_dir / "dataset_info.txt", "w") as f:
        f.write("Adult (UCI) Dataset Information\n")
        f.write("=" * 80 + "\n")
        f.write(f"Train samples: {len(dataset['X_train'])}\n")
        f.write(f"Val samples:   {len(dataset['X_val'])}\n")
        f.write(f"Test samples:  {len(dataset['X_test'])}\n")
        f.write(f"Features:      {dataset['X_train'].shape[1]}\n")
        f.write(f"Feature names: {dataset['feature_names']}\n")
        f.write("\nProtected attributes: sex, race\n")
        f.write("Target: income (>50K = 1, <=50K = 0)\n")

    # ------------------------------------------------------------------
    # Step 2: train classifier (same LR setup as ACS run for parity)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: Training Classifier (LogisticRegression)")
    print("=" * 80)

    model = LogisticRegression(
        max_iter=config.classifier_max_iter,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    model.fit(dataset["X_train"], dataset["y_train"])

    y_test_pred = model.predict(dataset["X_test"])
    y_test_proba = model.predict_proba(dataset["X_test"])[:, 1]
    print(f"\nTest Accuracy: {accuracy_score(dataset['y_test'], y_test_pred):.4f}")
    print(f"Test AUC:      {roc_auc_score(dataset['y_test'], y_test_proba):.4f}")

    with open(results_dir / "classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    # ------------------------------------------------------------------
    # Step 3: select balanced negative-prediction sample
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: Selecting Test Samples (balanced by sex)")
    print("=" * 80)

    negative_mask = y_test_pred == 0
    negative_indices = np.where(negative_mask)[0]
    print(f"Negative-prediction pool size: {len(negative_indices)}")

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

    print(f"Selected {len(X_sample)} samples (target {config.num_test_samples})")
    uniq, cnts = np.unique(protected_sample_sex, return_counts=True)
    print(f"  sex distribution: {dict(zip(uniq.tolist(), cnts.tolist()))}")

    # ------------------------------------------------------------------
    # Step 4: Wachter counterfactuals
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: Generating Wachter Counterfactuals")
    print("=" * 80)

    wachter = WachterCounterfactual(
        classifier=model,
        feature_names=dataset["feature_names"],
        continuous_features=dataset["continuous_features"],
        categorical_features=dataset["categorical_features"],
        lambda_param=config.wachter_lambda,
        lr=config.wachter_lr,
        max_iter=config.wachter_max_iter,
        config=config,
    )

    wachter_results = wachter.generate_batch(
        X_sample,
        target_class=1,
        verbose=True,
    )

    wachter_metrics = evaluate_counterfactuals(
        wachter_results,
        X_sample,
        protected_attr=protected_sample_sex,
    )

    with open(results_dir / "wachter" / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": wachter_results,
                "metrics": wachter_metrics,
                "X_orig": X_sample,
                "protected": protected_sample_sex,
            },
            f,
        )

    # ------------------------------------------------------------------
    # Step 5: GLANCE counterfactuals
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: Generating GLANCE Counterfactuals")
    print("=" * 80)

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
    global_rules = glance.discover_global_rules(
        X_negative,
        target_class=1,
        max_samples=config.glance_max_samples,
        verbose=True,
    )
    glance.print_global_rules()

    print(f"\nApplying global rules to {len(X_sample)} test samples...")
    glance_results = glance.generate_with_rules(
        X_sample,
        target_class=1,
        verbose=True,
    )
    glance_metrics = evaluate_glance(
        glance_results,
        X_sample,
        protected_attr=protected_sample_sex,
    )

    with open(results_dir / "glance" / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": glance_results,
                "metrics": glance_metrics,
                "rules": global_rules,
                "X_orig": X_sample,
                "protected": protected_sample_sex,
            },
            f,
        )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Classifier:      {results_dir / 'classifier.pkl'}")
    print(f"  - Wachter results: {results_dir / 'wachter/results.pkl'}")
    print(f"  - GLANCE results:  {results_dir / 'glance/results.pkl'}")
    print("\nTo run comparison analysis:")
    print("  python run_adult_comparison.py")

    return {
        "dataset": dataset,
        "model": model,
        "wachter_results": wachter_results,
        "glance_results": glance_results,
    }


if __name__ == "__main__":
    main()
