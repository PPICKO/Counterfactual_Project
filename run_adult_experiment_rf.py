"""
Run complete counterfactual experiment on Adult dataset with RANDOM FOREST.

Mirrors run_adult_experiment.py exactly, with one change: the classifier is
sklearn.ensemble.RandomForestClassifier instead of LogisticRegression.
Outputs go to results_adult_rf/ to keep the original LR results intact.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import Config
from src.data_loader import AdultDataLoader
from src.glance_method import GLANCECounterfactual, evaluate_glance
from src.wachter_method import WachterCounterfactual, evaluate_counterfactuals


def _balanced_negative_sample(
    neg_indices: np.ndarray,
    protected_values: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick up to n_total indices from neg_indices, balanced across groups."""
    groups = np.unique(protected_values[neg_indices])
    per_group = n_total // len(groups)
    picked: list[int] = []
    for g in groups:
        g_pool = neg_indices[protected_values[neg_indices] == g]
        take = min(per_group, len(g_pool))
        picked.extend(rng.choice(g_pool, size=take, replace=False).tolist())
    return np.array(picked, dtype=int)


def main() -> Dict[str, Any]:
    """Run the complete Adult counterfactual experiment with Random Forest."""
    config = Config()
    t_total_start = time.time()

    print("=" * 80)
    print("COUNTERFACTUAL EXPERIMENT ON ADULT DATASET (RANDOM FOREST)")
    print(f"  num_test_samples = {config.num_test_samples}")
    print(f"  n_estimators     = {config.n_estimators}")
    print(f"  max_depth        = {config.max_depth}")
    print("=" * 80)

    results_dir = Path("results_adult_rf")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "wachter").mkdir(exist_ok=True)
    (results_dir / "glance").mkdir(exist_ok=True)
    (results_dir / "comparison").mkdir(exist_ok=True)

    # Step 1: load Adult
    print("\n" + "=" * 80)
    print("STEP 1: Loading Adult Dataset (local files)")
    print("=" * 80)

    loader = AdultDataLoader(data_dir="data", config=config)
    dataset = loader.load_processed_data()

    with open(results_dir / "dataset_info.txt", "w") as f:
        f.write("Adult (UCI) Dataset Information (Random Forest run)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Train samples: {len(dataset['X_train'])}\n")
        f.write(f"Val samples:   {len(dataset['X_val'])}\n")
        f.write(f"Test samples:  {len(dataset['X_test'])}\n")
        f.write(f"Features:      {dataset['X_train'].shape[1]}\n")
        f.write(f"Feature names: {dataset['feature_names']}\n")
        f.write("\nProtected attributes: sex, race\n")
        f.write("Target: income (>50K = 1, <=50K = 0)\n")
        f.write("\nClassifier: RandomForestClassifier\n")
        f.write(f"  n_estimators={config.n_estimators}, "
                f"max_depth={config.max_depth}, random_state={config.random_seed}\n")

    # Step 2: train Random Forest
    print("\n" + "=" * 80)
    print("STEP 2: Training Classifier (RandomForestClassifier)")
    print("=" * 80)

    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_seed,
        n_jobs=-1,
    )
    y_train = np.asarray(dataset["y_train"]).ravel()
    model.fit(dataset["X_train"], y_train)
    train_time = time.time() - t0

    y_test_pred = model.predict(dataset["X_test"])
    y_test_proba = model.predict_proba(dataset["X_test"])[:, 1]
    print(f"\nTest Accuracy: {accuracy_score(dataset['y_test'], y_test_pred):.4f}")
    print(f"Test AUC:      {roc_auc_score(dataset['y_test'], y_test_proba):.4f}")
    print(f"Train time:    {train_time:.1f}s")

    with open(results_dir / "classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    # Step 3: balanced negative sample
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

    # Step 4: Wachter
    print("\n" + "=" * 80)
    print("STEP 4: Generating Wachter Counterfactuals (RF)")
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

    t0 = time.time()
    wachter_results = wachter.generate_batch(X_sample, target_class=1, verbose=True)
    wachter_time = time.time() - t0
    print(f"\nWachter wall-clock: {wachter_time:.1f}s ({wachter_time/60:.2f} min)")

    wachter_metrics = evaluate_counterfactuals(
        wachter_results, X_sample, protected_attr=protected_sample_sex
    )

    with open(results_dir / "wachter" / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": wachter_results,
                "metrics": wachter_metrics,
                "X_orig": X_sample,
                "protected": protected_sample_sex,
                "wall_clock_seconds": wachter_time,
            },
            f,
        )

    # Step 5: GLANCE
    print("\n" + "=" * 80)
    print("STEP 5: Generating GLANCE Counterfactuals (RF)")
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
    t0 = time.time()
    global_rules = glance.discover_global_rules(
        X_negative,
        target_class=1,
        max_samples=config.glance_max_samples,
        verbose=True,
    )
    glance.print_global_rules()

    print(f"\nApplying global rules to {len(X_sample)} test samples...")
    glance_results = glance.generate_with_rules(X_sample, target_class=1, verbose=True)
    glance_time = time.time() - t0
    print(f"\nGLANCE wall-clock: {glance_time:.1f}s ({glance_time/60:.2f} min)")

    glance_metrics = evaluate_glance(
        glance_results, X_sample, protected_attr=protected_sample_sex
    )

    with open(results_dir / "glance" / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": glance_results,
                "metrics": glance_metrics,
                "rules": global_rules,
                "X_orig": X_sample,
                "protected": protected_sample_sex,
                "wall_clock_seconds": glance_time,
            },
            f,
        )

    total_time = time.time() - t_total_start
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE (Random Forest)")
    print("=" * 80)
    print(f"\nTotal wall-clock: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"  Train:   {train_time:.1f}s")
    print(f"  Wachter: {wachter_time:.1f}s")
    print(f"  GLANCE:  {glance_time:.1f}s")
    print(f"\nResults saved to: {results_dir}")

    return {
        "dataset": dataset,
        "model": model,
        "wachter_results": wachter_results,
        "glance_results": glance_results,
        "timings": {
            "train": train_time,
            "wachter": wachter_time,
            "glance": glance_time,
            "total": total_time,
        },
    }


if __name__ == "__main__":
    main()
