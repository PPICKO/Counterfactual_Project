"""
Run complete counterfactual experiment on ACS dataset with RANDOM FOREST.

Mirrors run_acs_experiment.py exactly, with one change: the classifier is
sklearn.ensemble.RandomForestClassifier instead of LogisticRegression.
Outputs go to results_acs_rf/ to keep the original LR results intact.

This script addresses reviewer concern that linear decision boundaries
(rather than dataset characteristics) might be driving fairness outcomes.
"""

import pickle
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.acs_data_loader import ACSDataLoader
from src.config import Config
from src.glance_method import GLANCECounterfactual, evaluate_glance
from src.wachter_method import WachterCounterfactual, evaluate_counterfactuals


def main() -> Dict[str, Any]:
    """Run the complete ACS counterfactual experiment with Random Forest."""
    config = Config()
    t_total_start = time.time()

    print("=" * 80)
    print("COUNTERFACTUAL EXPERIMENT ON ACS DATASET (RANDOM FOREST)")
    print(f"  num_test_samples = {config.num_test_samples}")
    print(f"  n_estimators     = {config.n_estimators}")
    print(f"  max_depth        = {config.max_depth}")
    print("=" * 80)

    results_dir = Path("results_acs_rf")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "wachter").mkdir(exist_ok=True)
    (results_dir / "glance").mkdir(exist_ok=True)
    (results_dir / "comparison").mkdir(exist_ok=True)

    # Step 1: Load ACS data
    print("\n" + "=" * 80)
    print("STEP 1: Loading ACS Dataset")
    print("=" * 80)

    loader = ACSDataLoader(data_dir="data")
    dataset = loader.load_processed_data(states=["CA"])

    with open(results_dir / "dataset_info.txt", "w") as f:
        f.write("ACS Dataset Information (Random Forest run)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Train samples: {len(dataset['X_train'])}\n")
        f.write(f"Val samples: {len(dataset['X_val'])}\n")
        f.write(f"Test samples: {len(dataset['X_test'])}\n")
        f.write(f"Features: {dataset['X_train'].shape[1]}\n")
        f.write(f"Feature names: {dataset['feature_names']}\n")
        f.write("\nProtected attributes: SEX, RAC1P\n")
        f.write("SEX encoding: 1=Male, 2=Female\n")
        f.write("RAC1P encoding: 1=White, 2=Black, 3-9=Other\n")
        f.write("\nClassifier: RandomForestClassifier\n")
        f.write(f"  n_estimators={config.n_estimators}, "
                f"max_depth={config.max_depth}, random_state={config.random_seed}\n")

    # Step 2: Train classifier (Random Forest)
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
    # ravel y to silence sklearn DataConversionWarning
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

    # Step 3: Select test samples
    print("\n" + "=" * 80)
    print("STEP 3: Selecting Test Samples")
    print("=" * 80)

    negative_mask = y_test_pred == 0
    negative_indices = np.where(negative_mask)[0]

    np.random.seed(config.random_seed)
    if len(negative_indices) > config.num_test_samples:
        sample_indices = np.random.choice(
            negative_indices,
            size=config.num_test_samples,
            replace=False,
        )
    else:
        sample_indices = negative_indices

    X_sample = dataset["X_test"][sample_indices]
    protected_sample_sex = dataset["protected_test"]["SEX"][sample_indices]

    print(f"Selected {len(X_sample)} samples with negative predictions")
    sex_counts = np.unique(protected_sample_sex, return_counts=True)
    print(f"  SEX distribution: {dict(zip(sex_counts[0], sex_counts[1]))}")

    # Step 4: Wachter counterfactuals
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

    # Step 5: GLANCE counterfactuals
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
    print(f"\nDiscovering global rules from {min(200, len(X_negative))} samples...")
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
