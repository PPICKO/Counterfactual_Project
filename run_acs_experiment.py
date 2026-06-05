"""
Run complete counterfactual experiment on ACS dataset.

This script:
1. Loads ACS data from Folktables
2. Trains a classifier
3. Generates counterfactuals using Wachter and GLANCE methods
4. Saves results for comparison analysis
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.acs_data_loader import ACSDataLoader
from src.config import Config
from src.glance_method import GLANCECounterfactual, evaluate_glance
from src.wachter_method import WachterCounterfactual, evaluate_counterfactuals


def main() -> Dict[str, Any]:
    """
    Run the complete ACS counterfactual experiment.

    Returns:
        Dictionary containing dataset, model, and results from both methods.
    """
    config = Config()

    print("=" * 80)
    print("COUNTERFACTUAL EXPERIMENT ON ACS DATASET")
    print("=" * 80)

    # Create results directory
    results_dir = Path("results_acs")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "wachter").mkdir(exist_ok=True)
    (results_dir / "glance").mkdir(exist_ok=True)
    (results_dir / "comparison").mkdir(exist_ok=True)

    # Step 1: Load ACS data
    print("\n" + "=" * 80)
    print("STEP 1: Loading ACS Dataset")
    print("=" * 80)

    loader = ACSDataLoader(data_dir="data")
    # Use California data for faster experimentation
    dataset = loader.load_processed_data(states=["CA"])

    # Save dataset info
    with open(results_dir / "dataset_info.txt", "w") as f:
        f.write("ACS Dataset Information\n")
        f.write("=" * 80 + "\n")
        f.write(f"Train samples: {len(dataset['X_train'])}\n")
        f.write(f"Val samples: {len(dataset['X_val'])}\n")
        f.write(f"Test samples: {len(dataset['X_test'])}\n")
        f.write(f"Features: {dataset['X_train'].shape[1]}\n")
        f.write(f"Feature names: {dataset['feature_names']}\n")
        f.write(f"\nProtected attributes: SEX, RAC1P\n")
        f.write(f"SEX encoding: 1=Male, 2=Female\n")
        f.write(f"RAC1P encoding: 1=White, 2=Black, 3-9=Other\n")

    # Step 2: Train classifier
    print("\n" + "=" * 80)
    print("STEP 2: Training Classifier")
    print("=" * 80)

    model = LogisticRegression(
        max_iter=config.classifier_max_iter,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    model.fit(dataset["X_train"], dataset["y_train"])

    # Evaluate
    y_test_pred = model.predict(dataset["X_test"])
    y_test_proba = model.predict_proba(dataset["X_test"])[:, 1]

    print(f"\nTest Accuracy: {accuracy_score(dataset['y_test'], y_test_pred):.4f}")
    print(f"Test AUC: {roc_auc_score(dataset['y_test'], y_test_proba):.4f}")

    # Save model
    with open(results_dir / "classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    # Step 3: Select test samples for counterfactual generation
    print("\n" + "=" * 80)
    print("STEP 3: Selecting Test Samples")
    print("=" * 80)

    # Select 100 negative predictions (same as original experiment)
    negative_mask = y_test_pred == 0
    negative_indices = np.where(negative_mask)[0]

    # Sample 100 instances
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

    # Step 4: Generate Wachter counterfactuals
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

    # Save Wachter results
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

    # Step 5: Generate GLANCE counterfactuals
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

    # Discover global rules from negative samples
    X_negative = dataset["X_test"][negative_mask]
    print(f"\nDiscovering global rules from {min(200, len(X_negative))} samples...")
    global_rules = glance.discover_global_rules(
        X_negative,
        target_class=1,
        max_samples=config.glance_max_samples,
        verbose=True,
    )

    glance.print_global_rules()

    # Apply rules to our 100 test samples
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

    # Save GLANCE results
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
    print(f"  - Classifier: {results_dir / 'classifier.pkl'}")
    print(f"  - Wachter results: {results_dir / 'wachter/results.pkl'}")
    print(f"  - GLANCE results: {results_dir / 'glance/results.pkl'}")
    print(f"\nTo run comparison analysis:")
    print(f"  python run_acs_comparison.py")

    return {
        "dataset": dataset,
        "model": model,
        "wachter_results": wachter_results,
        "glance_results": glance_results,
    }


if __name__ == "__main__":
    results = main()
