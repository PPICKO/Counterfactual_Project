"""
Baseline classifiers for Adult Income prediction.

Implements Logistic Regression and Random Forest classifiers
to serve as black-box models for counterfactual generation.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Config, DEFAULT_CONFIG
from src.data_loader import AdultDataLoader


class BaselineClassifier:
    """Wrapper for baseline classifiers with evaluation utilities."""

    def __init__(
        self,
        model_type: str = "logistic",
        random_state: Optional[int] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the baseline classifier.

        Args:
            model_type: Type of classifier ('logistic' or 'random_forest').
            random_state: Random seed for reproducibility.
            config: Optional Config object for settings.

        Raises:
            ValueError: If model_type is not 'logistic' or 'random_forest'.
        """
        self.config = config or DEFAULT_CONFIG
        self.model_type = model_type
        self.random_state = random_state or self.config.random_seed
        self.model: Optional[Union[LogisticRegression, RandomForestClassifier]] = None
        self.feature_names: Optional[List[str]] = None
        self.metrics: Dict[str, Dict[str, float]] = {}

        # Initialize model
        if model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=self.config.classifier_max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classifier.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
        """
        print(f"Training {self.model_type} classifier...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted probabilities for each class.
        """
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance.

        Args:
            X: Feature matrix.
            y: True labels.
            dataset_name: Name for logging (e.g., 'train', 'val', 'test').

        Returns:
            Dictionary with evaluation metrics:
                - accuracy: Classification accuracy.
                - precision: Precision score.
                - recall: Recall score.
                - f1: F1 score.
                - roc_auc: ROC AUC score.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba),
        }

        self.metrics[dataset_name] = metrics

        print(f"\n{self.model_type.upper()} - {dataset_name.upper()} Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def evaluate_fairness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected_attr: np.ndarray,
        dataset_name: str = "test",
    ) -> Dict[str, Any]:
        """
        Evaluate fairness metrics across protected groups.

        Args:
            X: Feature matrix.
            y: True labels.
            protected_attr: Array of protected attribute values (e.g., sex, race).
            dataset_name: Name for logging.

        Returns:
            Dictionary with fairness metrics per group:
                - Per group: count, positive_rate, true_positive_rate, accuracy, roc_auc.
                - dp_difference: Demographic parity difference.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        unique_groups = np.unique(protected_attr)
        fairness_metrics: Dict[str, Any] = {}

        print(f"\n{self.model_type.upper()} - Fairness Analysis ({dataset_name}):")

        for group in unique_groups:
            mask = protected_attr == group
            if mask.sum() == 0:
                continue

            group_metrics = {
                "count": int(mask.sum()),
                "positive_rate": float((y_pred[mask] == 1).mean()),
                "true_positive_rate": float(y[mask].mean()),
                "accuracy": accuracy_score(y[mask], y_pred[mask]),
                "roc_auc": (
                    roc_auc_score(y[mask], y_proba[mask])
                    if len(np.unique(y[mask])) > 1
                    else np.nan
                ),
            }

            fairness_metrics[group] = group_metrics

            print(f"  {group}:")
            print(f"    Samples: {group_metrics['count']}")
            print(f"    Positive Prediction Rate: {group_metrics['positive_rate']:.4f}")
            print(f"    True Positive Rate: {group_metrics['true_positive_rate']:.4f}")
            print(f"    Accuracy: {group_metrics['accuracy']:.4f}")
            if not np.isnan(group_metrics["roc_auc"]):
                print(f"    ROC AUC: {group_metrics['roc_auc']:.4f}")

        # Calculate demographic parity difference
        positive_rates = [m["positive_rate"] for m in fairness_metrics.values()]
        dp_diff = max(positive_rates) - min(positive_rates) if positive_rates else 0.0
        print(f"\n  Demographic Parity Difference: {dp_diff:.4f}")

        fairness_metrics["dp_difference"] = dp_diff

        return fairness_metrics

    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save trained model to disk.

        Args:
            save_path: Path to save the model.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {save_path}")

    @classmethod
    def load(cls, load_path: Union[str, Path]) -> "BaselineClassifier":
        """
        Load trained model from disk.

        Args:
            load_path: Path to load the model from.

        Returns:
            Loaded BaselineClassifier instance.
        """
        with open(load_path, "rb") as f:
            model_data = pickle.load(f)

        classifier = cls(model_type=model_data["model_type"])
        classifier.model = model_data["model"]
        classifier.feature_names = model_data["feature_names"]
        classifier.metrics = model_data["metrics"]

        print(f"Model loaded from {load_path}")
        return classifier

    def get_feature_importance(
        self,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Get feature importances (for tree-based models or logistic regression).

        Args:
            top_k: Number of top features to return.

        Returns:
            List of (feature_name, importance) tuples sorted by importance.

        Raises:
            ValueError: If feature names not set or model type not supported.
        """
        if self.feature_names is None:
            raise ValueError("Feature names not set. Train the model first.")

        if self.model_type == "logistic":
            # Use absolute coefficients as importance
            importance = np.abs(self.model.coef_[0])
        elif self.model_type == "random_forest":
            importance = self.model.feature_importances_
        else:
            raise ValueError(f"Feature importance not supported for {self.model_type}")

        # Sort by importance
        indices = np.argsort(importance)[::-1][:top_k]
        top_features = [(self.feature_names[i], float(importance[i])) for i in indices]

        return top_features


def train_and_evaluate_models(
    dataset: Dict[str, Any],
    save_dir: Union[str, Path] = "../results/models",
    config: Optional[Config] = None,
) -> Dict[str, BaselineClassifier]:
    """
    Train both Logistic Regression and Random Forest classifiers.

    Args:
        dataset: Dictionary from AdultDataLoader.
        save_dir: Directory to save trained models.
        config: Optional Config object for settings.

    Returns:
        Dictionary mapping model type to trained classifier.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    classifiers: Dict[str, BaselineClassifier] = {}

    for model_type in ["logistic", "random_forest"]:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Classifier")
        print(f"{'='*60}")

        clf = BaselineClassifier(model_type=model_type, config=config)
        clf.feature_names = dataset["feature_names"]

        # Train
        clf.train(dataset["X_train"], dataset["y_train"])

        # Evaluate on all splits
        clf.evaluate(dataset["X_train"], dataset["y_train"], "train")
        clf.evaluate(dataset["X_val"], dataset["y_val"], "val")
        clf.evaluate(dataset["X_test"], dataset["y_test"], "test")

        # Fairness evaluation (sex)
        clf.evaluate_fairness(
            dataset["X_test"],
            dataset["y_test"],
            dataset["protected_test"]["sex"],
            "test_sex",
        )

        # Fairness evaluation (race)
        clf.evaluate_fairness(
            dataset["X_test"],
            dataset["y_test"],
            dataset["protected_test"]["race"],
            "test_race",
        )

        # Feature importance
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(clf.get_feature_importance(top_k=10), 1):
            print(f"  {i}. {feature}: {importance:.4f}")

        # Save model
        clf.save(save_dir / f"{model_type}_classifier.pkl")

        classifiers[model_type] = clf

    return classifiers


def main() -> None:
    """Main function to train and evaluate baseline classifiers."""
    print("Loading Adult Income dataset...")
    loader = AdultDataLoader(data_dir="../data")
    dataset = loader.load_processed_data()

    print("\n" + "=" * 60)
    print("TRAINING BASELINE CLASSIFIERS")
    print("=" * 60)

    classifiers = train_and_evaluate_models(dataset)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModels saved to: results/models/")
    print("Ready for counterfactual generation!")


if __name__ == "__main__":
    main()
