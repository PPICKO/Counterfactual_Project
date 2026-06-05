"""
Data loader for Adult Income dataset from UCI ML Repository.

Downloads and preprocesses the Adult dataset for counterfactual experiments.
Protected attributes: sex, race
Target: income (>50K or <=50K)
"""

import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import Config, DEFAULT_CONFIG


class AdultDataLoader:
    """Loads and preprocesses the Adult Income dataset."""

    # Column names for Adult dataset
    COLUMN_NAMES: List[str] = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    # Protected attributes
    PROTECTED_ATTRS: List[str] = ["sex", "race"]

    # Feature types
    CONTINUOUS_FEATURES: List[str] = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    CATEGORICAL_FEATURES: List[str] = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    def __init__(
        self,
        data_dir: str = "../data",
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the Adult data loader.

        Args:
            data_dir: Directory for storing downloaded data.
            config: Optional Config object for settings.
        """
        self.config = config or DEFAULT_CONFIG
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None

    def download_data(self) -> Tuple[Path, Path]:
        """
        Download Adult dataset from UCI if not already present.

        Returns:
            Tuple of (train_path, test_path) for the downloaded files.

        Raises:
            ConnectionError: If unable to download data from UCI repository.
            IOError: If unable to save downloaded data to disk.
        """
        train_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        )
        test_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        )

        train_path = self.data_dir / "adult.data"
        test_path = self.data_dir / "adult.test"

        if not train_path.exists():
            print(f"Downloading training data from {train_url}")
            try:
                urllib.request.urlretrieve(train_url, train_path)
                print(f"Saved to {train_path}")
            except urllib.error.URLError as e:
                raise ConnectionError(
                    f"Failed to download training data from {train_url}: {e}"
                ) from e
            except IOError as e:
                raise IOError(
                    f"Failed to save training data to {train_path}: {e}"
                ) from e

        if not test_path.exists():
            print(f"Downloading test data from {test_url}")
            try:
                urllib.request.urlretrieve(test_url, test_path)
                print(f"Saved to {test_path}")
            except urllib.error.URLError as e:
                raise ConnectionError(
                    f"Failed to download test data from {test_url}: {e}"
                ) from e
            except IOError as e:
                raise IOError(f"Failed to save test data to {test_path}: {e}") from e

        return train_path, test_path

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw Adult dataset.

        Returns:
            Tuple of (train_df, test_df) DataFrames.
        """
        train_path, test_path = self.download_data()

        # Load training data
        df_train = pd.read_csv(
            train_path,
            names=self.COLUMN_NAMES,
            skipinitialspace=True,
            na_values="?",
        )

        # Load test data (skip first line which is a comment)
        df_test = pd.read_csv(
            test_path,
            names=self.COLUMN_NAMES,
            skipinitialspace=True,
            na_values="?",
            skiprows=1,
        )

        # Clean income labels (test set has a dot)
        df_test["income"] = df_test["income"].str.rstrip(".")

        print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
        print(f"Missing values:\n{df_train.isnull().sum()}")

        return df_train, df_test

    def preprocess(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Preprocess the dataset.

        Args:
            df: Raw dataframe.
            fit: If True, fit encoders/scalers. If False, use existing ones.

        Returns:
            Tuple of:
                - X: Feature matrix (np.ndarray of shape (n_samples, n_features)).
                - y: Labels (np.ndarray, 0 for <=50K, 1 for >50K).
                - protected: Dictionary with protected attribute values.
        """
        # Drop rows with missing values
        df_clean = df.dropna()
        print(f"After dropping NAs: {len(df_clean)} samples")

        # Extract target
        y = (df_clean["income"] == ">50K").astype(int).values

        # Extract protected attributes before encoding
        # Note: No need for .copy() as we're extracting values directly
        protected = {
            "sex": df_clean["sex"].values,
            "race": df_clean["race"].values,
        }

        # Separate features
        X = df_clean.drop("income", axis=1).copy()

        # Encode categorical features
        for col in self.CATEGORICAL_FEATURES:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                # Handle unseen categories safely
                le = self.label_encoders[col]
                known_classes = set(le.classes_)
                X[col] = X[col].apply(
                    lambda x: (
                        le.transform([x])[0] if x in known_classes else len(le.classes_)
                    )
                )

        # Scale continuous features
        if fit:
            self.scaler = StandardScaler()
            X[self.CONTINUOUS_FEATURES] = self.scaler.fit_transform(
                X[self.CONTINUOUS_FEATURES]
            )
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call preprocess with fit=True first.")
            X[self.CONTINUOUS_FEATURES] = self.scaler.transform(
                X[self.CONTINUOUS_FEATURES]
            )

        # Drop fnlwgt (sampling weight, not meaningful for predictions)
        if "fnlwgt" in X.columns:
            X = X.drop("fnlwgt", axis=1)

        return X.values, y, protected

    def load_processed_data(
        self,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load and preprocess the complete dataset.

        Args:
            test_size: Proportion for test set (default: from config).
            random_state: Random seed (default: from config).

        Returns:
            Dictionary with train/val/test splits and metadata:
                - X_train, X_val, X_test: Feature matrices.
                - y_train, y_val, y_test: Labels.
                - protected_train, protected_val, protected_test: Protected attributes.
                - feature_names: List of feature names.
                - continuous_features: List of continuous feature names.
                - categorical_features: List of categorical feature names.
                - scaler: Fitted StandardScaler.
                - label_encoders: Dictionary of fitted LabelEncoders.
        """
        test_size = test_size or self.config.test_size
        random_state = random_state or self.config.random_seed

        # Load raw data
        df_train, df_test = self.load_raw_data()

        # Preprocess training data
        X_train_full, y_train_full, protected_train = self.preprocess(df_train, fit=True)

        # Split into train and validation
        indices = np.arange(len(X_train_full))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y_train_full,
        )

        X_train = X_train_full[train_idx]
        X_val = X_train_full[val_idx]
        y_train = y_train_full[train_idx]
        y_val = y_train_full[val_idx]

        protected_train_dict = {k: v[train_idx] for k, v in protected_train.items()}
        protected_val_dict = {k: v[val_idx] for k, v in protected_train.items()}

        # Preprocess test data
        X_test, y_test, protected_test = self.preprocess(df_test, fit=False)

        # Reconstruct protected attribute dictionaries
        protected_train_split = {
            k: protected_train_dict[k] for k in self.PROTECTED_ATTRS
        }
        protected_val_split = {k: protected_val_dict[k] for k in self.PROTECTED_ATTRS}

        # Get feature names (after dropping fnlwgt)
        feature_names = [col for col in self.COLUMN_NAMES[:-1] if col != "fnlwgt"]

        dataset: Dict[str, Any] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "protected_train": protected_train_split,
            "protected_val": protected_val_split,
            "protected_test": protected_test,
            "feature_names": feature_names,
            "continuous_features": [
                f for f in self.CONTINUOUS_FEATURES if f != "fnlwgt"
            ],
            "categorical_features": self.CATEGORICAL_FEATURES,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
        }

        print("\nDataset Statistics:")
        print(f"Train: {len(X_train)} samples, {(y_train==1).mean():.2%} positive")
        print(f"Val: {len(X_val)} samples, {(y_val==1).mean():.2%} positive")
        print(f"Test: {len(X_test)} samples, {(y_test==1).mean():.2%} positive")
        print(f"Features: {X_train.shape[1]}")

        return dataset


def main() -> None:
    """Test the data loader."""
    loader = AdultDataLoader(data_dir="../data")
    dataset = loader.load_processed_data()

    print("\nSample features:")
    print(dataset["feature_names"])

    print("\nProtected attribute distribution (train):")
    for attr in ["sex", "race"]:
        unique, counts = np.unique(dataset["protected_train"][attr], return_counts=True)
        print(f"{attr}:")
        for val, count in zip(unique, counts):
            print(
                f"  {val}: {count} ({count/len(dataset['protected_train'][attr]):.2%})"
            )


if __name__ == "__main__":
    main()
