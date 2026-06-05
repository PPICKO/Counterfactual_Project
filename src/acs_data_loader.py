"""
Data loader for ACS (American Community Survey) dataset using Folktables.

The ACS Income dataset is similar to the Adult dataset but uses more recent
census data from the American Community Survey.
Protected attributes: sex, race
Target: income (>50K or <=50K)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import Config, DEFAULT_CONFIG


class ACSDataLoader:
    """Loads and preprocesses the ACS Income dataset using Folktables."""

    def __init__(
        self,
        data_dir: str = "../data",
        survey_year: Optional[str] = None,
        horizon: Optional[str] = None,
        survey: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the ACS data loader.

        Args:
            data_dir: Directory for storing downloaded data.
            survey_year: ACS survey year (default: from config).
            horizon: Survey horizon, e.g., '1-Year' (default: from config).
            survey: Survey type, e.g., 'person' (default: from config).
            config: Optional Config object for settings.
        """
        self.config = config or DEFAULT_CONFIG
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.survey_year = survey_year or self.config.survey_year
        self.horizon = horizon or self.config.horizon
        self.survey = survey or self.config.survey

        # Protected attributes
        self.protected_attrs: List[str] = ["SEX", "RAC1P"]

        self.scaler: Optional[StandardScaler] = None

        # ACSIncome task uses these features:
        # AGEP (age), COW (class of worker), SCHL (educational attainment),
        # MAR (marital status), OCCP (occupation), POBP (place of birth),
        # RELP (relationship), WKHP (hours worked per week),
        # SEX (sex), RAC1P (race)
        # Target: PINCP (income) > 50000

    def load_raw_data(
        self,
        states: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Load raw ACS dataset from Folktables.

        Args:
            states: List of state codes to include (default: from config).
                   For full US data, use states=None or list of all states.

        Returns:
            Tuple of:
                - features: DataFrame with features.
                - labels: Array with binary labels (1 if income > 50K).
                - df: Full dataframe with all columns.

        Raises:
            ConnectionError: If unable to download ACS data.
        """
        states = states or self.config.states
        print(f"Loading ACS {self.survey_year} {self.horizon} data...")

        try:
            data_source = ACSDataSource(
                survey_year=self.survey_year,
                horizon=self.horizon,
                survey=self.survey,
            )

            # Download data for specified states
            acs_data = data_source.get_data(states=states, download=True)
        except Exception as e:
            raise ConnectionError(f"Failed to download ACS data: {e}") from e

        print(f"Raw data shape: {acs_data.shape}")
        print(f"Columns: {list(acs_data.columns)}")

        # Use ACSIncome task definition
        features, labels, _ = ACSIncome.df_to_pandas(acs_data)

        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Positive rate: {float(labels.mean()):.2%}")

        return features, labels, acs_data

    def preprocess(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        fit: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Preprocess the dataset.

        Args:
            features: Feature DataFrame from Folktables.
            labels: Binary labels.
            fit: If True, fit scaler. If False, use existing scaler.

        Returns:
            Tuple of:
                - X: Preprocessed feature matrix.
                - y: Labels.
                - protected: Dictionary with protected attribute values.

        Raises:
            ValueError: If scaler not fitted when fit=False.
        """
        # Extract protected attributes before preprocessing
        protected: Dict[str, np.ndarray] = {
            "SEX": features["SEX"].values,
            "RAC1P": features["RAC1P"].values,
        }

        # Convert to numpy array
        X = features.values.astype(float)
        y = labels.values.astype(int)

        # Handle any NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize features
        if fit:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Must fit scaler before transforming")
            X = self.scaler.transform(X)

        print(f"After preprocessing: {X.shape}")
        print(
            f"Protected attributes: SEX {len(np.unique(protected['SEX']))} unique, "
            f"RAC1P {len(np.unique(protected['RAC1P']))} unique"
        )

        return X, y, protected

    def load_processed_data(
        self,
        states: Optional[List[str]] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load and preprocess the complete dataset.

        Args:
            states: List of state codes (default: from config).
                   For multiple states: ['CA', 'NY', 'TX', ...]
            test_size: Proportion for test set (default: from config).
            val_size: Proportion for validation set (default: from config).
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
        """
        states = states or self.config.states
        test_size = test_size or self.config.test_size
        val_size = val_size or self.config.val_size
        random_state = random_state or self.config.random_seed

        # Load raw data
        features, labels, full_df = self.load_raw_data(states=states)

        # First split: separate test set
        indices = np.arange(len(features))
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Preprocess full dataset
        X_full, y_full, protected_full = self.preprocess(features, labels, fit=True)

        # Split into train/val/test
        X_train_val = X_full[train_val_idx]
        y_train_val = y_full[train_val_idx]
        X_test = X_full[test_idx]
        y_test = y_full[test_idx]

        protected_train_val = {k: v[train_val_idx] for k, v in protected_full.items()}
        protected_test = {k: v[test_idx] for k, v in protected_full.items()}

        # Second split: separate validation from training
        train_idx, val_idx = train_test_split(
            np.arange(len(X_train_val)),
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_val,
        )

        X_train = X_train_val[train_idx]
        X_val = X_train_val[val_idx]
        y_train = y_train_val[train_idx]
        y_val = y_train_val[val_idx]

        protected_train = {k: v[train_idx] for k, v in protected_train_val.items()}
        protected_val = {k: v[val_idx] for k, v in protected_train_val.items()}

        # Get feature names from ACSIncome
        feature_names = list(features.columns)

        # Identify continuous vs categorical features
        # In ACS, most are categorical codes, but AGEP and WKHP are continuous
        continuous_features = ["AGEP", "WKHP"]
        categorical_features = [f for f in feature_names if f not in continuous_features]

        dataset: Dict[str, Any] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "protected_train": protected_train,
            "protected_val": protected_val,
            "protected_test": protected_test,
            "feature_names": feature_names,
            "continuous_features": continuous_features,
            "categorical_features": categorical_features,
            "scaler": self.scaler,
        }

        print("\nDataset Statistics:")
        print(f"Train: {len(X_train)} samples, {y_train.mean():.2%} positive")
        print(f"Val: {len(X_val)} samples, {y_val.mean():.2%} positive")
        print(f"Test: {len(X_test)} samples, {y_test.mean():.2%} positive")
        print(f"Features: {X_train.shape[1]}")

        return dataset


def main() -> None:
    """Test the ACS data loader."""
    loader = ACSDataLoader(data_dir="../data")

    # Load California data (smaller for testing)
    # For full experiment, use states=['CA', 'NY', 'TX', 'FL', 'PA'] or more
    dataset = loader.load_processed_data(states=["CA"])

    print("\nSample features:")
    print(dataset["feature_names"])

    print("\nProtected attribute distribution (train):")
    for attr in ["SEX", "RAC1P"]:
        unique, counts = np.unique(dataset["protected_train"][attr], return_counts=True)
        print(f"{attr}:")
        for val, count in zip(unique, counts):
            print(
                f"  {val}: {count} ({count/len(dataset['protected_train'][attr]):.2%})"
            )

    # Decode protected attributes
    print("\nProtected attribute meanings:")
    print("SEX: 1=Male, 2=Female")
    print("RAC1P: 1=White alone, 2=Black alone, 3-9=Other races")


if __name__ == "__main__":
    main()
