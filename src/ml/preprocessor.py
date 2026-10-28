"""AgriDataPreprocessor: Leakage-free, reproducible preprocessing and input validation."""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.core.config import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    CROP_NAME_TO_ID,
    SCALER_PATH,
    RAW_DATA_PATH,
    LEGACY_DATA_PATH
)
from src.core.exceptions import ValidationError, ModelArtifactError


class AgriDataPreprocessor:
    """Preprocesses agronomic features for training and live inference.
    
    Guarantees:
    - Scaler is fitted ONLY on training data.
    - Transform never refits or overwrites learned normalization parameters.
    - Strict input bounds and data type validation before transformation.
    """

    def __init__(self, scaler: Optional[MinMaxScaler] = None):
        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self.is_fitted = hasattr(self.scaler, "data_min_") and self.scaler.data_min_ is not None

    @classmethod
    def load_dataset(cls, data_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads and validates the crop recommendation dataset.
        
        Returns:
            X (pd.DataFrame): 7 feature columns (N, P, K, temperature, humidity, ph, rainfall)
            y (pd.Series): Target integer labels (1 to 22)
        """
        if data_path is None:
            if RAW_DATA_PATH.exists():
                data_path = RAW_DATA_PATH
            elif LEGACY_DATA_PATH.exists():
                data_path = LEGACY_DATA_PATH
            else:
                raise FileNotFoundError(f"Crop dataset not found at {RAW_DATA_PATH} or {LEGACY_DATA_PATH}")

        df = pd.read_csv(data_path)

        # Validate mandatory columns
        required_cols = FEATURE_COLUMNS + ["label"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValidationError(f"Dataset is missing required columns: {missing_cols}")

        if df.isnull().sum().sum() > 0:
            raise ValidationError("Dataset contains missing values. Clean the data before training.")

        X = df[FEATURE_COLUMNS].copy()
        
        # Map crop string labels to 1-indexed integers
        y = df["label"].str.lower().map(CROP_NAME_TO_ID)
        if y.isnull().any():
            unmapped = df["label"][y.isnull()].unique().tolist()
            raise ValidationError(f"Encountered unknown crop labels in dataset: {unmapped}")

        return X, y

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "AgriDataPreprocessor":
        """Fits the MinMaxScaler strictly on training feature matrix.
        
        Refuses to fit on a single sample to prevent catastrophic scaler corruption.
        """
        # Handle DataFrame or ndarray
        if isinstance(X, pd.DataFrame):
            X_arr = X[FEATURE_COLUMNS].values
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if len(X_arr) <= 1:
            raise ValidationError("Cannot fit scaler on a single sample. Scaler must be fit on training dataset.")

        self.scaler.fit(X_arr)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transforms features using the fitted scaler parameters without mutating them."""
        if not self.is_fitted:
            raise ModelArtifactError("AgriDataPreprocessor is not fitted. Fit on training data or load saved scaler.")

        if isinstance(X, pd.DataFrame):
            X_arr = X[FEATURE_COLUMNS].values
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != len(FEATURE_COLUMNS):
            raise ValidationError(
                f"Expected {len(FEATURE_COLUMNS)} features ({FEATURE_COLUMNS}), but got {X_arr.shape[1]}."
            )

        return self.scaler.transform(X_arr)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fits on training set and returns transformed features."""
        return self.fit(X).transform(X)

    @staticmethod
    def validate_input(inputs: Dict[str, Union[float, int, str]]) -> np.ndarray:
        """Validates raw user input dictionary and returns a 1x7 numpy array.
        
        Supports legacy typos (e.g. 'phosporus') and short aliases.
        
        Raises:
            ValidationError: If any field is missing, not a valid number, or outside agronomic limits.
        """
        # Feature alias mapping for resilient input parsing
        KEY_ALIASES = {
            "N": ["n", "nitrogen"],
            "P": ["p", "phosphorus", "phosporus"],
            "K": ["k", "potassium"],
            "temperature": ["temperature", "temp"],
            "humidity": ["humidity"],
            "ph": ["ph", "soil_ph"],
            "rainfall": ["rainfall", "rain"]
        }

        # Case-insensitive, stripped key matching
        normalized_keys = {k.strip().lower(): v for k, v in inputs.items()}
        cleaned_values = []

        for col in FEATURE_COLUMNS:
            spec = FEATURE_SPECS[col]
            matching_key = None
            
            # Match aliases
            candidates = KEY_ALIASES.get(col, [col.lower(), spec["name"].lower()])
            for candidate in candidates:
                if candidate in normalized_keys:
                    matching_key = candidate
                    break

            if matching_key is None:
                raise ValidationError(f"Missing required parameter: {spec['name']} ({col})", field=col)

            raw_val = normalized_keys[matching_key]

            # Validate numeric conversion
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid non-numeric value for {spec['name']}: '{raw_val}'",
                    field=col,
                    value=raw_val
                )

            if np.isnan(val) or np.isinf(val):
                raise ValidationError(f"Value for {spec['name']} cannot be NaN or Infinite", field=col, value=val)

            # Validate agronomic bounds
            if val < spec["min"] or val > spec["max"]:
                raise ValidationError(
                    f"{spec['name']} value {val} is outside acceptable range [{spec['min']}, {spec['max']} {spec['unit']}].",
                    field=col,
                    value=val
                )

            cleaned_values.append(val)

        return np.array(cleaned_values, dtype=np.float64).reshape(1, -1)

    def save(self, file_path: Optional[Union[str, Path]] = None) -> Path:
        """Serializes the fitted scaler to disk."""
        if not self.is_fitted:
            raise ModelArtifactError("Cannot save an unfitted scaler.")

        save_path = Path(file_path or SCALER_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.scaler, f)
        return save_path

    @classmethod
    def load(cls, file_path: Optional[Union[str, Path]] = None) -> "AgriDataPreprocessor":
        """Loads and verifies a saved scaler artifact."""
        load_path = Path(file_path or SCALER_PATH)
        if not load_path.exists():
            raise ModelArtifactError(f"Scaler file not found at: {load_path}")

        with open(load_path, "rb") as f:
            scaler = pickle.load(f)

        # Integrity verification: ensure scaler is valid and not corrupted by single-sample minimums
        if not hasattr(scaler, "data_min_") or scaler.data_min_ is None:
            raise ModelArtifactError(f"Loaded scaler artifact at {load_path} is missing fitted parameters.")

        # Check for the known legacy bug: data_min_ == data_max_
        if hasattr(scaler, "data_max_") and np.allclose(scaler.data_min_, scaler.data_max_):
            raise ModelArtifactError(
                f"Detected corrupted scaler at {load_path} (data_min_ equals data_max_). Re-train and save a valid scaler."
            )

        instance = cls(scaler=scaler)
        return instance
