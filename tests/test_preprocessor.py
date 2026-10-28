"""Unit tests for AgriDataPreprocessor."""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.core.config import FEATURE_COLUMNS, SCALER_PATH
from src.core.exceptions import ValidationError, ModelArtifactError
from src.ml.preprocessor import AgriDataPreprocessor


def test_load_dataset():
    """Verifies that the dataset loads properly with correct shape and mapping."""
    X, y = AgriDataPreprocessor.load_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == 2200
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.nunique() == 22
    assert y.min() == 1
    assert y.max() == 22


def test_preprocessor_fit_transform():
    """Verifies that features are properly scaled to [0, 1] range."""
    X, _ = AgriDataPreprocessor.load_dataset()
    preprocessor = AgriDataPreprocessor()
    X_scaled = preprocessor.fit_transform(X)

    assert X_scaled.shape == X.shape
    assert np.all(X_scaled >= -1e-7)
    assert np.all(X_scaled <= 1.0 + 1e-7)
    assert preprocessor.is_fitted


def test_transform_immutability():
    """Ensures calling transform never mutates the learned scaler parameters."""
    X, _ = AgriDataPreprocessor.load_dataset()
    preprocessor = AgriDataPreprocessor()
    preprocessor.fit(X)

    original_min = preprocessor.scaler.data_min_.copy()
    original_max = preprocessor.scaler.data_max_.copy()

    # Transform an arbitrary single sample
    sample = np.array([[10, 10, 10, 15, 80, 4.5, 10]])
    _ = preprocessor.transform(sample)

    assert np.array_equal(preprocessor.scaler.data_min_, original_min), "Scaler minimum was mutated!"
    assert np.array_equal(preprocessor.scaler.data_max_, original_max), "Scaler maximum was mutated!"


def test_refuse_fit_on_single_sample():
    """Prevents the fatal legacy bug where fit was called on a single 1-row vector."""
    preprocessor = AgriDataPreprocessor()
    single_sample = np.array([[10, 10, 10, 15, 80, 4.5, 10]])

    with pytest.raises(ValidationError, match="Cannot fit scaler on a single sample"):
        preprocessor.fit(single_sample)


def test_validate_input_valid():
    """Tests successful validation on valid user input dictionary."""
    valid_dict = {
        "Nitrogen": 90,
        "Phosporus": 42,
        "Potassium": 43,
        "Temperature": 20.87,
        "Humidity": 82.0,
        "Ph": 6.5,
        "Rainfall": 202.93
    }
    vec = AgriDataPreprocessor.validate_input(valid_dict)
    assert vec.shape == (1, 7)
    assert vec[0][0] == 90.0
    assert vec[0][5] == 6.5


def test_validate_input_missing_field():
    """Tests that missing required field raises ValidationError."""
    incomplete_dict = {
        "Nitrogen": 90,
        "Potassium": 43,
        # Phosphorus missing
        "Temperature": 20.87,
        "Humidity": 82.0,
        "Ph": 6.5,
        "Rainfall": 202.93
    }
    with pytest.raises(ValidationError, match="Missing required parameter"):
        AgriDataPreprocessor.validate_input(incomplete_dict)


def test_validate_input_non_numeric():
    """Tests that non-numeric inputs raise ValidationError."""
    bad_dict = {
        "Nitrogen": "not_a_number",
        "Phosporus": 42,
        "Potassium": 43,
        "Temperature": 20.87,
        "Humidity": 82.0,
        "Ph": 6.5,
        "Rainfall": 202.93
    }
    with pytest.raises(ValidationError, match="Invalid non-numeric value"):
        AgriDataPreprocessor.validate_input(bad_dict)


def test_validate_input_out_of_bounds():
    """Tests that agronomic out-of-range inputs raise ValidationError."""
    # Negative pH
    bad_ph = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.0, "Ph": -2.0, "Rainfall": 202.93
    }
    with pytest.raises(ValidationError, match="outside acceptable range"):
        AgriDataPreprocessor.validate_input(bad_ph)

    # Extreme rainfall
    bad_rain = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.0, "Ph": 6.5, "Rainfall": 5000.0
    }
    with pytest.raises(ValidationError, match="outside acceptable range"):
        AgriDataPreprocessor.validate_input(bad_rain)


def test_detect_corrupted_scaler_artifact(tmp_path):
    """Verifies that AgriDataPreprocessor.load rejects corrupted scalers where data_min == data_max."""
    import pickle
    corrupted_scaler = MinMaxScaler()
    # Force single-sample fit
    corrupted_scaler.fit(np.array([[10, 10, 10, 15, 80, 4.5, 10]]))
    corrupted_path = tmp_path / "corrupted_scaler.pkl"
    with open(corrupted_path, "wb") as f:
        pickle.dump(corrupted_scaler, f)

    with pytest.raises(ModelArtifactError, match="Detected corrupted scaler"):
        AgriDataPreprocessor.load(corrupted_path)
