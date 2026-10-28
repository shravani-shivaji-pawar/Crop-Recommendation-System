"""Regression tests ensuring the legacy scaling bug never returns."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.core.config import RANDOM_STATE, TEST_SPLIT_RATIO, CROP_ID_TO_NAME
from src.ml.preprocessor import AgriDataPreprocessor
from src.ml.predictor import CropPredictor


def test_legacy_bug_resolution():
    """Confirms that inputs that collapsed to Apple/Grapes in legacy app predict their true classes."""
    predictor = CropPredictor()

    test_cases = [
        ("Rice", {"Nitrogen": 90, "Phosporus": 42, "Potassium": 43, "Temperature": 20.87, "Humidity": 82.00, "Ph": 6.50, "Rainfall": 202.93}),
        ("Maize", {"Nitrogen": 71, "Phosporus": 54, "Potassium": 16, "Temperature": 22.61, "Humidity": 63.70, "Ph": 5.72, "Rainfall": 87.76}),
        ("Chickpea", {"Nitrogen": 40, "Phosporus": 67, "Potassium": 79, "Temperature": 17.02, "Humidity": 16.99, "Ph": 7.39, "Rainfall": 80.20}),
        ("Coffee", {"Nitrogen": 101, "Phosporus": 29, "Potassium": 32, "Temperature": 26.54, "Humidity": 57.77, "Ph": 6.78, "Rainfall": 158.48}),
        ("Cotton", {"Nitrogen": 118, "Phosporus": 46, "Potassium": 20, "Temperature": 23.97, "Humidity": 81.41, "Ph": 6.98, "Rainfall": 70.04}),
    ]

    for expected_crop, inputs in test_cases:
        res = predictor.predict(inputs)
        predicted_crop = res["primary_recommendation"]["crop_name"]
        assert predicted_crop == expected_crop, (
            f"Regression detected! Expected {expected_crop}, but got {predicted_crop}. "
            f"Confidence: {res['primary_recommendation']['model_confidence_score']}%"
        )


def test_scaler_immutability_under_stress():
    """Runs 100 consecutive predictions with varying inputs and ensures scaler parameters remain static."""
    predictor = CropPredictor()
    scaler = predictor.preprocessor.scaler

    initial_min = scaler.data_min_.copy()
    initial_max = scaler.data_max_.copy()

    # Vary inputs across realistic ranges
    for i in range(100):
        varied_inputs = {
            "Nitrogen": 10 + (i % 120),
            "Phosporus": 10 + (i % 100),
            "Potassium": 10 + (i % 150),
            "Temperature": 15.0 + (i % 25),
            "Humidity": 20.0 + (i % 70),
            "Ph": 4.5 + ((i % 45) / 10.0),
            "Rainfall": 30.0 + (i % 200)
        }
        res = predictor.predict(varied_inputs)
        assert res["status"] == "success"

    # Assert that scaler minimum and maximum were not altered by even 1 epsilon
    assert np.array_equal(scaler.data_min_, initial_min), "Scaler data_min_ changed during live predictions!"
    assert np.array_equal(scaler.data_max_, initial_max), "Scaler data_max_ changed during live predictions!"


def test_held_out_accuracy_benchmark():
    """Empirically evaluates the production predictor on the held-out test split."""
    X, y = AgriDataPreprocessor.load_dataset()

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT_RATIO,
        stratify=y,
        random_state=RANDOM_STATE
    )

    predictor = CropPredictor()
    X_test_scaled = predictor.preprocessor.transform(X_test)
    preds = predictor.model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    assert acc >= 0.99, f"Production model accuracy {acc*100:.2f}% is below 99% threshold!"
