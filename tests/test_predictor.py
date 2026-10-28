"""Unit tests for CropPredictor."""

import pytest
from src.core.exceptions import ValidationError
from src.ml.predictor import CropPredictor


@pytest.fixture
def predictor():
    """Initializes a CropPredictor instance."""
    return CropPredictor()


def test_predictor_initialization(predictor):
    """Verifies that model and preprocessor load successfully."""
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert predictor.preprocessor.is_fitted


def test_predict_realistic_rice(predictor):
    """Tests prediction on typical rice environmental input."""
    rice_inputs = {
        "Nitrogen": 90,
        "Phosporus": 42,
        "Potassium": 43,
        "Temperature": 20.87,
        "Humidity": 82.0,
        "Ph": 6.5,
        "Rainfall": 202.93
    }
    result = predictor.predict(rice_inputs, top_k=5)

    assert result["status"] == "success"
    assert result["primary_recommendation"]["crop_name"] == "Rice"
    assert result["primary_recommendation"]["model_confidence_score"] > 50.0
    assert len(result["top_candidates"]) == 5
    assert result["top_candidates"][0]["rank"] == 1
    assert result["top_candidates"][0]["crop_name"] == "Rice"
    assert "scientific_disclaimer" in result


def test_predict_realistic_cotton(predictor):
    """Tests prediction on typical cotton input."""
    cotton_inputs = {
        "Nitrogen": 118,
        "Phosporus": 46,
        "Potassium": 20,
        "Temperature": 23.97,
        "Humidity": 81.41,
        "Ph": 6.98,
        "Rainfall": 70.04
    }
    result = predictor.predict(cotton_inputs, top_k=5)
    assert result["primary_recommendation"]["crop_name"] == "Cotton"


def test_top_k_ordering_and_structure(predictor):
    """Ensures Top-K candidates are properly ordered by confidence score."""
    sample_inputs = {
        "Nitrogen": 71,
        "Phosporus": 54,
        "Potassium": 16,
        "Temperature": 22.61,
        "Humidity": 63.70,
        "Ph": 5.72,
        "Rainfall": 87.76
    }
    result = predictor.predict(sample_inputs, top_k=5)
    candidates = result["top_candidates"]

    scores = [c["model_confidence_score"] for c in candidates]
    # Check descending sort
    assert scores == sorted(scores, reverse=True)

    for rank, c in enumerate(candidates, start=1):
        assert c["rank"] == rank
        assert isinstance(c["crop_name"], str)
        assert 0.0 <= c["model_confidence_score"] <= 100.0
        assert c["confidence_level"] in ["High", "Moderate", "Low"]
        assert len(c["explanation"]) > 0


def test_predict_invalid_input(predictor):
    """Ensures invalid inputs raise ValidationError with clean messages."""
    invalid_inputs = {
        "Nitrogen": 90,
        "Phosporus": 42,
        "Potassium": 43,
        "Temperature": 120.0,  # Unrealistic temperature
        "Humidity": 82.0,
        "Ph": 6.5,
        "Rainfall": 202.93
    }
    with pytest.raises(ValidationError, match="outside acceptable range"):
        predictor.predict(invalid_inputs)
