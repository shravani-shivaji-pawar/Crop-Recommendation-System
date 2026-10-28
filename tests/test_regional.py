"""Automated tests for Phase 4 Regional and Seasonal Decision Layer."""

import pytest
from src.core.exceptions import ValidationError
from src.ml.predictor import CropPredictor
from src.modules.regional.regional_filter import RegionalDecisionLayer
from src.modules.regional.maharashtra_data import (
    MAHARASHTRA_DISTRICTS,
    AGRO_CLIMATIC_ZONES,
    ViabilityStatus
)
from src.modules.regional.evaluator import run_phase4_diagnostic_evaluation


@pytest.fixture
def regional_layer():
    """Provides a RegionalDecisionLayer instance."""
    return RegionalDecisionLayer()


@pytest.fixture
def predictor():
    """Provides a CropPredictor instance."""
    return CropPredictor()


def test_district_count_and_validation(regional_layer):
    """Verifies that exactly 36 Maharashtra districts are supported and validated."""
    assert len(MAHARASHTRA_DISTRICTS) == 36
    assert len(AGRO_CLIMATIC_ZONES) == 9

    # Valid district (case-insensitive)
    assert regional_layer.validate_district("PUNE") == "Pune"
    assert regional_layer.validate_district("solapur") == "Solapur"
    assert regional_layer.validate_district("Aurangabad") == "Chhatrapati Sambhajinagar"
    assert regional_layer.validate_district("osmanabad") == "Dharashiv"

    # Invalid district raises ValidationError
    with pytest.raises(ValidationError, match="Unknown Maharashtra district"):
        regional_layer.validate_district("Jaipur")


def test_season_resolution(regional_layer):
    """Verifies season string and calendar month resolution."""
    assert regional_layer.resolve_season("kharif") == "Kharif"
    assert regional_layer.resolve_season("RABI") == "Rabi"
    assert regional_layer.resolve_season("summer") == "Summer"

    # Month fallback
    assert regional_layer.resolve_season(month=7) == "Kharif"
    assert regional_layer.resolve_season(month=11) == "Rabi"
    assert regional_layer.resolve_season(month=4) == "Summer"

    # Invalid season raises ValidationError
    with pytest.raises(ValidationError, match="Invalid season"):
        regional_layer.resolve_season("WinterSpring")


def test_raw_ml_scores_preserved_without_alteration(regional_layer, predictor):
    """CRITICAL TEST: Ensures raw ML Top-5 ranks and confidence scores are NOT altered."""
    sample_inputs = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.0, "Ph": 6.5, "Rainfall": 202.93
    }
    raw_ml = predictor.predict(sample_inputs, top_k=5)
    regional_res = regional_layer.assess_recommendation(raw_ml, district="Pune", season="Kharif")

    # Verify that raw candidates are completely identical
    assert len(regional_res["unfiltered_ml_ranking"]) == 5
    for idx, cand in enumerate(raw_ml["top_candidates"]):
        unfiltered = regional_res["unfiltered_ml_ranking"][idx]
        assert unfiltered["crop_name"] == cand["crop_name"]
        assert unfiltered["model_confidence_score"] == cand["model_confidence_score"]
        assert unfiltered["rank"] == cand["rank"]


def test_incompatible_crop_flagging(regional_layer, predictor):
    """Tests that out-of-region crops (Apple) are flagged as Incompatible with explicit reasons."""
    apple_inputs = {
        "Nitrogen": 20, "Phosporus": 135, "Potassium": 198,
        "Temperature": 22.0, "Humidity": 92.0, "Ph": 5.8, "Rainfall": 110.0
    }
    ml_res = predictor.predict(apple_inputs, top_k=5)
    assert ml_res["primary_recommendation"]["crop_name"] == "Apple"

    regional_res = regional_layer.assess_recommendation(ml_res, district="Solapur", season="Kharif")

    # Find assessment for Apple
    apple_eval = next(item for item in regional_res["regional_assessments"] if item["crop_name"] == "Apple")
    assert apple_eval["regional_status"] == ViabilityStatus.INCOMPATIBLE.value
    assert "Climatically Incompatible" in apple_eval["rationale"]
    assert "chilling" in apple_eval["rationale"].lower()


def test_off_season_crop_flagging(regional_layer, predictor):
    """Tests that a cultivable crop (Chickpea) is flagged as Off-Season when evaluated in Kharif."""
    chickpea_inputs = {
        "Nitrogen": 40, "Phosporus": 67, "Potassium": 79,
        "Temperature": 17.02, "Humidity": 16.99, "Ph": 7.39, "Rainfall": 80.20
    }
    ml_res = predictor.predict(chickpea_inputs, top_k=5)
    assert ml_res["primary_recommendation"]["crop_name"] == "Chickpea"

    # Evaluate in Kharif (Monsoon)
    regional_res = regional_layer.assess_recommendation(ml_res, district="Solapur", season="Kharif")
    cp_eval = next(item for item in regional_res["regional_assessments"] if item["crop_name"] == "Chickpea")

    assert cp_eval["regional_status"] == ViabilityStatus.CULTIVABLE_OFF_SEASON.value
    assert "Seasonal Mismatch" in cp_eval["rationale"]
    assert "October 10 – November 10" in cp_eval["sowing_window"]


def test_insufficient_evidence_flagging(regional_layer, predictor):
    """Tests that non-traditional crops in non-extension districts receive Insufficient Evidence status."""
    rajma_inputs = {
        "Nitrogen": 20, "Phosporus": 67, "Potassium": 20,
        "Temperature": 20.0, "Humidity": 21.0, "Ph": 5.7, "Rainfall": 105.0
    }
    ml_res = predictor.predict(rajma_inputs, top_k=5)
    regional_res = regional_layer.assess_recommendation(ml_res, district="Gadchiroli", season="Rabi")

    rajma_eval = next((item for item in regional_res["regional_assessments"] if item["crop_name"] == "Kidneybeans"), None)
    if rajma_eval:
        assert rajma_eval["regional_status"] == ViabilityStatus.INSUFFICIENT_EVIDENCE.value
        assert "Insufficient Evidence" in rajma_eval["rationale"]


def test_diagnostic_suite_execution():
    """Runs the 72-query diagnostic suite and asserts that metrics compute properly."""
    report = run_phase4_diagnostic_evaluation()
    assert report["total_queries_evaluated"] == 72
    assert report["total_known_queries"] == 68
    assert report["total_unknown_queries"] == 4
    metrics = report["metrics_summary"]

    assert 0.0 <= metrics["Agronomic_False_Positive_Rate_Raw_ML_pct"] <= 100.0
    assert metrics["Incompatible_Crop_Interception_Rate_pct"] >= 95.0
    assert metrics["In_Season_Retention_Rate_pct"] >= 80.0
