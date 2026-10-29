"""Tests for Phase 6: Verified Secondary Datasets Integration (DES Crop Production & IMD Climatology)."""

import pytest
import pandas as pd
from pathlib import Path

from src.core.config import PROJECT_ROOT
from src.modules.regional.secondary_data import (
    HistoricalProductionRepository,
    ClimatologyRepository,
    COTTON_BALES_TO_TONNES
)
from src.modules.regional.regional_filter import RegionalDecisionLayer
from src.ml.predictor import CropPredictor


@pytest.fixture
def des_repo():
    return HistoricalProductionRepository()


@pytest.fixture
def climatology_repo():
    return ClimatologyRepository()


def test_des_dataset_file_and_schema():
    """Validates that the raw DES crop production dataset is downloaded, non-empty, and has expected schema."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "maharashtra_crop_production_des_raw.csv"
    assert csv_path.exists(), "Raw DES crop production CSV must exist in data/raw/"
    
    df = pd.read_csv(csv_path)
    assert len(df) >= 12000, f"Expected at least 12,000 Maharashtra rows, got {len(df)}"
    expected_cols = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area', 'Production']
    for col in expected_cols:
        assert col in df.columns, f"Missing expected DES column: {col}"

    assert (df['State_Name'].str.strip() == 'Maharashtra').all()


def test_imd_climatology_dataset_file_and_schema():
    """Validates that the IMD rainfall normals dataset exists and covers all 36 Maharashtra districts."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "maharashtra_district_rainfall_normals_imd.csv"
    assert csv_path.exists(), "IMD rainfall normals CSV must exist in data/raw/"

    df = pd.read_csv(csv_path)
    assert len(df) == 36, f"Expected exactly 36 districts, got {len(df)}"
    assert 'annual_normal_mm' in df.columns
    assert 'sw_monsoon_jjas_normal_mm' in df.columns
    assert (df['annual_normal_mm'] > 400).all()


def test_cotton_bale_unit_conversion(des_repo):
    """Verifies that cotton(lint) is converted from bales to metric tonnes using 0.170 factor."""
    assert COTTON_BALES_TO_TONNES == 0.170
    cotton_hist = des_repo.get_crop_history("Cotton", "Yavatmal")
    assert cotton_hist["has_records"] is True
    assert cotton_hist["commercial_scale"] == "Major Commercial Backbone"
    assert cotton_hist["mean_area_ha"] > 100000.0  # Yavatmal has massive cotton acreage (>300,000 ha)
    assert cotton_hist["mean_production_tonnes"] is not None


def test_rice_historical_presence_in_konkan(des_repo):
    """Verifies that Rice has verified high commercial presence in Konkan / Wainganga districts."""
    rice_hist = des_repo.get_crop_history("Rice", "Bhandara")
    assert rice_hist["has_records"] is True
    assert rice_hist["mean_area_ha"] > 50000.0
    assert "Kharif" in rice_hist["recorded_seasons"]


def test_missing_crop_safe_handling(des_repo):
    """Verifies that out-of-region or unrecorded crops (Apple) return safe, un-fabricated records."""
    apple_hist = des_repo.get_crop_history("Apple", "Solapur")
    assert apple_hist["has_records"] is False
    assert apple_hist["mean_area_ha"] is None
    assert "No historical DES" in apple_hist["summary"] or "Zero commercial" in apple_hist["summary"]


def test_unmatched_district_safe_handling(des_repo):
    """Verifies that an unknown or non-Maharashtra district returns a graceful un-fabricated response."""
    hist = des_repo.get_crop_history("Rice", "NonExistentDistrict")
    assert hist["has_records"] is False
    assert hist["years_surveyed"] == 0


def test_imd_rainfall_departure_regimes(climatology_repo):
    """Verifies that rainfall departure calculations map correctly to official IMD regimes."""
    # Ahmednagar normal is 579.5 mm
    # 1. Deficient scenario: 200 mm (approx -65% departure)
    res_def = climatology_repo.evaluate_rainfall_departure("Ahmednagar", observed_rainfall_mm=200.0)
    assert res_def["has_climatology"] is True
    assert res_def["regime"] in ["Deficient", "Large Deficient"]
    assert res_def["departure_pct"] < -20.0

    # 2. Normal scenario: 580 mm (approx 0% departure)
    res_norm = climatology_repo.evaluate_rainfall_departure("Ahmednagar", observed_rainfall_mm=580.0)
    assert res_norm["regime"] == "Normal"
    assert -19.0 <= res_norm["departure_pct"] <= 19.0

    # 3. Excess scenario: 900 mm (approx +55% departure)
    res_exc = climatology_repo.evaluate_rainfall_departure("Ahmednagar", observed_rainfall_mm=900.0)
    assert res_exc["regime"] in ["Excess", "Large Excess"]
    assert res_exc["departure_pct"] >= 20.0


def test_regional_layer_secondary_enrichment():
    """Verifies that RegionalDecisionLayer outputs both DES historical summaries and IMD departure evaluations."""
    predictor = CropPredictor()
    regional_layer = RegionalDecisionLayer()

    rice_inputs = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.00, "Ph": 6.50, "Rainfall": 202.93
    }
    pred_res = predictor.predict(rice_inputs, top_k=5)
    reg_eval = regional_layer.assess_recommendation(pred_res, district="Ratnagiri", season="Kharif")

    # Climatology comparison check
    assert "climatology_comparison" in reg_eval
    clim = reg_eval["climatology_comparison"]
    assert clim is not None
    assert clim["district"] == "Ratnagiri"
    assert "departure_pct" in clim

    # Candidate historical summary check
    top_cand_eval = reg_eval["regional_assessments"][0]
    assert "historical_production_summary" in top_cand_eval
    assert "evidence_status" in top_cand_eval["historical_production_summary"]
