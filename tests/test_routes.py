"""Integration tests for Flask application routes including XAI and What-If studio."""

import sys
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = WORKSPACE_ROOT / "Crop-Recommendation-System-Using-Machine-Learning-main"

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app import app


@pytest.fixture
def client():
    """Provides a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verifies that the home page loads with 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"AgroVista AI" in response.data


def test_predict_route_success_with_shap(client):
    """Verifies successful POST prediction including TreeSHAP feature attributions."""
    data = {
        "Nitrogen": "90",
        "Phosporus": "42",
        "Potassium": "43",
        "Temperature": "20.87",
        "Humidity": "82.0",
        "Ph": "6.5",
        "Rainfall": "202.93"
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Rice is the best crop to be cultivated" in html
    assert "Top Candidate Crops" in html
    assert "Rank #1 Match" in html
    assert "TreeSHAP Feature Attributions" in html
    assert "Pairwise Contrastive Analysis" in html
    assert "Academic &amp; Scientific Note" in html or "Academic & Scientific Note" in html


def test_predict_route_invalid_bounds(client):
    """Verifies that out-of-bound inputs render error banner rather than 500 error."""
    bad_data = {
        "Nitrogen": "90",
        "Phosporus": "42",
        "Potassium": "43",
        "Temperature": "20.87",
        "Humidity": "82.0",
        "Ph": "-4.0",  # Invalid pH
        "Rainfall": "202.93"
    }
    response = client.post("/predict", data=bad_data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Input Validation Error" in html
    assert "outside acceptable range" in html


def test_predict_route_non_numeric(client):
    """Verifies that non-numeric inputs render error banner."""
    bad_data = {
        "Nitrogen": "abc",
        "Phosporus": "42",
        "Potassium": "43",
        "Temperature": "20.87",
        "Humidity": "82.0",
        "Ph": "6.5",
        "Rainfall": "202.93"
    }
    response = client.post("/predict", data=bad_data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Input Validation Error" in html
    assert "Invalid non-numeric value" in html


def test_whatif_get_route(client):
    """Verifies that the What-If Studio loads with 200 OK and default form values."""
    response = client.get("/what-if")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "What-If Counterfactual Simulation Studio" in html
    assert "Scenario A" in html
    assert "Scenario B" in html


def test_whatif_post_route_success(client):
    """Verifies side-by-side simulation comparing baseline against reduced rainfall."""
    sim_data = {
        "base_Nitrogen": "90",
        "base_Phosporus": "42",
        "base_Potassium": "43",
        "base_Temperature": "20.87",
        "base_Humidity": "82.00",
        "base_Ph": "6.50",
        "base_Rainfall": "202.93",
        "mod_Nitrogen": "90",
        "mod_Phosporus": "42",
        "mod_Potassium": "43",
        "mod_Temperature": "20.87",
        "mod_Humidity": "82.00",
        "mod_Ph": "6.50",
        "mod_Rainfall": "50.00"  # Reduced rainfall
    }
    response = client.post("/what-if", data=sim_data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Environmental Parameter Variations" in html
    assert "Top-5 Crop Candidate Rank &amp; Score Shifts" in html or "Top-5 Crop Candidate Rank & Score Shifts" in html
    assert "Agronomic &amp; Sensitivity Guardrail" in html or "Agronomic & Sensitivity Guardrail" in html


def test_whatif_post_route_invalid_bounds(client):
    """Verifies that invalid counterfactual inputs render error banner in What-If studio."""
    bad_sim_data = {
        "base_Nitrogen": "90", "base_Phosporus": "42", "base_Potassium": "43",
        "base_Temperature": "20.87", "base_Humidity": "82.00", "base_Ph": "6.50", "base_Rainfall": "202.93",
        "mod_Nitrogen": "90", "mod_Phosporus": "42", "mod_Potassium": "43",
        "mod_Temperature": "20.87", "mod_Humidity": "82.00",
        "mod_Ph": "15.00",  # Out of range pH
        "mod_Rainfall": "202.93"
    }
    response = client.post("/what-if", data=bad_sim_data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Simulation Error" in html
    assert "outside acceptable range" in html


def test_predict_route_with_maharashtra_district(client):
    """Verifies that submitting with district and season renders the regional decision layer."""
    data = {
        "Nitrogen": "90",
        "Phosporus": "42",
        "Potassium": "43",
        "Temperature": "20.87",
        "Humidity": "82.0",
        "Ph": "6.5",
        "Rainfall": "202.93",
        "district": "Ratnagiri",
        "season": "Kharif"
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Regional &amp; Seasonal Viability Assessment" in html or "Regional & Seasonal Viability Assessment" in html
    assert "Ratnagiri District" in html
    assert "Kharif Season" in html
    assert "MH-Z1: South Konkan Coastal Zone" in html
    assert "Cultivable &amp; In-Season" in html or "Cultivable & In-Season" in html
    assert "Non-Destructive Regional Feasibility Analysis" in html


def test_predict_route_with_apple_incompatible_flagged(client):
    """Verifies that climatically incompatible candidates receive red badge while raw ML is preserved."""
    apple_data = {
        "Nitrogen": "20",
        "Phosporus": "135",
        "Potassium": "198",
        "Temperature": "22.0",
        "Humidity": "92.0",
        "Ph": "5.8",
        "Rainfall": "110.0",
        "district": "Solapur",
        "season": "Kharif"
    }
    response = client.post("/predict", data=apple_data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Solapur District" in html
    assert "Incompatible / Unsuited" in html
    assert "Solapur" in html
    # Raw ML scores should still be visible in the candidate table
    assert "Top Candidate Crops" in html


def test_whatif_post_route_with_regional_context(client):
    """Verifies that What-If simulation with district renders regional viability shift."""
    sim_data = {
        "base_Nitrogen": "90",
        "base_Phosporus": "42",
        "base_Potassium": "43",
        "base_Temperature": "20.87",
        "base_Humidity": "82.00",
        "base_Ph": "6.50",
        "base_Rainfall": "202.93",
        "mod_Nitrogen": "90",
        "mod_Phosporus": "42",
        "mod_Potassium": "43",
        "mod_Temperature": "20.87",
        "mod_Humidity": "82.00",
        "mod_Ph": "6.50",
        "mod_Rainfall": "50.00",
        "district": "Kolhapur",
        "season": "Kharif"
    }
    response = client.post("/what-if", data=sim_data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Regional Viability Response" in html
    assert "Kolhapur" in html


def test_ui_static_css_and_presets_rendered(client):
    """Verifies that the upgraded Phase 5 UI links custom CSS and provides quick presets."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "static/css/style.css" in html
    assert "Quick Test Presets" in html
    assert "Rice (Konkan Kharif)" in html
    assert "Phase 5: Interactive Dashboard" in html
    assert "Soil Macronutrients &amp; Chemistry" in html or "Soil Macronutrients & Chemistry" in html


def test_whatif_ui_elements_rendered(client):
    """Verifies that the upgraded What-If Studio links custom CSS and interactive shifts."""
    response = client.get("/what-if")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "static/css/style.css" in html
    assert "Quick Perturbation Scenarios" in html
    assert "Severe Drought" in html
    assert "Heatwave" in html
    assert "Phase 5: Interactive Dashboard" in html


def test_predict_route_renders_secondary_data_insights(client):
    """Verifies that submitting prediction with district renders IMD rainfall departure and DES badges."""
    data = {
        "Nitrogen": "90",
        "Phosporus": "42",
        "Potassium": "43",
        "Temperature": "20.87",
        "Humidity": "82.0",
        "Ph": "6.5",
        "Rainfall": "202.93",
        "district": "Ratnagiri",
        "season": "Kharif"
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "IMD Pune Rainfall Departure Analysis" in html
    assert "DES Historical Record" in html or "DES Record" in html



