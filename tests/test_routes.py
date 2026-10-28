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
    assert b"AgriWise: Crop Recommendation System" in response.data


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
