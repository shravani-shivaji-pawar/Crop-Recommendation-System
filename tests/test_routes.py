"""Integration tests for Flask application routes."""

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


def test_predict_route_success(client):
    """Verifies successful POST prediction for typical Rice inputs."""
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
