"""Unit tests for WhatIfSimulator."""

import pytest
from src.core.exceptions import ValidationError
from src.xai.counterfactual import WhatIfSimulator


@pytest.fixture
def simulator():
    """Provides a WhatIfSimulator instance."""
    return WhatIfSimulator()


def test_whatif_initialization(simulator):
    """Verifies that WhatIfSimulator loads properly."""
    assert simulator.predictor is not None


def test_whatif_identical_inputs(simulator):
    """Tests simulation when modified inputs equal baseline inputs."""
    base = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.0, "Ph": 6.5, "Rainfall": 202.93
    }
    res = simulator.simulate(base, base)

    assert res["status"] == "success"
    assert res["has_changes"] is False
    assert res["primary_shifted"] is False
    assert "remains the top recommendation" in res["summary"]


def test_whatif_rainfall_reduction_shift(simulator):
    """Tests that drastically reducing rainfall shifts prediction from Rice to a lower-water crop."""
    base = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.0, "Ph": 6.5, "Rainfall": 202.93  # Rice
    }
    modified = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.0, "Ph": 6.5, "Rainfall": 40.0   # Drastic water drop
    }
    res = simulator.simulate(base, modified)

    assert res["status"] == "success"
    assert res["has_changes"] is True
    assert "scientific_disclaimer" in res
    assert len(res["parameter_deltas"]) == 7
    assert len(res["rank_shifts"]) >= 5

    # Check rainfall delta
    rain_delta = next(d for d in res["parameter_deltas"] if d["feature"] == "rainfall")
    assert rain_delta["delta"] == -162.93
    assert rain_delta["direction"] == "Decreased"


def test_whatif_invalid_modified_input(simulator):
    """Ensures out-of-range modified inputs are rejected with ValidationError."""
    base = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.0, "Ph": 6.5, "Rainfall": 202.93
    }
    bad_modified = {
        "Nitrogen": 90, "Phosporus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.0, "Ph": 14.5, "Rainfall": 202.93  # pH out of range
    }
    with pytest.raises(ValidationError, match="outside acceptable range"):
        simulator.simulate(base, bad_modified)
