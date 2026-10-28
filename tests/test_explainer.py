"""Unit tests for AgriExplainer (TreeSHAP integration)."""

import pytest
import numpy as np
from src.core.config import FEATURE_COLUMNS
from src.xai.explainer import AgriExplainer


@pytest.fixture
def explainer():
    """Provides an AgriExplainer instance."""
    return AgriExplainer()


def test_explainer_initialization(explainer):
    """Verifies that AgriExplainer and TreeExplainer load cleanly."""
    assert explainer.predictor is not None
    assert explainer.explainer is not None


def test_explain_recommendation_rice(explainer):
    """Tests TreeSHAP explanation for typical Rice inputs."""
    rice_inputs = {
        "Nitrogen": 90,
        "Phosporus": 42,
        "Potassium": 43,
        "Temperature": 20.87,
        "Humidity": 82.0,
        "Ph": 6.5,
        "Rainfall": 202.93
    }
    explanation = explainer.explain_recommendation(rice_inputs)

    assert explanation["target_crop"] == "Rice"
    assert len(explanation["attributions"]) == len(FEATURE_COLUMNS)
    assert len(explanation["positive_drivers"]) > 0
    assert "scientific_disclaimer" in explanation

    # Check that attributions have all required metadata fields
    for attr in explanation["attributions"]:
        assert attr["feature"] in FEATURE_COLUMNS
        assert isinstance(attr["name"], str)
        assert isinstance(attr["shap_value"], float)
        assert attr["impact"] in ["Positive", "Negative", "Neutral"]
        assert len(attr["reason"]) > 0

    # Verify mathematical Shapley additivity
    base_val = explanation["base_value_pct"] / 100.0
    sum_shap = sum(a["shap_value"] for a in explanation["attributions"])
    reconstructed_prob = base_val + sum_shap
    model_score = explanation["predicted_score_pct"] / 100.0

    assert np.isclose(reconstructed_prob, model_score, atol=1e-3), (
        f"Shapley efficiency violated: base ({base_val}) + sum_shap ({sum_shap}) != model ({model_score})"
    )


def test_contrastive_explanation(explainer):
    """Verifies pairwise contrastive comparison between Rank 1 and Rank 2."""
    sample_inputs = {
        "Nitrogen": 71,
        "Phosporus": 54,
        "Potassium": 16,
        "Temperature": 22.61,
        "Humidity": 63.70,
        "Ph": 5.72,
        "Rainfall": 87.76
    }
    explanation = explainer.explain_recommendation(sample_inputs)
    contrastive = explanation["contrastive_explanation"]

    assert contrastive is not None
    assert "runner_up_name" in contrastive
    assert "decisive_feature" in contrastive
    assert "favored_crop" in contrastive
    assert len(contrastive["feature_deltas"]) == len(FEATURE_COLUMNS)
