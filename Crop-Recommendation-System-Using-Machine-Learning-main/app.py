"""Flask web application for AgriWise Crop Recommendation System."""

import os
import sys
from pathlib import Path
from flask import Flask, request, render_template

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.exceptions import ValidationError, AgriWiseError
from src.ml.predictor import CropPredictor

# Initialize Flask app
app = Flask(__name__)

# Initialize robust predictor instance (loads verified production artifacts)
try:
    predictor = CropPredictor()
    print("AgriWise CropPredictor initialized successfully.")
except Exception as e:
    print(f"Warning: Could not initialize CropPredictor: {e}")
    predictor = None


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    if predictor is None:
        return render_template(
            'index.html',
            error_message="Model service is currently unavailable. Please check backend artifacts."
        )

    try:
        # Pass request form dictionary to validated predictor
        prediction_result = predictor.predict(request.form, top_k=5)

        primary = prediction_result["primary_recommendation"]
        top_candidates = prediction_result["top_candidates"]
        disclaimer = prediction_result["scientific_disclaimer"]
        validated_inputs = prediction_result["inputs_validated"]

        # Legacy backward-compatible string
        result_str = f"{primary['crop_name']} is the best crop to be cultivated right there"

        return render_template(
            'index.html',
            result=result_str,
            primary=primary,
            top_candidates=top_candidates,
            disclaimer=disclaimer,
            form_data=request.form,
            validated_inputs=validated_inputs
        )

    except ValidationError as ve:
        return render_template(
            'index.html',
            error_message=str(ve),
            form_data=request.form
        )
    except Exception as e:
        return render_template(
            'index.html',
            error_message=f"An unexpected error occurred during prediction: {str(e)}",
            form_data=request.form
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)