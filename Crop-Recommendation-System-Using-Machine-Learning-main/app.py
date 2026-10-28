"""Flask web application for AgriWise Crop Recommendation & Decision Support System."""

import os
import sys
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.config import FEATURE_SPECS
from src.core.exceptions import ValidationError, AgriWiseError
from src.ml.predictor import CropPredictor
from src.xai.explainer import AgriExplainer
from src.xai.counterfactual import WhatIfSimulator
from src.modules.regional.regional_filter import RegionalDecisionLayer
from src.modules.regional.maharashtra_data import MAHARASHTRA_DISTRICTS

# Initialize Flask app
app = Flask(__name__)

# Initialize robust predictor, XAI, and Regional decision services
try:
    predictor = CropPredictor()
    explainer = AgriExplainer(predictor=predictor)
    simulator = WhatIfSimulator(predictor=predictor)
    regional_layer = RegionalDecisionLayer()
    print("AgriWise CropPredictor, AgriExplainer, WhatIfSimulator, and RegionalDecisionLayer initialized successfully.")
except Exception as e:
    print(f"Warning: Could not initialize ML/XAI/Regional services: {e}")
    predictor = None
    explainer = None
    simulator = None
    regional_layer = None

DISTRICT_NAMES = sorted(list(MAHARASHTRA_DISTRICTS.keys()))
SEASON_NAMES = ["Auto (Calendar-based)", "Kharif", "Rabi", "Summer", "Perennial"]


@app.route('/')
def index():
    return render_template(
        "index.html",
        districts=DISTRICT_NAMES,
        seasons=SEASON_NAMES
    )


@app.route("/predict", methods=['POST'])
def predict():
    if predictor is None:
        return render_template(
            'index.html',
            error_message="Model service is currently unavailable. Please check backend artifacts.",
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
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

        # Generate local SHAP explanation for the primary crop
        explanation = None
        if explainer:
            try:
                explanation = explainer.explain_recommendation(request.form)
            except Exception as ex_err:
                print(f"Warning: SHAP explanation failed: {ex_err}")

        # Regional & Seasonal Decision Synthesis (Non-destructive)
        district = request.form.get("district", "").strip()
        season = request.form.get("season", "").strip()
        season_val = None if (not season or season.startswith("Auto")) else season

        regional_assessment = None
        if regional_layer and district:
            try:
                regional_assessment = regional_layer.assess_recommendation(
                    prediction_result,
                    district=district,
                    season=season_val
                )
            except ValidationError as rve:
                print(f"Regional validation error: {rve}")
            except Exception as re_err:
                print(f"Warning: Regional assessment failed: {re_err}")

        return render_template(
            'index.html',
            result=result_str,
            primary=primary,
            top_candidates=top_candidates,
            disclaimer=disclaimer,
            form_data=request.form,
            validated_inputs=validated_inputs,
            explanation=explanation,
            regional_assessment=regional_assessment,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES,
            selected_district=district,
            selected_season=season
        )

    except ValidationError as ve:
        return render_template(
            'index.html',
            error_message=str(ve),
            form_data=request.form,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
        )
    except Exception as e:
        return render_template(
            'index.html',
            error_message=f"An unexpected error occurred during prediction: {str(e)}",
            form_data=request.form,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
        )


@app.route("/explain", methods=['POST'])
def explain():
    if explainer is None:
        return render_template(
            'index.html',
            error_message="Explainability service is currently unavailable.",
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
        )

    try:
        explanation = explainer.explain_recommendation(request.form)
        return render_template(
            'explain.html',
            explanation=explanation,
            form_data=request.form
        )
    except ValidationError as ve:
        return render_template(
            'index.html',
            error_message=str(ve),
            form_data=request.form,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
        )
    except Exception as e:
        return render_template(
            'index.html',
            error_message=f"Could not compute explanation: {str(e)}",
            form_data=request.form,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
        )


@app.route("/what-if", methods=['GET', 'POST'])
def what_if():
    if simulator is None:
        return render_template(
            'index.html',
            error_message="What-If simulation service is currently unavailable.",
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
        )

    # Default preset values (Rice benchmark profile)
    default_base = {
        "Nitrogen": 90,
        "Phosporus": 42,
        "Potassium": 43,
        "Temperature": 20.87,
        "Humidity": 82.00,
        "Ph": 6.50,
        "Rainfall": 202.93
    }

    if request.method == 'GET':
        # Check if query params were passed from the recommendation page
        base_inputs = {}
        for k in ["Nitrogen", "Phosporus", "Potassium", "Temperature", "Humidity", "Ph", "Rainfall"]:
            val = request.args.get(k)
            base_inputs[k] = float(val) if val is not None else default_base[k]

        district_param = request.args.get("district", "")
        season_param = request.args.get("season", "")

        return render_template(
            'what_if.html',
            base_inputs=base_inputs,
            modified_inputs=base_inputs.copy(),
            feature_specs=FEATURE_SPECS,
            simulation_result=None,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES,
            selected_district=district_param,
            selected_season=season_param
        )

    # POST simulation request
    try:
        base_inputs = {
            "Nitrogen": request.form.get("base_Nitrogen"),
            "Phosporus": request.form.get("base_Phosporus"),
            "Potassium": request.form.get("base_Potassium"),
            "Temperature": request.form.get("base_Temperature"),
            "Humidity": request.form.get("base_Humidity"),
            "Ph": request.form.get("base_Ph"),
            "Rainfall": request.form.get("base_Rainfall")
        }

        mod_inputs = {
            "Nitrogen": request.form.get("mod_Nitrogen"),
            "Phosporus": request.form.get("mod_Phosporus"),
            "Potassium": request.form.get("mod_Potassium"),
            "Temperature": request.form.get("mod_Temperature"),
            "Humidity": request.form.get("mod_Humidity"),
            "Ph": request.form.get("mod_Ph"),
            "Rainfall": request.form.get("mod_Rainfall")
        }

        sim_res = simulator.simulate(base_inputs, mod_inputs)

        district = request.form.get("district", "").strip()
        season = request.form.get("season", "").strip()
        season_val = None if (not season or season.startswith("Auto")) else season

        base_regional = None
        mod_regional = None
        if regional_layer and district:
            try:
                base_regional = regional_layer.assess_recommendation(
                    sim_res["base_prediction"],
                    district=district,
                    season=season_val
                )
                mod_regional = regional_layer.assess_recommendation(
                    sim_res["modified_prediction"],
                    district=district,
                    season=season_val
                )
            except Exception as r_err:
                print(f"Warning: Regional What-If assessment failed: {r_err}")

        return render_template(
            'what_if.html',
            base_inputs=base_inputs,
            modified_inputs=mod_inputs,
            feature_specs=FEATURE_SPECS,
            simulation_result=sim_res,
            base_regional=base_regional,
            mod_regional=mod_regional,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES,
            selected_district=district,
            selected_season=season
        )
    except ValidationError as ve:
        return render_template(
            'what_if.html',
            base_inputs=default_base,
            modified_inputs=default_base.copy(),
            feature_specs=FEATURE_SPECS,
            error_message=str(ve),
            simulation_result=None,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
        )
    except Exception as e:
        return render_template(
            'what_if.html',
            base_inputs=default_base,
            modified_inputs=default_base.copy(),
            feature_specs=FEATURE_SPECS,
            error_message=f"Simulation error: {str(e)}",
            simulation_result=None,
            districts=DISTRICT_NAMES,
            seasons=SEASON_NAMES
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)