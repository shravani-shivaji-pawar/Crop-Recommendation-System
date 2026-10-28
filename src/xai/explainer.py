"""AgriExplainer: TreeSHAP local feature attribution and contrastive explanation engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import shap

from src.core.config import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    CROP_ID_TO_NAME
)
from src.core.exceptions import PredictionError
from src.ml.predictor import CropPredictor


class AgriExplainer:
    """Computes exact Shapley feature attributions for crop recommendations using TreeSHAP.
    
    Guarantees:
    - Uses actual model outputs and mathematically exact TreeSHAP values.
    - Separates positive contributing factors from limiting constraints.
    - Provides pairwise contrastive explanations (Crop A vs. Crop B).
    - Explicitly distinguishes statistical model attribution from biological causality.
    """

    def __init__(self, predictor: Optional[CropPredictor] = None):
        self.predictor = predictor or CropPredictor()
        # Initialize TreeExplainer on the underlying production RandomForest model
        self.explainer = shap.TreeExplainer(self.predictor.model)
        self.classes = self.predictor.model.classes_

    def explain_recommendation(
        self,
        inputs: Dict[str, Any],
        target_crop_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Computes local SHAP attributions for an input vector.
        
        Args:
            inputs: Dictionary of soil and weather inputs
            target_crop_id: Optional crop ID to explain (defaults to Rank 1 recommended crop)
            
        Returns:
            Dictionary containing base value, SHAP feature attributions, positive drivers,
            limiting constraints, contrastive comparison with runner-up, and scientific disclaimer.
        """
        # 1. Run validated prediction
        pred_res = self.predictor.predict(inputs, top_k=5)
        top_candidates = pred_res["top_candidates"]
        primary = pred_res["primary_recommendation"]

        if target_crop_id is None:
            target_crop_id = primary["crop_id"]

        target_crop_name = CROP_ID_TO_NAME.get(target_crop_id, f"Crop {target_crop_id}")

        # 2. Get scaled vector
        cleaned_vec = self.predictor.preprocessor.validate_input(inputs)
        scaled_vec = self.predictor.preprocessor.transform(cleaned_vec)

        # 3. Find class index in model.classes_
        class_idx_matches = np.where(self.classes == target_crop_id)[0]
        if len(class_idx_matches) == 0:
            raise PredictionError(f"Target crop ID {target_crop_id} not found in model classes.")
        class_idx = int(class_idx_matches[0])

        # 4. Compute TreeSHAP values
        raw_shap = self.explainer.shap_values(scaled_vec)

        # TreeSHAP for multiclass returns (samples, features, classes) or list of (samples, features)
        if isinstance(raw_shap, list):
            sample_shap = raw_shap[class_idx][0]  # shape (7,)
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            sample_shap = raw_shap[0, :, class_idx]  # shape (7,)
        else:
            raise PredictionError("Unexpected SHAP output structure from TreeExplainer.")

        # Expected base value (prior probability)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_val = float(self.explainer.expected_value[class_idx])
        else:
            base_val = float(self.explainer.expected_value)

        # 5. Format feature-wise attributions
        attributions: List[Dict[str, Any]] = []
        positive_drivers: List[Dict[str, Any]] = []
        limiting_constraints: List[Dict[str, Any]] = []

        for idx, col in enumerate(FEATURE_COLUMNS):
            spec = FEATURE_SPECS[col]
            val = float(cleaned_vec[0][idx])
            phi = float(sample_shap[idx])
            phi_pct = round(phi * 100, 2)

            if phi > 0.001:
                impact = "Positive"
                reason = f"{spec['name']} level ({val} {spec['unit']}) strongly aligned with optimal growth envelope for {target_crop_name}."
            elif phi < -0.001:
                impact = "Negative"
                reason = f"{spec['name']} level ({val} {spec['unit']}) acted as a restricting factor for {target_crop_name}."
            else:
                impact = "Neutral"
                reason = f"{spec['name']} ({val} {spec['unit']}) had negligible influence on the recommendation."

            attr_item = {
                "feature": col,
                "name": spec["name"],
                "value": val,
                "unit": spec["unit"],
                "shap_value": round(phi, 4),
                "contribution_pct": phi_pct,
                "impact": impact,
                "reason": reason
            }
            attributions.append(attr_item)

            if phi > 0:
                positive_drivers.append(attr_item)
            elif phi < 0:
                limiting_constraints.append(attr_item)

        # Sort drivers by absolute magnitude
        positive_drivers.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        limiting_constraints.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # 6. Contrastive explanation (Rank 1 vs. Rank 2)
        contrastive = None
        if len(top_candidates) >= 2:
            runner_up = top_candidates[1]
            runner_up_id = runner_up["crop_id"]
            runner_up_name = runner_up["crop_name"]
            runner_idx = int(np.where(self.classes == runner_up_id)[0][0])

            if isinstance(raw_shap, list):
                runner_shap = raw_shap[runner_idx][0]
            else:
                runner_shap = raw_shap[0, :, runner_idx]

            # Delta SHAP: which features favored Target over Runner-Up
            delta_shap = sample_shap - runner_shap
            contrastive_items = []
            for idx, col in enumerate(FEATURE_COLUMNS):
                spec = FEATURE_SPECS[col]
                d_val = float(delta_shap[idx])
                contrastive_items.append({
                    "feature": col,
                    "name": spec["name"],
                    "delta_shap": round(d_val, 4),
                    "favored_crop": target_crop_name if d_val > 0 else runner_up_name,
                    "magnitude_pct": round(abs(d_val) * 100, 2)
                })

            contrastive_items.sort(key=lambda x: abs(x["delta_shap"]), reverse=True)
            top_deciding_factor = contrastive_items[0]

            contrastive = {
                "runner_up_name": runner_up_name,
                "runner_up_score": runner_up["model_confidence_score"],
                "decisive_feature": top_deciding_factor["name"],
                "favored_crop": top_deciding_factor["favored_crop"],
                "explanation": (
                    f"{top_deciding_factor['name']} was the single most decisive factor "
                    f"separating {target_crop_name} from {runner_up_name} (differential contribution: "
                    f"{top_deciding_factor['magnitude_pct']}%)."
                ),
                "feature_deltas": contrastive_items
            }

        return {
            "target_crop": target_crop_name,
            "target_crop_id": target_crop_id,
            "base_value_pct": round(base_val * 100, 2),
            "predicted_score_pct": primary["model_confidence_score"] if target_crop_id == primary["crop_id"] else None,
            "attributions": attributions,
            "positive_drivers": positive_drivers,
            "limiting_constraints": limiting_constraints,
            "contrastive_explanation": contrastive,
            "top_candidates": top_candidates,
            "scientific_disclaimer": (
                "Notice: SHAP values represent local feature attributions within the trained decision tree ensemble. "
                "They explain how the model mapped input features to predicted probabilities, not biological causality or "
                "physical agronomic guarantees."
            )
        }
