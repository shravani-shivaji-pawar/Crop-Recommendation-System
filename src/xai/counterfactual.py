"""WhatIfSimulator: Interactive counterfactual simulation and input perturbation studio."""

from typing import Dict, Any, List, Optional
import numpy as np

from src.core.config import (
    FEATURE_COLUMNS,
    FEATURE_SPECS
)
from src.ml.predictor import CropPredictor


class WhatIfSimulator:
    """Simulates environmental and soil perturbations to explore sensitivity and rank shifts.
    
    Guarantees:
    - Strict validation of both baseline and counterfactual inputs against agronomic bounds.
    - Side-by-side comparison of Top-5 candidate rank shifts.
    - Explicit distinction between statistical model sensitivity and actionable field interventions.
    """

    def __init__(self, predictor: Optional[CropPredictor] = None):
        self.predictor = predictor or CropPredictor()

    def simulate(
        self,
        base_inputs: Dict[str, Any],
        modified_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs side-by-side simulation comparing baseline against modified conditions.
        
        Args:
            base_inputs: Original input dictionary
            modified_inputs: Modified/perturbed input dictionary
            
        Returns:
            Dictionary with base and modified predictions, parameter deltas,
            rank shift analysis, and simulation disclaimers.
        """
        # 1. Run predictions for both scenarios (validates inputs internally)
        base_res = self.predictor.predict(base_inputs, top_k=5)
        mod_res = self.predictor.predict(modified_inputs, top_k=5)

        base_primary = base_res["primary_recommendation"]
        mod_primary = mod_res["primary_recommendation"]

        # 2. Compute parameter deltas
        parameter_deltas: List[Dict[str, Any]] = []
        has_changes = False

        for col in FEATURE_COLUMNS:
            spec = FEATURE_SPECS[col]
            b_val = float(base_res["inputs_validated"][col]["value"])
            m_val = float(mod_res["inputs_validated"][col]["value"])
            delta = round(m_val - b_val, 2)
            
            if abs(b_val) > 1e-5:
                pct_change = round((delta / b_val) * 100, 1)
            else:
                pct_change = 0.0

            if abs(delta) > 1e-4:
                has_changes = True

            parameter_deltas.append({
                "feature": col,
                "name": spec["name"],
                "unit": spec["unit"],
                "base_value": b_val,
                "modified_value": m_val,
                "delta": delta,
                "pct_change": pct_change,
                "direction": "Increased" if delta > 0 else ("Decreased" if delta < 0 else "Unchanged")
            })

        # 3. Track rank and score shifts across Top-5 candidate pools
        base_cand_map = {c["crop_name"]: c for c in base_res["top_candidates"]}
        mod_cand_map = {c["crop_name"]: c for c in mod_res["top_candidates"]}

        all_crops = list(dict.fromkeys(
            [c["crop_name"] for c in base_res["top_candidates"]] +
            [c["crop_name"] for c in mod_res["top_candidates"]]
        ))

        rank_shifts: List[Dict[str, Any]] = []
        for crop in all_crops:
            b_cand = base_cand_map.get(crop)
            m_cand = mod_cand_map.get(crop)

            b_rank = b_cand["rank"] if b_cand else ">5"
            m_rank = m_cand["rank"] if m_cand else ">5"

            b_score = b_cand["model_confidence_score"] if b_cand else 0.0
            m_score = m_cand["model_confidence_score"] if m_cand else 0.0
            score_delta = round(m_score - b_score, 2)

            if isinstance(b_rank, int) and isinstance(m_rank, int):
                if m_rank < b_rank:
                    movement = f"Gained {b_rank - m_rank} rank(s) ↑"
                elif m_rank > b_rank:
                    movement = f"Dropped {m_rank - b_rank} rank(s) ↓"
                else:
                    movement = "Unchanged rank"
            elif isinstance(b_rank, int) and not isinstance(m_rank, int):
                movement = "Exited Top 5 ↓"
            else:
                movement = "Entered Top 5 ↑"

            rank_shifts.append({
                "crop_name": crop,
                "base_rank": b_rank,
                "modified_rank": m_rank,
                "base_score": b_score,
                "modified_score": m_score,
                "score_delta": score_delta,
                "movement": movement
            })

        # 4. Generate structured natural summary
        primary_shifted = base_primary["crop_name"] != mod_primary["crop_name"]
        if primary_shifted:
            summary = (
                f"Recommendation changed: The primary recommended crop shifted from "
                f"**{base_primary['crop_name']}** ({base_primary['model_confidence_score']}%) to "
                f"**{mod_primary['crop_name']}** ({mod_primary['model_confidence_score']}%)."
            )
        else:
            score_diff = round(mod_primary['model_confidence_score'] - base_primary['model_confidence_score'], 2)
            direction_str = "increased" if score_diff > 0 else "decreased"
            summary = (
                f"Recommendation retained: **{base_primary['crop_name']}** remains the top recommendation. "
                f"Its confidence score {direction_str} by {abs(score_diff)}% "
                f"({base_primary['model_confidence_score']}% → {mod_primary['model_confidence_score']}%)."
            )

        return {
            "status": "success",
            "has_changes": has_changes,
            "primary_shifted": primary_shifted,
            "summary": summary,
            "parameter_deltas": parameter_deltas,
            "base_prediction": base_res,
            "modified_prediction": mod_res,
            "rank_shifts": rank_shifts,
            "scientific_disclaimer": (
                "Simulation Notice: What-If counterfactual analysis is a sensitivity evaluation of the predictive model. "
                "It reveals how the model's decision boundaries respond to altered numerical variables. In actual agriculture, "
                "altering soil properties or seasonal water availability involves complex bio-chemical dynamics and does not "
                "guarantee proportional real-world outcomes."
            )
        }
