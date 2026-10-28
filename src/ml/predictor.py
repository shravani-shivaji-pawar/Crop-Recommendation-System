"""CropPredictor: Validated Top-K recommendation engine with calibrated score explanations."""

from pathlib import Path
import pickle
from typing import Dict, Any, List, Optional, Union
import numpy as np

from src.core.config import (
    MODEL_PATH,
    SCALER_PATH,
    CROP_ID_TO_NAME,
    FEATURE_SPECS,
    DEFAULT_TOP_K
)
from src.core.exceptions import (
    ValidationError,
    ModelArtifactError,
    PredictionError
)
from src.ml.preprocessor import AgriDataPreprocessor


class CropPredictor:
    """Inference engine for crop recommendation.
    
    Provides:
    - Robust input sanitization and range checks.
    - Leakage-free feature scaling using trained preprocessor.
    - Top-K crop candidate generation.
    - Transparent distinction between model confidence scores and real-world agronomic suitability.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None
    ):
        self.model_path = Path(model_path or MODEL_PATH)
        self.scaler_path = Path(scaler_path or SCALER_PATH)
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads and verifies model and scaler artifacts."""
        if not self.model_path.exists():
            raise ModelArtifactError(
                f"Model artifact not found at '{self.model_path}'. Run training script to create it."
            )

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise ModelArtifactError(f"Failed to load model artifact at '{self.model_path}': {e}")

        # Load preprocessor using the verified loader
        self.preprocessor = AgriDataPreprocessor.load(self.scaler_path)

    def predict(
        self,
        inputs: Dict[str, Any],
        top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """Validates input, runs inference, and returns ranked Top-K candidate crops.
        
        Args:
            inputs: Dictionary with keys 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
            top_k: Number of candidate crops to return (default: 5)
            
        Returns:
            Dictionary with validated inputs, primary recommendation, top_candidates list,
            and an explicit scientific disclaimer.
        """
        # 1. Input validation & range checking
        cleaned_vector = self.preprocessor.validate_input(inputs)

        # 2. Scale features using the frozen scaler
        try:
            scaled_vector = self.preprocessor.transform(cleaned_vector)
        except Exception as e:
            raise PredictionError(f"Feature transformation failed: {e}")

        # 3. Model inference
        try:
            # Check if model supports predict_proba
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(scaled_vector)[0]
                classes = self.model.classes_
            else:
                pred = self.model.predict(scaled_vector)[0]
                probs = np.zeros(len(CROP_ID_TO_NAME))
                probs[pred - 1] = 1.0
                classes = np.array(list(CROP_ID_TO_NAME.keys()))
        except Exception as e:
            raise PredictionError(f"Inference error in model.predict: {e}")

        # 4. Rank candidates by probability
        top_indices = np.argsort(probs)[::-1][:top_k]

        candidates: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            class_id = int(classes[idx])
            crop_name = CROP_ID_TO_NAME.get(class_id, f"Unknown Crop ({class_id})")
            prob_pct = round(float(probs[idx]) * 100, 2)

            # Categorize confidence band
            if prob_pct >= 50.0:
                band = "High"
                explanation = "Strong statistical alignment with historical benchmark environmental envelope."
            elif prob_pct >= 15.0:
                band = "Moderate"
                explanation = "Viable alternative with partial alignment across current soil and climatic parameters."
            else:
                band = "Low"
                explanation = "Secondary candidate; exhibits marginal suitability under current conditions."

            candidates.append({
                "rank": rank,
                "crop_id": class_id,
                "crop_name": crop_name,
                "model_confidence_score": prob_pct,
                "confidence_level": band,
                "explanation": explanation
            })

        primary = candidates[0] if candidates else None

        # Build clean formatted inputs with units for display
        validated_display_inputs = {}
        for idx, col in enumerate(FEATURE_SPECS.keys()):
            val = float(cleaned_vector[0][idx])
            spec = FEATURE_SPECS[col]
            validated_display_inputs[col] = {
                "name": spec["name"],
                "value": val,
                "unit": spec["unit"]
            }

        return {
            "status": "success",
            "inputs_validated": validated_display_inputs,
            "primary_recommendation": primary,
            "top_candidates": candidates,
            "scientific_disclaimer": (
                "Notice: Model confidence scores indicate statistical proximity to historical benchmark training envelopes. "
                "They must not be conflated with real-world agricultural success, yield, or profitability. Field viability "
                "depends on seasonal sowing calendars, regional water availability, soil depth, and farm management practices."
            )
        }
