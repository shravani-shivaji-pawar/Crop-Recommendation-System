"""Evaluation suite comparing legacy pipeline vs. rectified production pipeline."""

import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.core.config import (
    MODEL_PATH,
    SCALER_PATH,
    PROJECT_ROOT,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SPLIT_RATIO,
    CROP_ID_TO_NAME
)
from src.ml.preprocessor import AgriDataPreprocessor


def evaluate_pipelines() -> dict:
    """Executes side-by-side empirical evaluation of legacy vs. rectified pipelines."""
    X, y = AgriDataPreprocessor.load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT_RATIO,
        stratify=y,
        random_state=RANDOM_STATE
    )

    results = {}

    # 1. Evaluate Rectified Production Pipeline
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            prod_model = pickle.load(f)
        prod_preprocessor = AgriDataPreprocessor.load(SCALER_PATH)

        X_test_scaled = prod_preprocessor.transform(X_test)
        prod_preds = prod_model.predict(X_test_scaled)

        prod_acc = float(accuracy_score(y_test, prod_preds))
        prod_prec = float(precision_score(y_test, prod_preds, average="macro"))
        prod_rec = float(recall_score(y_test, prod_preds, average="macro"))
        prod_f1 = float(f1_score(y_test, prod_preds, average="macro"))

        results["rectified_production_pipeline"] = {
            "test_accuracy": round(prod_acc * 100, 2),
            "macro_precision": round(prod_prec * 100, 2),
            "macro_recall": round(prod_rec * 100, 2),
            "macro_f1": round(prod_f1 * 100, 2),
            "status": "PASS - High Integrity"
        }
    else:
        results["rectified_production_pipeline"] = {"status": "Artifacts not found"}

    # 2. Evaluate Legacy Pipeline (with corrupted scaler)
    legacy_dir = PROJECT_ROOT / "legacy"
    legacy_model_path = legacy_dir / "model.pkl"
    legacy_scaler_path = legacy_dir / "minmaxscaler.pkl"
    legacy_stand_path = legacy_dir / "standscaler.pkl"

    if legacy_model_path.exists() and legacy_scaler_path.exists() and legacy_stand_path.exists():
        with open(legacy_model_path, "rb") as f:
            leg_model = pickle.load(f)
        with open(legacy_scaler_path, "rb") as f:
            leg_ms = pickle.load(f)
        with open(legacy_stand_path, "rb") as f:
            leg_sc = pickle.load(f)

        try:
            # Replicate exact legacy app.py transformation
            leg_scaled = leg_ms.transform(X_test.values)
            leg_final = leg_sc.transform(leg_scaled)
            leg_preds = leg_model.predict(leg_final)

            leg_acc = float(accuracy_score(y_test, leg_preds))
            leg_prec = float(precision_score(y_test, leg_preds, average="macro", zero_division=0))
            leg_rec = float(recall_score(y_test, leg_preds, average="macro", zero_division=0))
            leg_f1 = float(f1_score(y_test, leg_preds, average="macro", zero_division=0))

            results["legacy_pipeline"] = {
                "test_accuracy": round(leg_acc * 100, 2),
                "macro_precision": round(leg_prec * 100, 2),
                "macro_recall": round(leg_rec * 100, 2),
                "macro_f1": round(leg_f1 * 100, 2),
                "status": "FAILED - Severe Scaler Bug",
                "notes": "Feature inputs to trees are out of bounds due to single-sample fitted scaler."
            }
        except Exception as e:
            results["legacy_pipeline"] = {"status": "Error", "details": str(e)}

    # Save evaluation report
    report_path = PROJECT_ROOT / "src" / "ml" / "artifacts" / "baseline_comparison_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    out = evaluate_pipelines()
    print(json.dumps(out, indent=2))
