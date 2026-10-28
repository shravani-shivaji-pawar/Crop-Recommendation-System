"""Training harness for AgriWise recommendation models."""

import json
import time
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.core.config import (
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SPLIT_RATIO,
    CROP_ID_TO_NAME
)
from src.ml.preprocessor import AgriDataPreprocessor


def train_production_model(save_artifacts: bool = True) -> dict:
    """Trains a verified production model using leakage-free scaling and reproducible splits.
    
    Returns:
        metrics (dict): Performance metrics on held-out test split.
    """
    t0 = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    X, y = AgriDataPreprocessor.load_dataset()

    # 2. Reproducible stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT_RATIO,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 3. Fit scaler ONLY on training data
    preprocessor = AgriDataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 4. Train RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # 5. Evaluate on held-out test set
    y_pred = clf.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro"))
    rec = float(recall_score(y_test, y_pred, average="macro"))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    training_duration_ms = (time.time() - t0) * 1000

    metrics = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": len(CROP_ID_TO_NAME),
        "test_accuracy": acc,
        "test_precision_macro": prec,
        "test_recall_macro": rec,
        "test_f1_macro": f1,
        "training_duration_ms": round(training_duration_ms, 2),
        "trained_at": datetime.now().isoformat(),
        "features": FEATURE_COLUMNS
    }

    # 6. Save artifacts if requested
    if save_artifacts:
        preprocessor.save(SCALER_PATH)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        with open(METADATA_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    results = train_production_model(save_artifacts=True)
    print("Production Model Training Completed:")
    print(json.dumps(results, indent=2))
