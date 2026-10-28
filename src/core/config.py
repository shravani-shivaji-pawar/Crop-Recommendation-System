"""Configuration constants, paths, and feature bounds for AgriWise."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Crop_recommendation.csv"
LEGACY_DATA_PATH = PROJECT_ROOT / "legacy" / "Crop_recommendation.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# Core Features
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Agronomic Valid Input Ranges [min, max] and human labels
FEATURE_SPECS = {
    "N": {
        "name": "Nitrogen",
        "min": 0.0,
        "max": 200.0,
        "unit": "ratio / kg/ha",
        "description": "Ratio of Nitrogen content in soil",
        "typical_range": [0.0, 140.0],
    },
    "P": {
        "name": "Phosphorus",
        "min": 0.0,
        "max": 200.0,
        "unit": "ratio / kg/ha",
        "description": "Ratio of Phosphorus content in soil",
        "typical_range": [5.0, 145.0],
    },
    "K": {
        "name": "Potassium",
        "min": 0.0,
        "max": 300.0,
        "unit": "ratio / kg/ha",
        "description": "Ratio of Potassium content in soil",
        "typical_range": [5.0, 205.0],
    },
    "temperature": {
        "name": "Temperature",
        "min": 0.0,
        "max": 60.0,
        "unit": "°C",
        "description": "Ambient temperature in degrees Celsius",
        "typical_range": [8.0, 45.0],
    },
    "humidity": {
        "name": "Humidity",
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "description": "Relative humidity in percentage",
        "typical_range": [14.0, 100.0],
    },
    "ph": {
        "name": "Soil pH",
        "min": 3.0,
        "max": 11.0,
        "unit": "pH scale (0-14)",
        "description": "pH value of the soil solution",
        "typical_range": [3.5, 10.0],
    },
    "rainfall": {
        "name": "Rainfall",
        "min": 0.0,
        "max": 1000.0,
        "unit": "mm",
        "description": "Seasonal or annual precipitation in mm",
        "typical_range": [20.0, 300.0],
    },
}

# Canonical Crop Dictionary (1-indexed for legacy compatibility and string-mapping)
CROP_ID_TO_NAME = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

CROP_NAME_TO_ID = {name.lower(): cid for cid, name in CROP_ID_TO_NAME.items()}

# Default Top-K Recommendation Count
DEFAULT_TOP_K = 5
RANDOM_STATE = 42
TEST_SPLIT_RATIO = 0.2
