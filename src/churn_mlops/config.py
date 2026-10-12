"""Central configuration. Everything can be overridden with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "churn-prediction")

# Quality gate: the pipeline fails if the new model is worse than these bars.
MIN_ROC_AUC = float(os.getenv("MIN_ROC_AUC", "0.70"))
# A new model must not be worse than the current production model by more than this margin.
# (Slightly negative tolerates run-to-run noise.)
MIN_IMPROVEMENT = float(os.getenv("MIN_IMPROVEMENT", "-0.005"))

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "avg_monthly_usage_gb",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "has_internet_backup"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churned"
