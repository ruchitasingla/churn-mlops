"""Train, evaluate, gate, and (if good enough) promote a model.

Exit codes (used by CI):
    0 - model passed the quality gate (and was promoted if it beat production)
    1 - model failed the quality gate -> the pipeline must stop
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import config
from .data import generate_data
from .pipeline import build_pipeline

try:  # MLflow is optional locally so the unit tests stay fast.
    import mlflow
    import mlflow.sklearn
except ImportError:  # pragma: no cover
    mlflow = None


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }


def load_production_metrics() -> dict | None:
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text())
    return None


def quality_gate(new: dict, production: dict | None) -> tuple[bool, str]:
    """Return (passed, reason)."""
    if new["roc_auc"] < config.MIN_ROC_AUC:
        return False, f"ROC AUC {new['roc_auc']:.3f} is below the minimum {config.MIN_ROC_AUC}"
    if production is not None:
        delta = new["roc_auc"] - production["roc_auc"]
        if delta < config.MIN_IMPROVEMENT:
            return False, (
                f"ROC AUC {new['roc_auc']:.3f} regressed vs production "
                f"{production['roc_auc']:.3f} (delta {delta:+.3f})"
            )
    return True, "passed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args(argv)

    df = generate_data(n=args.n_samples)
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )

    params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_samples": args.n_samples,
    }
    model = build_pipeline(args.n_estimators, args.learning_rate, args.max_depth)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    production = load_production_metrics()
    passed, reason = quality_gate(metrics, production)

    # --- Experiment tracking -------------------------------------------------
    if mlflow is not None:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("quality_gate", "passed" if passed else "failed")
            mlflow.set_tag("quality_gate_reason", reason)
            mlflow.sklearn.log_model(model, artifact_path="model")

    print("Metrics:", json.dumps(metrics, indent=2))
    if production:
        print(f"Production ROC AUC: {production['roc_auc']:.3f}")

    if not passed:
        print(f"QUALITY GATE FAILED: {reason}")
        return 1

    # --- Promote -------------------------------------------------------------
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    record = {
        **metrics,
        "params": params,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config.METRICS_PATH.write_text(json.dumps(record, indent=2))
    print(f"QUALITY GATE PASSED. Model promoted to {config.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
