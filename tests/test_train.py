import json

import pytest

from churn_mlops import config, train


@pytest.fixture
def tmp_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(train, "mlflow", None)  # keep tests fast, no tracking side effects
    return tmp_path


def test_gate_fails_below_minimum():
    passed, reason = train.quality_gate({"roc_auc": 0.5}, None)
    assert not passed and "below the minimum" in reason


def test_gate_fails_on_regression():
    passed, reason = train.quality_gate({"roc_auc": 0.72}, {"roc_auc": 0.80})
    assert not passed and "regressed" in reason


def test_gate_passes_when_better():
    passed, _ = train.quality_gate({"roc_auc": 0.82}, {"roc_auc": 0.80})
    assert passed


def test_training_promotes_model_when_gate_passes(tmp_model_dir):
    code = train.main(["--n-samples", "1500", "--n-estimators", "50"])
    assert code == 0
    assert (tmp_model_dir / "model.joblib").exists()
    saved = json.loads((tmp_model_dir / "metrics.json").read_text())
    assert saved["roc_auc"] >= config.MIN_ROC_AUC


def test_training_blocks_bad_model_and_keeps_production(tmp_model_dir):
    # Pretend production is excellent, so a fresh model looks like a regression.
    (tmp_model_dir / "metrics.json").write_text(json.dumps({"roc_auc": 0.99}))
    code = train.main(["--n-samples", "1500", "--n-estimators", "50"])
    assert code == 1
    assert not (tmp_model_dir / "model.joblib").exists()
