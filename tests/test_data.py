from churn_mlops.config import FEATURES, TARGET
from churn_mlops.data import generate_data


def test_generate_data_shape_and_columns():
    df = generate_data(n=500)
    assert len(df) == 500
    assert set(FEATURES + [TARGET]) == set(df.columns)


def test_generate_data_is_reproducible():
    assert generate_data(n=200, seed=1).equals(generate_data(n=200, seed=1))


def test_drift_shifts_distribution():
    base = generate_data(n=3000, drift=0.0)
    drifted = generate_data(n=3000, drift=1.0)
    assert drifted["monthly_charges"].mean() > base["monthly_charges"].mean() + 10
    assert drifted["tenure_months"].mean() < base["tenure_months"].mean()
