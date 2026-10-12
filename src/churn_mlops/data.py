"""Synthetic customer-churn data generator.

A seeded generator keeps the repo self-contained (no dataset download needed in CI)
and lets us simulate *data drift* later by changing the `drift` argument.
"""

import numpy as np
import pandas as pd

from .config import RANDOM_STATE

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENTS = ["credit-card", "bank-transfer", "e-wallet", "mailed-check"]


def generate_data(n: int = 5000, seed: int = RANDOM_STATE, drift: float = 0.0) -> pd.DataFrame:
    """Generate a churn dataset.

    Args:
        n: number of rows.
        seed: RNG seed for reproducibility.
        drift: 0.0 = training-like distribution. Higher values shift the inputs
            (newer customers, higher bills, more tickets) to simulate production drift.
    """
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.gamma(2.0, 15.0 * (1 - 0.5 * drift), n), 1, 72).round().astype(int)
    monthly = np.clip(rng.normal(65 + 25 * drift, 25, n), 15, 150).round(2)
    total = (tenure * monthly * rng.uniform(0.9, 1.1, n)).round(2)
    tickets = rng.poisson(1.2 + 1.5 * drift, n)
    usage = np.clip(rng.normal(120, 50, n), 5, 400).round(1)
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(PAYMENTS, n)
    backup = rng.choice(["yes", "no"], n)

    # Ground-truth churn logic (noisy): learnable, but not perfectly.
    logit = (
        -1.0
        - 0.045 * tenure
        + 0.02 * (monthly - 65)
        + 0.35 * tickets
        + np.where(contract == "month-to-month", 1.2, np.where(contract == "one-year", 0.1, -0.8))
        + np.where(payment == "e-wallet", 0.3, 0.0)
        + np.where(backup == "no", 0.25, 0.0)
        - 0.003 * (usage - 120)
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = (rng.uniform(size=n) < prob).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_tickets": tickets,
            "avg_monthly_usage_gb": usage,
            "contract_type": contract,
            "payment_method": payment,
            "has_internet_backup": backup,
            "churned": churned,
        }
    )


if __name__ == "__main__":
    df = generate_data()
    out = "data/customers.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out} (churn rate {df['churned'].mean():.1%})")
