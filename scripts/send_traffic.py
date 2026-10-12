"""Send synthetic requests to the API. Use --drift 1.0 to simulate production data drift
and watch the churn_probability histogram shift in Prometheus/Grafana.

Standard library only, so it runs anywhere:
    python scripts/send_traffic.py --n 200 --drift 1.0
"""

import argparse
import json
import random
import urllib.request

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENTS = ["credit-card", "bank-transfer", "e-wallet", "mailed-check"]


def make_customer(drift: float) -> dict:
    tenure = max(1, min(72, int(random.gammavariate(2.0, 15.0 * (1 - 0.5 * drift)))))
    monthly = max(15.0, min(150.0, random.gauss(65 + 25 * drift, 25)))
    return {
        "tenure_months": tenure,
        "monthly_charges": round(monthly, 2),
        "total_charges": round(tenure * monthly, 2),
        "support_tickets": max(0, int(random.gauss(1.2 + 1.5 * drift, 1.2))),
        "avg_monthly_usage_gb": round(max(5.0, random.gauss(120, 50)), 1),
        "contract_type": random.choice(CONTRACTS),
        "payment_method": random.choice(PAYMENTS),
        "has_internet_backup": random.choice(["yes", "no"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/predict")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--drift", type=float, default=0.0)
    args = p.parse_args()

    churn = 0
    for _ in range(args.n):
        req = urllib.request.Request(
            args.url,
            data=json.dumps(make_customer(args.drift)).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            churn += json.load(resp)["will_churn"]
    print(f"Sent {args.n} requests (drift={args.drift}); predicted churn rate {churn / args.n:.1%}")


if __name__ == "__main__":
    main()
