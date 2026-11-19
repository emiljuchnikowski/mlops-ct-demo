"""Generate the synthetic churn datasets used in the demo (deterministic)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
PLANS = ["basic", "standard", "premium"]


def customers(n: int, seed: int, first_id: int, tickets: float = 0.55, tenure: float = -0.06,
              basic: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = rng.choice(PLANS, size=n, p=[0.5, 0.35, 0.15])
    base_fee = pd.Series(plan).map({"basic": 39.0, "standard": 69.0, "premium": 129.0}).to_numpy()
    df = pd.DataFrame({
        "customer_id": np.arange(first_id, first_id + n),
        "tenure_months": rng.integers(0, 73, size=n),
        "monthly_fee": np.round(base_fee + rng.normal(0, 8, size=n), 2),
        "plan": plan,
        "support_tickets": rng.poisson(1.2, size=n),
        "age": rng.integers(18, 81, size=n),
    })
    logit = (
        0.4
        + tenure * df.tenure_months
        + 0.02 * (df.monthly_fee - 60)
        + tickets * df.support_tickets
        + basic * (df.plan == "basic")
        - 0.015 * (df.age - 40)
    )
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def main() -> None:
    (DATA / "incoming").mkdir(parents=True, exist_ok=True)
    customers(3000, seed=1, first_id=1).to_csv(DATA / "reference.csv", index=False)
    customers(1500, seed=2, first_id=100_001).to_csv(DATA / "holdout.csv", index=False)

    # A healthy monthly batch: same distribution as the reference data.
    customers(1500, seed=3, first_id=200_001).to_csv(DATA / "incoming" / "batch_ok.csv", index=False)

    # Upstream system changed: fee sent in grosze, a new plan code, missing tenure.
    broken = customers(1500, seed=4, first_id=300_001)
    broken["monthly_fee"] = (broken.monthly_fee * 100).round(0)
    rng = np.random.default_rng(40)
    broken.loc[rng.random(len(broken)) < 0.06, "plan"] = "PREMIUM_PLUS"
    broken["tenure_months"] = broken.tenure_months.astype("Int64")
    broken.loc[rng.random(len(broken)) < 0.08, "tenure_months"] = pd.NA
    broken.to_csv(DATA / "incoming" / "batch_broken.csv", index=False)

    # Valid schema and ranges, but a misaligned join gave customers someone else's label.
    shuffled = customers(3000, seed=5, first_id=400_001)
    shuffled["churn"] = np.random.default_rng(50).permutation(shuffled.churn.to_numpy())
    shuffled.to_csv(DATA / "incoming" / "batch_shuffled_labels.csv", index=False)

    # Concept drift in November: after a price change basic-plan customers stay, and every support ticket
    # weighs twice as much. Labelled November data to retrain on, and a later labelled window to evaluate on.
    november = dict(tickets=1.1, tenure=-0.03, basic=-1.0)
    customers(1500, seed=6, first_id=500_001, **november).to_csv(DATA / "incoming" / "batch_nov.csv", index=False)
    customers(1500, seed=7, first_id=600_001, **november).to_csv(DATA / "holdout_nov.csv", index=False)


if __name__ == "__main__":
    main()
