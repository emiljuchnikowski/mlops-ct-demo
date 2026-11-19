"""Feature engineering shared by training and serving."""
import pandas as pd

NUMERIC = ["tenure_months", "monthly_fee", "support_tickets", "age"]
PLANS = ["basic", "standard", "premium"]
TARGET = "churn"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns as-is plus a one-hot plan; an unknown plan becomes all zeros."""
    X = df[NUMERIC].astype(float).copy()
    for plan in PLANS:
        X[f"plan_{plan}"] = (df["plan"] == plan).astype(int)
    return X
