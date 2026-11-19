import pandas as pd

from mlpipe.features import PLANS, build_features


def test_plan_is_one_hot():
    df = pd.DataFrame({"tenure_months": [1], "monthly_fee": [39.0], "support_tickets": [0], "age": [30],
                       "plan": ["standard"]})
    X = build_features(df)
    assert X[[f"plan_{p}" for p in PLANS]].iloc[0].tolist() == [0, 1, 0]


def test_unknown_plan_maps_to_zeros():
    df = pd.DataFrame({"tenure_months": [1], "monthly_fee": [39.0], "support_tickets": [0], "age": [30],
                       "plan": ["PREMIUM_PLUS"]})
    assert build_features(df)[[f"plan_{p}" for p in PLANS]].sum(axis=1).iloc[0] == 0


def test_no_target_leak():
    df = pd.DataFrame({"tenure_months": [1], "monthly_fee": [39.0], "support_tickets": [0], "age": [30],
                       "plan": ["basic"], "churn": [1]})
    assert "churn" not in build_features(df).columns
