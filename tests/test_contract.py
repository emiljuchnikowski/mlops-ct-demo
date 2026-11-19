import pandas as pd
import pytest

from mlpipe.contract import load_contract, validate
from mlpipe.stages import CONTRACT


@pytest.fixture
def contract():
    return {**load_contract(CONTRACT), "min_rows": 1}


@pytest.fixture
def good():
    return pd.DataFrame({
        "customer_id": [1, 2], "tenure_months": [3, 40], "monthly_fee": [39.9, 129.0],
        "plan": ["basic", "premium"], "support_tickets": [0, 2], "age": [25, 61], "churn": [1, 0],
    })


def rules(violations):
    return {(v.column, v.rule) for v in violations}


def test_good_batch_passes(good, contract):
    assert validate(good, contract) == []


def test_missing_column(good, contract):
    assert ("age", "missing") in rules(validate(good.drop(columns="age"), contract))


def test_fee_in_grosze_is_out_of_range(good, contract):
    good["monthly_fee"] = good.monthly_fee * 100
    assert ("monthly_fee", "range") in rules(validate(good, contract))


def test_unknown_category(good, contract):
    good.loc[0, "plan"] = "PREMIUM_PLUS"
    assert ("plan", "allowed") in rules(validate(good, contract))


def test_nulls_in_required_column(good, contract):
    good["tenure_months"] = [None, 12]
    assert ("tenure_months", "nulls") in rules(validate(good, contract))


def test_duplicate_ids(good, contract):
    good["customer_id"] = [7, 7]
    assert ("customer_id", "unique") in rules(validate(good, contract))
