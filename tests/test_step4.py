"""Offline checks for Step 4 (the churn tables) and for the dataset-free parts of Step 3."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender import churn, step3


def iranian_like(n=400, seed=0):
    """A small table with the raw Iranian Churn column names; customer value depends on the age group."""
    rng = np.random.default_rng(seed)
    group = rng.integers(1, 6, n)
    usage = rng.integers(0, 50, (n, 8))
    raw = pd.DataFrame(usage, columns=["Call  Failure", "Complains", "Subscription  Length", "Charge  Amount",
                                       "Seconds of Use", "Frequency of use", "Frequency of SMS",
                                       "Distinct Called Numbers"])
    raw["Age Group"] = group
    raw["Tariff Plan"] = rng.integers(1, 3, n)
    raw["Status"] = rng.integers(1, 3, n)
    raw["Age"] = np.array([15, 25, 30, 45, 55])[group - 1]
    raw["Customer Value"] = usage[:, 3] * group + usage[:, 6]
    raw["Churn"] = rng.integers(0, 2, n)
    return pd.concat([raw, raw.iloc[:30]], ignore_index=True)        # 30 exact copies


def test_iranian_cleaning_drops_the_label_columns_and_copies():
    raw = iranian_like()
    table = churn.clean_iranian(raw)
    assert len(table) == len(raw) - 30
    assert "CUSTOMER_VALUE" not in table and "AGE_GROUP" not in table
    assert set(table["AGE"]) <= set(range(5))
    released = churn.iranian_attributes(table)
    assert "AGE" not in released and "CHURN" not in released and len(released) == 10
    assert table["TARIFF_PLAN"].dtype == object                    # a code, not an amount


def test_telco_cleaning_keeps_the_age_flag_out_of_the_release():
    raw = pd.DataFrame({"customerID": ["a", "b", "c"], "gender": ["Female", "Male", "Female"],
                        "SeniorCitizen": [0, 1, 0], "tenure": [0, 5, 12], "MonthlyCharges": [20.0, 30.0, 40.0],
                        "TotalCharges": [" ", "150", "480"], "Churn": ["No", "Yes", "No"]})
    table = churn.clean_telco(raw)
    assert table["FEMALE"].tolist() == [1, 0, 1] and table["CHURN"].tolist() == [0, 1, 0]
    assert table.loc[0, "TotalCharges"] == 0.0                     # new customer, nothing paid yet
    assert churn.telco_attributes(table) == ["tenure", "MonthlyCharges", "TotalCharges"]


def test_secret_code_combines_the_private_labels():
    gender, age = np.array([0, 1, 1, 0]), np.array([5, 0, 3, 2])
    spec = step3.Spec(["a"], {"gender": (gender, 2), "age": (age, 6)}, np.zeros(4), "regression")
    assert spec.secret(np.arange(4)).tolist() == (gender * 6 + age).tolist()
    single = step3.Spec(["a"], {"age": (age, 6)}, np.array([0, 1, 1, 0]), "classification")
    assert single.secret(np.arange(4)).tolist() == age.tolist()
    assert single.target_codes(np.arange(4)).tolist() == [0, 1, 1, 0]


def test_release_sizes_stay_below_the_number_of_attributes():
    assert step3.k_grid([f"a{i}" for i in range(10)]) == (4, 6, 8)
    assert step3.k_grid([f"a{i}" for i in range(24)]) == step3.K_GRID


def test_churn_splits_keep_label_and_churn_mix():
    table = churn.clean_iranian(iranian_like(n=2000, seed=1))
    table_splits, strata = churn.make_splits(table, "AGE")
    parts = table_splits["repeat_0"]
    whole = strata.value_counts(normalize=True)
    test = strata[parts.to_numpy() == "test"].value_counts(normalize=True)
    assert (whole - test.reindex(whole.index).fillna(0)).abs().max() < 0.02
