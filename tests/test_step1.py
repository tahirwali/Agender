"""Offline checks for Step 1 (cleaning, Copenhagen person table, splits)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender import copenhagen, lat, splits

ROOT = Path(__file__).resolve().parents[1]


def raw_lat():
    rows = []
    for i in range(4):
        row = {"AGE": [16, 30, 47, 62][i], "GENDER": "FM"[i % 2], "PAYMENT_METHOD": "PMethod 1",
               "N_DEVICES": 1, "N_SERVICES": 2, "DEVICE_TYPE": [None, "Phone", "Phone", "Tablet"][i],
               "DEVICE_CAPABILITY": "3G", "DEVICE_AGE": ["2014 Q2", None, "2015 Q4", "2013 Q1"][i],
               "CUSTOMER_SERVICE": 0, "NEED_CUSTOMER_SERVICE": 0, "RECHARGE_WAY": "Way 1",
               "MAIN_RECHARGE_WAY": "Way 1", "INTERNET_MB": 5.0}
        for column in lat.SESSION_COLUMNS:
            row[column] = 0 if i == 2 else 10 + i
        row["BANK_SESSIONS"] = 0 if i == 2 else 40
        rows.append(row)
    return pd.DataFrame(rows)


def test_cleaning_rules():
    clean = lat.clean_lat(raw_lat())
    assert "NEED_CUSTOMER_SERVICE" not in clean and "BANK_SESSIONS" not in clean
    assert list(clean["SESSIONS"]) == [40, 40, 0, 40]           # largest of the six, not their sum
    assert list(clean["NO_DATA_USE"]) == [0, 0, 1, 0]
    assert clean.loc[0, "DEVICE_TYPE"] == "unknown"
    # newest quarter 2015 Q4 = 2015.75; 2014 Q2 = 2014.25 -> 1.5 years
    assert clean.loc[0, "DEVICE_AGE_YEARS"] == 1.5 and clean.loc[2, "DEVICE_AGE_YEARS"] == 0.0
    assert clean.loc[1, "DEVICE_AGE_MISSING"] == 1
    assert list(clean["AGE_GROUP"]) == ["<18", "26-35", "46-60", "60+"]
    assert list(clean["FEMALE"]) == [1, 0, 1, 0]


def test_partner_target_and_labels_are_never_released():
    columns = lat.feature_columns(lat.clean_lat(raw_lat()))
    for hidden in ("AGE", "GENDER", "FEMALE", "AGE_GROUP", "ROW", lat.UTILITY_TARGET):
        assert hidden not in columns


def test_splits_are_stratified_disjoint_and_repeatable():
    strata = pd.Series(["a"] * 100 + ["b"] * 50)
    first = splits.make_splits(range(150), strata)
    again = splits.make_splits(range(150), strata)
    assert splits.same_splits(first, again)
    parts = first["repeat_0"]
    assert parts.value_counts().to_dict() == {"train": 90, "test": 30, "validation": 15, "calibration": 15}
    assert (parts[strata == "b"] == "train").sum() == 30
    assert not (first["repeat_0"] == first["repeat_1"]).all()       # different seeds, different splits


def test_copenhagen_people():
    tables = {
        "genders": pd.DataFrame({"user": [1, 2, 3], "female": [1, 0, -1]}),
        "calls": pd.DataFrame({"timestamp": [0, 100, 90000], "caller": [1, 2, 1], "callee": [2, 1, 3],
                               "duration": [30, -1, 50]}),
        "sms": pd.DataFrame({"timestamp": [0, 600, 10000], "sender": [2, 1, 2], "recipient": [1, 2, 1]}),
    }
    people = copenhagen.build_people(tables).set_index("user")
    assert list(people.index) == [1, 2]                     # user 3 has unknown gender
    assert people.loc[1, "calls_made"] == 2 and people.loc[2, "calls_missed_made"] == 1
    assert people.loc[1, "call_active_days"] == 2           # seconds 0 and 90000 fall on two days
    assert people.loc[1, "sms_reply_share"] == 0.5          # answered the first message within an hour
    assert people.loc[2, "sms_reply_share"] == 0.0          # its only answer came 9400 s later


def test_shipped_splits_have_the_expected_size():
    lat_splits = pd.read_csv(ROOT / "splits" / "lat_splits.csv")
    cns_splits = pd.read_csv(ROOT / "splits" / "copenhagen_splits.csv")
    assert lat_splits.shape == (11034, 6) and cns_splits.shape == (582, 6)
    for table in (lat_splits, cns_splits):
        for column in [c for c in table.columns if c.startswith("repeat_")]:
            assert set(table[column]) == set(splits.PARTS)
