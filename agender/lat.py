"""
Cleaning rules for LAT, agreed with the authors on 2026-11-13 (see docs/step1_splits.md).

Nothing here is fitted on data: every rule is fixed in advance, so cleaning cannot leak information
from the test split. Encodings that learn from data (frequencies, scaling) are fitted later, on the
training split only.
"""
import numpy as np
import pandas as pd

from agender.audit import AGE_NAMES, age_groups

SESSION_COLUMNS = ["SOCIAL_NETWORK_SESSIONS", "INSTAGRAM_SESSIONS", "WHATSAPP_SESSIONS",
                   "BANK_SESSIONS", "UBER_SESSIONS", "INTERNET_SESSIONS"]
CATEGORICAL = ["PAYMENT_METHOD", "DEVICE_TYPE", "DEVICE_CAPABILITY", "RECHARGE_WAY", "MAIN_RECHARGE_WAY"]
UTILITY_TARGET = "INTERNET_MB"      # the partner's task: predict each subscriber's data demand


def quarter_index(text):
    """'2014 Q2' -> 2014.25 (start of the quarter as a fraction of a year); empty -> NaN."""
    if not isinstance(text, str) or " Q" not in text:
        return np.nan
    year, quarter = text.split(" Q")
    return int(year) + (int(quarter) - 1) / 4


def clean_lat(raw):
    table = raw.copy()

    # The six per-app session counts copy each other (bank = Uber in 80 % of rows), so they are not
    # separate measurements. Adding them would count the same sessions several times; we keep the
    # largest of the six as the subscriber's session count.
    table["SESSIONS"] = table[SESSION_COLUMNS].max(axis=1)
    table = table.drop(columns=SESSION_COLUMNS)

    # NEED_CUSTOMER_SERVICE is almost a yes/no copy of CUSTOMER_SERVICE; keep the count.
    table = table.drop(columns=["NEED_CUSTOMER_SERVICE"])

    # Device age in years before the newest device quarter in the file (larger = older phone).
    quarters = table["DEVICE_AGE"].map(quarter_index)
    table["DEVICE_AGE_YEARS"] = quarters.max() - quarters
    table["DEVICE_AGE_MISSING"] = quarters.isna().astype(int)
    table["DEVICE_AGE_YEARS"] = table["DEVICE_AGE_YEARS"].fillna(0.0)
    table = table.drop(columns=["DEVICE_AGE"])

    # Missing device fields are a category of their own; missingness may itself carry information.
    for column in CATEGORICAL:
        table[column] = table[column].fillna("unknown").astype(str)

    # The 913 subscribers without any mobile data use are kept as they are: zero is a real value.
    table["NO_DATA_USE"] = (table["SESSIONS"] == 0).astype(int)

    table["FEMALE"] = (table["GENDER"] == "F").astype(int)
    table["AGE_GROUP"] = age_groups(table["AGE"]).astype(str)
    table["AGE_GROUP_INDEX"] = table["AGE_GROUP"].map({name: i for i, name in enumerate(AGE_NAMES)})
    table.insert(0, "ROW", np.arange(len(table)))      # position in the original S1 file
    return table


def feature_columns(table):
    """Columns that may be released to the partner: everything except the labels, the row id and the
    partner's own target (if the target were released, the partner's task would be trivial)."""
    labels = {"ROW", "AGE", "GENDER", "FEMALE", "AGE_GROUP", "AGE_GROUP_INDEX", UTILITY_TARGET}
    return [c for c in table.columns if c not in labels]
