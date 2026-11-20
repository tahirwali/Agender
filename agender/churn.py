"""
The two churn tables of Step 4: Iranian Churn (real, age group) and IBM Telco (simulated, gender).

Iranian Churn (UCI 563, CC BY 4.0). 3,150 customers of an Iranian operator, usage over nine months, age
in five groups, no gender. Three things are fixed before any model is trained:
- `Age` holds one value per age group (15, 25, 30, 45, 55), so it is the label written as a number;
- `Customer Value` is a linear formula of usage whose coefficients differ between age groups (inside each
  group it fits exactly), so it also carries the label;
- some rows are exact copies; a copy in the training part and its twin in the test part would make the
  attack look stronger than it is.
Both columns are dropped and copies removed. What is left is what an operator could release.

IBM Telco (IBM sample data, 7,043 customers of a fictional operator). Gender is part of the simulation,
so it should not be predictable from usage: the table is a negative control. If the audit reported a
leak here, the audit would be wrong. `SeniorCitizen` is an age flag, so it is not released.

The partner's task on both tables is churn, the task these tables were made for.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from agender import data, splits
from agender.step3 import Spec

IRANIAN_URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
IRANIAN_MEMBER = "Customer Churn.csv"
IRANIAN_AGE_NAMES = {0: "group 1 (15)", 1: "group 2 (25)", 2: "group 3 (30)", 3: "group 4 (45)", 4: "group 5 (55)"}
IRANIAN_LEAKING = ["AGE", "CUSTOMER_VALUE"]

TELCO_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
             "Telco-Customer-Churn.csv")
GENDER_NAMES = {0: "male", 1: "female"}


# ---- Iranian Churn ----

def fetch_iranian(folder):
    return data.download(IRANIAN_URL, Path(folder) / "iranian_churn.zip")


def read_iranian(path):
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return pd.read_csv(io.BytesIO(z.read(IRANIAN_MEMBER)))
    return pd.read_csv(path)


def clean_iranian(raw):
    """Upper-case names, copies removed, `Age` and `Customer Value` dropped, AGE = group - 1 (0 ... 4)."""
    table = raw.copy()
    table.columns = [" ".join(c.split()).upper().replace(" ", "_") for c in table.columns]
    table = table.drop_duplicates().drop(columns=IRANIAN_LEAKING).rename(columns={"AGE_GROUP": "AGE"})
    table["AGE"] = table["AGE"] - 1
    # rows that differed only in a dropped column are copies now
    table = table.drop_duplicates().reset_index(drop=True)
    # the tariff plan and the status are codes (1/2), not amounts
    for column in ("TARIFF_PLAN", "STATUS"):
        table[column] = table[column].astype(str)
    table.insert(0, "ID", np.arange(len(table)))
    return table


def iranian_attributes(table):
    return [c for c in table.columns if c not in ("ID", "AGE", "CHURN")]


def iranian_spec(table):
    return Spec(iranian_attributes(table), {"age": (table["AGE"].to_numpy(), 5)},
                table["CHURN"].to_numpy(dtype=int), "classification")


def iranian_tasks(table):
    return {"age": (table["AGE"].to_numpy(), IRANIAN_AGE_NAMES)}


# ---- IBM Telco ----

def fetch_telco(folder):
    return data.download(TELCO_URL, Path(folder) / "telco_customer_churn.csv")


def clean_telco(raw):
    """FEMALE = 1 for women; CHURN = 1 for leavers; TotalCharges blank only for new customers (tenure 0),
    who have paid nothing yet, so 0."""
    table = raw.copy()
    table["TotalCharges"] = pd.to_numeric(table["TotalCharges"].astype(str).str.strip().replace("", np.nan))
    table.loc[table["TotalCharges"].isna() & (table["tenure"] == 0), "TotalCharges"] = 0.0
    table.insert(1, "FEMALE", (table["gender"] == "Female").astype(int))
    table["CHURN"] = (table["Churn"] == "Yes").astype(int)
    table = table.drop(columns=["gender", "Churn"])
    table.insert(0, "ID", np.arange(len(table)))
    return table.reset_index(drop=True)


def telco_attributes(table):
    return [c for c in table.columns if c not in ("ID", "customerID", "FEMALE", "CHURN", "SeniorCitizen")]


def telco_spec(table):
    return Spec(telco_attributes(table), {"gender": (table["FEMALE"].to_numpy(), 2)},
                table["CHURN"].to_numpy(dtype=int), "classification")


def telco_tasks(table):
    return {"gender": (table["FEMALE"].to_numpy(), GENDER_NAMES)}


# ---- splits and description ----

def make_splits(table, label):
    """Same four parts and seeds as LAT, stratified by the private label and churn."""
    strata = table[label].astype(str) + "_" + table["CHURN"].astype(str)
    return splits.make_splits(table["ID"], strata), strata


def describe(name, raw, table, label, attributes):
    return {"data": name, "rows_raw": len(raw), "rows": len(table), "copies_removed": len(raw) - len(table),
            "attributes_released": attributes, "n_attributes": len(attributes),
            "label_counts": {str(k): int(v) for k, v in table[label].value_counts().sort_index().items()},
            "churn_share": round(float(table["CHURN"].mean()), 4),
            "missing_cells": int(table[attributes].isna().sum().sum())}
