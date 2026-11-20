"""
Step 4: the audit (Step 2B) and the release selection (Step 3) on the two churn tables.

Iranian Churn: does usage reveal the age group, and which attributes can be released for churn models
while the age group stays hidden?
IBM Telco: the same questions for gender, on a table where gender is simulated. The audit should find
no leak there; a leak would point to a fault in the audit itself.

Both use the same four-part splits and five seeds as LAT, stratified by the private label and churn.
"""
import json
from pathlib import Path

import pandas as pd

from agender import attack, churn, splits, step2b, step3

DATASETS = {
    "iranian": {"label": "AGE", "spec": churn.iranian_spec, "tasks": churn.iranian_tasks},
    "telco": {"label": "FEMALE", "spec": churn.telco_spec, "tasks": churn.telco_tasks},
}


def prepare(name, table, out_dir):
    """Splits for one table, saved next to the cleaned table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = DATASETS[name]["label"]
    table_splits, strata = churn.make_splits(table, label)
    splits.save(table_splits, splits.describe(table_splits, strata),
                out_dir / f"{name}_splits.csv", out_dir / f"{name}_splits_info.json")
    table.to_csv(out_dir / f"{name}_clean.csv", index=False)
    return table_splits


def audit(name, table, table_splits, out_dir, repeats=range(5)):
    """Step 2B on one churn table: four attackers (the multi-task net needs both labels) and the reference."""
    spec = DATASETS[name]["spec"](table)
    indexed = table.set_index("ID", drop=False)
    aligned = table_splits.set_index("id").loc[indexed.index].reset_index()
    attackers = [a for a in attack.ATTACKERS + (attack.NO_INFORMATION,) if a != "MTL"]
    return step2b.audit(indexed, spec.attributes, DATASETS[name]["tasks"](table), aligned, out_dir,
                        repeats=repeats, attackers=attackers, name=name)


def release(name, table, table_splits, out_dir, repeats=range(5), exact=True, log=print):
    """Step 3 on one churn table, written to <out_dir>/<name>_release/."""
    spec = DATASETS[name]["spec"](table)
    return step3.run(table, table_splits, Path(out_dir) / f"{name}_release", repeats=repeats, exact=exact,
                     log=log, spec=spec, id_column="ID")


def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=str)


def read_clean(name, folder):
    """A cleaned table saved by `prepare`, with the code columns read back as text."""
    text = {"iranian": {"TARIFF_PLAN": str, "STATUS": str}, "telco": {}}[name]
    return pd.read_csv(Path(folder) / f"{name}_clean.csv", dtype=text)
