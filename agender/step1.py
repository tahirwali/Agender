"""
Step 1: clean LAT, build the Copenhagen person table, make the fixed splits.

The splits shipped in splits/ were made with this code. The notebook makes them again from the
Step 0 files and checks they are identical, so the repository and the Kaggle run cannot drift apart.
"""
from pathlib import Path

import pandas as pd

from agender import copenhagen, data, lat, splits


def lat_tables(lat_zip):
    clean = lat.clean_lat(data.read_lat(lat_zip))
    strata = clean["GENDER"] + "_" + clean["AGE_GROUP"]
    return clean, splits.make_splits(clean["ROW"], strata), strata


def copenhagen_tables(data_folder):
    people = copenhagen.build_people(data.read_copenhagen(data_folder))
    strata = people["FEMALE"].map({0: "male", 1: "female"})
    return people, splits.make_splits(people["user"], strata), strata


def write_all(lat_zip, data_folder, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    clean, lat_splits, lat_strata = lat_tables(lat_zip)
    clean.to_csv(out_dir / "lat_clean.csv", index=False)
    splits.save(lat_splits, splits.describe(lat_splits, lat_strata),
                out_dir / "lat_splits.csv", out_dir / "lat_splits_info.json",
                extra={"source": "PLOS S1", "sha256": data.file_hash(lat_zip), "rows": len(clean)})
    report["lat"] = {"rows": len(clean), "features": lat.feature_columns(clean),
                     "strata": lat_strata.value_counts().sort_index().to_dict()}

    people, cns_splits, cns_strata = copenhagen_tables(data_folder)
    people.to_csv(out_dir / "copenhagen_people.csv", index=False)
    splits.save(cns_splits, splits.describe(cns_splits, cns_strata),
                out_dir / "copenhagen_splits.csv", out_dir / "copenhagen_splits_info.json",
                extra={"source": "figshare 7267433", "people": len(people)})
    report["copenhagen"] = {"people": len(people), "features": [c for c in people.columns if c not in ("user", "FEMALE")],
                            "strata": cns_strata.value_counts().sort_index().to_dict()}
    return report


def compare_with_shipped(out_dir, shipped_dir):
    """True for each split file that matches the copy in the repository's splits/ folder."""
    result = {}
    for name in ("lat_splits.csv", "copenhagen_splits.csv"):
        shipped = Path(shipped_dir) / name
        if not shipped.exists():
            result[name] = "not shipped"
            continue
        result[name] = bool(splits.same_splits(pd.read_csv(Path(out_dir) / name), pd.read_csv(shipped)))
    return result
