"""
Fixed four-part splits, repeated with several seeds.

train       the partner's labelled subscribers (the attacker trains here)
validation  choosing models and settings
calibration conformal prediction sets (never trained on)
test        reported results only

Each stratum (for LAT: gender x six age groups) is shuffled and cut in the same proportions, so every
part keeps the class mix of the whole dataset.
"""
import json

import numpy as np
import pandas as pd

PARTS = ("train", "validation", "calibration", "test")
SHARES = (0.6, 0.1, 0.1, 0.2)
SEEDS = (42, 7, 123, 2024, 99)       # five fixed seeds, one per repeat


def split_one(strata, seed, shares=SHARES):
    strata = pd.Series(strata).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = np.empty(len(strata), dtype=object)
    bounds = np.cumsum(shares)
    for value in sorted(strata.unique(), key=str):
        members = np.flatnonzero((strata == value).to_numpy())
        members = rng.permutation(members)
        cuts = np.rint(bounds * len(members)).astype(int)
        start = 0
        for part, stop in zip(PARTS, cuts):
            parts[members[start:stop]] = part
            start = stop
    return parts


def make_splits(ids, strata, seeds=SEEDS, shares=SHARES):
    table = pd.DataFrame({"id": list(ids)})
    for i, seed in enumerate(seeds):
        table[f"repeat_{i}"] = split_one(strata, seed, shares)
    return table


def describe(splits, strata):
    """Counts per part, per stratum and per repeat, for the record."""
    strata = pd.Series(strata).reset_index(drop=True).astype(str)
    info = {}
    for column in [c for c in splits.columns if c.startswith("repeat_")]:
        info[column] = pd.crosstab(strata, splits[column]).reindex(columns=list(PARTS), fill_value=0).to_dict()
    return info


def save(splits, info, csv_path, json_path, extra=None):
    splits.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({"parts": PARTS, "shares": SHARES, "seeds": SEEDS, "counts": info, **(extra or {})},
                  f, indent=1, default=str)


def same_splits(a, b):
    return a.shape == b.shape and (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all()
