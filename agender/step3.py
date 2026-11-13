"""
Step 3 driver: privacy-utility curves of the QUBO release and of the other defences, the operating
point of each defence chosen on validation, and a full Step 2B audit of those operating points.

Every candidate release is scored in the same way:
  utility  R^2 of a LightGBM regressor for log(1 + INTERNET_MB), the partner's task;
  privacy  test AUC of the attacker the partner would pick on validation (Logistic Regression or
           LightGBM, retrained on what is released), for gender and for the six age groups.
Leakage is (AUC - 0.5) / (AUC of the full release - 0.5), averaged over the two tasks: 1 means the
release leaks as much as releasing everything, 0 means the attacker is reduced to guessing.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import r2_score

from agender import attack, conformal, lat, release
from agender.features import AttributeEncoder
from agender.step2b import parts

K_GRID = (4, 6, 8, 10, 12, 16, 20)
GAMMA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
COARSE_BINS = (2, 3, 5, 10)
NOISE_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)
UTILITY_KEPT = 0.90          # an operating point must keep at least 90 % of the full release's validation R^2


class Repeat:
    """Everything one split repeat needs: rows, labels, target, the encoder fitted on training rows."""

    def __init__(self, clean, splits, repeat):
        self.clean, self.repeat = clean, repeat
        self.index = parts(splits, repeat)
        self.attributes = lat.feature_columns(clean)
        self.encoder = AttributeEncoder().fit(clean.iloc[self.index["train"]], self.attributes)
        self.target = np.log1p(clean[lat.UTILITY_TARGET].to_numpy(dtype=float))
        self.tasks = {"gender": (clean["FEMALE"].to_numpy(), 2), "age": (clean["AGE_GROUP_INDEX"].to_numpy(), 6)}

    # ---- the released matrix for each kind of defence ----
    def matrix(self, attributes, coarse_bins=None, noise=None):
        if not attributes:
            return np.zeros((len(self.clean), 1))
        if coarse_bins is None:
            X, _ = self.encoder.transform(self.clean, attributes)
        else:
            # every attribute replaced by its training-quantile bin (numbers) or its top categories
            codes = [self.encoder.codes(self.clean, name, bins=coarse_bins) for name in attributes]
            X = np.column_stack(codes).astype(float)
            tr = self.index["train"]
            X = (X - X[tr].mean(axis=0)) / np.where(X[tr].std(axis=0) > 0, X[tr].std(axis=0), 1.0)
        if noise is not None:
            rng = np.random.default_rng(1000 + self.repeat)
            X = X + rng.normal(0.0, noise, X.shape)
        return X

    def score(self, X):
        """Validation and test utility and privacy of one released matrix (fast models)."""
        tr, va, te = self.index["train"], self.index["validation"], self.index["test"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, random_state=self.repeat,
                                      verbose=-1, n_jobs=-1)
            regressor.fit(X[tr], self.target[tr], eval_set=[(X[va], self.target[va])],
                          callbacks=[early_stopping(50, verbose=False)])
        out = {"r2_validation": r2_score(self.target[va], regressor.predict(X[va])),
               "r2_test": r2_score(self.target[te], regressor.predict(X[te]))}
        probabilities = attack.probabilities_for_all(X, self.tasks, tr, va, attack.FAST_ATTACKERS,
                                                     seed=self.repeat, fast=True)
        for task, (y, _) in self.tasks.items():
            # the partner picks the attacker with the best validation AUC
            chosen = max(probabilities[task], key=lambda n: attack.auc(probabilities[task][n][va], y[va]))
            p = probabilities[task][chosen]
            out[f"{task}_auc_validation"] = attack.auc(p[va], y[va])
            out[f"{task}_auc_test"] = attack.auc(p[te], y[te])
            out[f"{task}_attacker"] = chosen
        return out


# ---- the defences ----

def curves(rep, information, log=print):
    """Score every candidate release of every defence for one repeat."""
    attributes = rep.attributes
    rows = []

    def add(defence, setting, released, **transform):
        started = time.time()
        result = rep.score(rep.matrix(released, **transform))
        rows.append({"repeat": rep.repeat, "defence": defence, "setting": setting, "n_released": len(released),
                     "released": "|".join(released), "seconds": round(time.time() - started, 1), **result})

    add("release everything", "all", attributes)

    # remove the attributes that reveal most about gender and age, one at a time (P_i from the training rows)
    order = [attributes[i] for i in np.argsort(-information["P"])]
    for m in range(1, len(attributes)):
        add("remove most revealing", f"removed {m}", [a for a in attributes if a not in order[:m]])

    # QUBO: gamma = 0 is the baseline selector aimed at the partner's task; gamma > 0 adds privacy
    solutions = []
    for k in K_GRID:
        for gamma in GAMMA_GRID:
            released, e, (matrix, constant) = release.qubo_release(information, attributes, k, gamma, seed=rep.repeat)
            name = "QUBO utility only" if gamma == 0 else "QUBO privacy-aware"
            add(name, f"k={k}, gamma={gamma}", released)
            solutions.append({"repeat": rep.repeat, "k": k, "gamma": gamma, "annealing_energy": e,
                              "released": "|".join(released)})
    log(f"repeat {rep.repeat}: removal and QUBO curves done")

    for bins in COARSE_BINS:
        add("coarsen", f"{bins} levels", attributes, coarse_bins=bins)
    for sigma in NOISE_LEVELS:
        add("add noise", f"sigma={sigma}", attributes, noise=sigma)
    log(f"repeat {rep.repeat}: coarsening and noise done")

    frame = pd.DataFrame(rows)
    full = frame[frame["defence"] == "release everything"].iloc[0]
    for part in ("validation", "test"):
        frame[f"utility_kept_{part}"] = frame[f"r2_{part}"] / full[f"r2_{part}"]
        leak = [(frame[f"{t}_auc_{part}"] - 0.5) / (full[f"{t}_auc_{part}"] - 0.5) for t in rep.tasks]
        frame[f"leakage_{part}"] = np.mean(leak, axis=0)
    return frame, pd.DataFrame(solutions)


def operating_points(curve):
    """Per repeat and defence: among the releases keeping >= 90 % of the validation utility, the one with
    the lowest validation leakage. Chosen without looking at test numbers."""
    chosen = []
    for (repeat, defence), group in curve.groupby(["repeat", "defence"]):
        allowed = group[group["utility_kept_validation"] >= UTILITY_KEPT]
        pick = (allowed.nsmallest(1, "leakage_validation") if len(allowed)
                else group.nlargest(1, "utility_kept_validation"))
        chosen.append(pick)
    return pd.concat(chosen)


def exact_check(information, attributes, repeat):
    """Does simulated annealing find the exact minimum of the energy (all 2^24 subsets)? Checked for the
    baseline single-flip annealer and for the annealer with swap moves used in Step 3."""
    rows = []
    for k in K_GRID:
        for gamma in GAMMA_GRID:
            matrix, constant = release.energy_matrix(information, k, gamma=gamma)
            x, best = release.exact_minimum(matrix, constant)
            exact_set = sorted(attributes[i] for i in np.flatnonzero(x))
            row = {"repeat": repeat, "k": k, "gamma": gamma, "exact_energy": best, "exact_released": "|".join(exact_set)}
            for label, swaps in (("single_flip", False), ("with_swaps", True)):
                released, e, _ = release.qubo_release(information, attributes, k, gamma, seed=repeat, swaps=swaps)
                row[f"{label}_energy"] = e
                row[f"{label}_found_minimum"] = bool(np.isclose(e, best, atol=1e-9))
                row[f"{label}_same_subset"] = sorted(released) == exact_set
            rows.append(row)
    return pd.DataFrame(rows)


def full_audit(rep, released, transform):
    """Step 2B on one operating point: all five attackers and the 90 % class-conditional sets."""
    X = rep.matrix(released, **transform)
    tr, va, ca, te = (rep.index[p] for p in ("train", "validation", "calibration", "test"))
    probabilities = attack.probabilities_for_all(X, rep.tasks, tr, va, attack.ATTACKERS + (attack.NO_INFORMATION,),
                                                 seed=rep.repeat)
    out = {}
    for task, (y, _) in rep.tasks.items():
        reference = probabilities[task].pop(attack.NO_INFORMATION)
        ref_sets = conformal.prediction_sets(reference[te], conformal.thresholds(reference[ca], y[ca], 0.10, by_class=True))
        ref = conformal.summary(ref_sets, y[te])
        out.update({f"{task}_no_information_singletons": ref["singletons"],
                    f"{task}_no_information_correct_singletons": ref["correct_singletons"]})
        chosen = max(probabilities[task], key=lambda n: attack.auc(probabilities[task][n][va], y[va]))
        p = probabilities[task][chosen]
        sets = conformal.prediction_sets(p[te], conformal.thresholds(p[ca], y[ca], 0.10, by_class=True))
        s = conformal.summary(sets, y[te])
        out.update({f"{task}_attacker": chosen, f"{task}_auc": attack.auc(p[te], y[te]),
                    f"{task}_strongest_test_auc": max(attack.auc(q[te], y[te]) for q in probabilities[task].values()),
                    f"{task}_coverage": s["coverage"], f"{task}_mean_set": s["mean_size"],
                    f"{task}_singletons": s["singletons"], f"{task}_correct_singletons": s["correct_singletons"],
                    f"{task}_at_most_two": s["at_most_two"]})
    return out


def transform_of(row):
    setting = row["setting"]
    if row["defence"] == "coarsen":
        return {"coarse_bins": int(setting.split()[0])}
    if row["defence"] == "add noise":
        return {"noise": float(setting.split("=")[1])}
    return {}


def run(clean, splits, out_dir, repeats=range(5), exact=True, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = clean.set_index("ROW", drop=False)
    splits = splits.set_index("id").loc[clean.index].reset_index()
    curve_frames, solution_frames, exact_frames, audit_rows, terms_out = [], [], [], [], {}
    for repeat in repeats:
        rep = Repeat(clean, splits, repeat)
        tr = rep.index["train"]
        train_rows = clean.iloc[tr]
        secret = (train_rows["FEMALE"] * 6 + train_rows["AGE_GROUP_INDEX"]).to_numpy()
        target_bins = np.searchsorted(np.quantile(rep.target[tr], np.linspace(0, 1, 11)[1:-1]), rep.target[tr], side="right")
        information = release.information_terms(rep.encoder, train_rows, rep.attributes, secret, target_bins)
        terms_out[repeat] = {n: {"U": float(information["U"][i]), "P": float(information["P"][i])}
                             for i, n in enumerate(rep.attributes)}
        if exact and repeat == min(repeats):
            exact_frames.append(exact_check(information, rep.attributes, repeat))
            log(f"repeat {repeat}: exact check done")
        curve, solutions = curves(rep, information, log)
        curve_frames.append(curve)
        solution_frames.append(solutions)
        pd.concat(curve_frames).to_csv(out_dir / "curves.csv", index=False)      # saved after every repeat

        for _, row in operating_points(curve).iterrows():
            released = row["released"].split("|") if row["released"] else []
            audit_rows.append({"repeat": repeat, "defence": row["defence"], "setting": row["setting"],
                               "n_released": row["n_released"], "utility_kept_test": row["utility_kept_test"],
                               "r2_test": row["r2_test"], "leakage_test": row["leakage_test"],
                               **full_audit(rep, released, transform_of(row))})
        pd.DataFrame(audit_rows).to_csv(out_dir / "operating_points.csv", index=False)
        log(f"repeat {repeat}: operating points audited")

    pd.concat(solution_frames).to_csv(out_dir / "qubo_solutions.csv", index=False)
    if exact_frames:
        pd.concat(exact_frames).to_csv(out_dir / "exact_check.csv", index=False)
    with open(out_dir / "information_terms.json", "w") as f:
        json.dump(terms_out, f, indent=1)
    return summarise(pd.DataFrame(audit_rows), pd.concat(exact_frames) if exact_frames else None)


def summarise(audit_table, exact_table=None):
    columns = ["n_released", "utility_kept_test", "leakage_test", "gender_auc", "age_auc",
               "gender_singletons", "gender_correct_singletons", "gender_no_information_correct_singletons",
               "age_at_most_two", "age_mean_set"]
    means = audit_table.groupby("defence")[columns].mean().round(3)
    out = {"operating_points": means.to_dict("index")}
    if exact_table is not None:
        n = len(exact_table)
        out["exact_minimum_found"] = {"single_flip (baseline)": f"{int(exact_table['single_flip_found_minimum'].sum())} of {n}",
                                      "with_swaps (Step 3)": f"{int(exact_table['with_swaps_found_minimum'].sum())} of {n}"}
    return out
