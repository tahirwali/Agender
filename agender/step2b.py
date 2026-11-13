"""
Step 2B: the leakage audit.

Threat model. The operator releases the attributes of all subscribers. The partner already knows age
and gender of some of them (the training, validation and calibration parts) and wants them for the
others (the test part). It trains attackers on the training part, picks settings on the validation
part and builds conformal sets on the calibration part. The test part plays the unknown subscribers.

For each subscriber the audit reports a conformal set of possible classes. A single-class set means
the partner has pinned that subscriber down with the stated guarantee.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agender import attack, conformal, lat, metrics
from agender.audit import AGE_NAMES
from agender.features import AttributeEncoder

ALPHAS = (0.05, 0.10, 0.20)
GENDER_NAMES = {0: "male", 1: "female"}


def parts(splits, repeat):
    column = splits[f"repeat_{repeat}"].to_numpy()
    return {p: np.flatnonzero(column == p) for p in ("train", "validation", "calibration", "test")}


def task_scores(p, y):
    """Test scores of one attacker on one task."""
    if p.shape[1] == 2:
        s = metrics.gender_scores(p[:, 1], y)
        return {"accuracy": s["accuracy"], "auc": s["auc"], "ece": s["ece"]}
    s = metrics.age_scores(p, y)
    return {k: s[k] for k in ("accuracy", "within_1", "macro_f1", "balanced", "auc_ovr", "ece")}


def bootstrap(values_fn, n, draws=1000, seed=0):
    """95 % interval of a test statistic by resampling test subscribers."""
    rng = np.random.default_rng(seed)
    stats = [values_fn(rng.integers(0, n, n)) for _ in range(draws)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def audit(table, attributes, tasks, splits, out_dir, repeats=range(5), attackers=attack.ATTACKERS + (attack.NO_INFORMATION,),
          name="lat"):
    """tasks: {"gender": (labels, class names), "age": (labels, class names)}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_rows, conformal_rows, risk_frames, interval_rows = [], [], [], []
    for repeat in repeats:
        index = parts(splits, repeat)
        tr, va, ca, te = index["train"], index["validation"], index["calibration"], index["test"]
        encoder = AttributeEncoder().fit(table.iloc[tr], attributes)
        X, _ = encoder.transform(table)
        probabilities = attack.probabilities_for_all(
            X, {task: (y, len(classes)) for task, (y, classes) in tasks.items()}, tr, va, attackers, seed=repeat)

        for task, (y, classes) in tasks.items():
            # the partner's attacker: best validation AUC among the real attackers (not the reference line)
            real = [m for m in probabilities[task] if m != attack.NO_INFORMATION]
            chosen = max(real, key=lambda m: attack.auc(probabilities[task][m][va], y[va]))
            for attacker, p in probabilities[task].items():
                score_rows.append({"data": name, "repeat": repeat, "task": task, "attacker": attacker,
                                   "chosen_on_validation": attacker == chosen, **task_scores(p[te], y[te])})
                for alpha in ALPHAS:
                    for by_class in (False, True):
                        q = conformal.thresholds(p[ca], y[ca], alpha, by_class)
                        sets = conformal.prediction_sets(p[te], q)
                        groups = [classes[c] for c in y[te]]
                        result = conformal.summary(sets, y[te], groups)
                        by_group = result.pop("by_group")
                        conformal_rows.append({"data": name, "repeat": repeat, "task": task, "attacker": attacker,
                                               "chosen_on_validation": attacker == chosen, "alpha": alpha,
                                               "per_class": by_class, **result,
                                               "by_group": json.dumps(by_group)})
            # per-subscriber risk and intervals for the attacker the partner would pick
            p = probabilities[task][chosen]
            q = conformal.thresholds(p[ca], y[ca], 0.10, by_class=True)
            sets = conformal.prediction_sets(p[te], q)
            risk_frames.append(pd.DataFrame({
                "data": name, "repeat": repeat, "task": task, "attacker": chosen, "id": table.index[te],
                "true": [classes[c] for c in y[te]], "set_size": sets.sum(axis=1),
                "set": ["|".join(classes[c] for c in np.flatnonzero(row)) for row in sets],
                "p_true": p[te, y[te]]}))
            interval_rows.append({
                "data": name, "repeat": repeat, "task": task, "attacker": chosen,
                "auc": attack.auc(p[te], y[te]),
                "auc_95": bootstrap(lambda b: attack.auc(p[te][b], y[te][b]), len(te), seed=repeat),
                "singletons_90": float((sets.sum(axis=1) == 1).mean()),
                "at_most_two_90": float((sets.sum(axis=1) <= 2).mean()),
                "singletons_90_95": bootstrap(lambda b: float((sets[b].sum(axis=1) == 1).mean()), len(te), seed=repeat)})
        print(f"{name}: repeat {repeat} done")

    scores = pd.DataFrame(score_rows)
    conformal_table = pd.DataFrame(conformal_rows)
    scores.to_csv(out_dir / f"{name}_scores.csv", index=False)
    conformal_table.to_csv(out_dir / f"{name}_conformal.csv", index=False)
    pd.concat(risk_frames).to_csv(out_dir / f"{name}_risk.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(out_dir / f"{name}_intervals.csv", index=False)
    return overview(scores, conformal_table)


def overview(scores, conformal_table):
    """Means over repeats: test scores of every attacker, and the 90 % class-conditional sets."""
    numeric = scores.drop(columns=["repeat", "chosen_on_validation"]).groupby(["task", "attacker"]).mean(numeric_only=True)
    sets = conformal_table[(conformal_table["alpha"] == 0.10) & conformal_table["per_class"]]
    sets = sets.groupby(["task", "attacker"])[["coverage", "mean_size", "singletons", "at_most_two",
                                               "correct_singletons"]].mean()
    out = {}
    for (task, attacker), row in numeric.round(3).iterrows():
        out[f"{task} / {attacker}"] = {**row.dropna().to_dict(), **sets.loc[(task, attacker)].round(3).to_dict()}
    return out


# ---- the two datasets ----

def audit_lat(clean, splits, out_dir, **options):
    clean = clean.set_index("ROW", drop=False)
    splits = splits.set_index("id").loc[clean.index].reset_index()
    tasks = {"gender": (clean["FEMALE"].to_numpy(), GENDER_NAMES),
             "age": (clean["AGE_GROUP_INDEX"].to_numpy(), dict(enumerate(AGE_NAMES)))}
    return audit(clean, lat.feature_columns(clean), tasks, splits, out_dir, name="lat", **options)


def audit_copenhagen(people, splits, out_dir, **options):
    people = people.set_index("user", drop=False)
    splits = splits.set_index("id").loc[people.index].reset_index()
    attributes = [c for c in people.columns if c not in ("user", "FEMALE")]
    tasks = {"gender": (people["FEMALE"].to_numpy(), GENDER_NAMES)}
    attackers = [a for a in options.pop("attackers", attack.ATTACKERS + (attack.NO_INFORMATION,)) if a != "MTL"]
    return audit(people, attributes, tasks, splits, out_dir, name="copenhagen", attackers=attackers, **options)
