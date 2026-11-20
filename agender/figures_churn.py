"""
Step F, second part: the figures and tables of the two churn tables (Step 4).

Same rules as figures.py: nothing is typed in, every value comes from the files Step 4 saved, and every
number the text quotes is added to numbers.json under keys that start with "churn /".

Iranian Churn: the private label is the age group (five groups), the partner's task is churn.
IBM Telco: the private label is gender, which is simulated, so the audit should find nothing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .figures import (ATTACKER_NAMES, BLUE, DEFENCE_NAMES, DEFENCE_ORDER, DEFENCE_STYLE, DOUBLE, INK, LIGHT, MM,
                      MUTED, OURS, REFERENCE, curve_means, fmt, fmt_interval, mean_interval, paired, plt,
                      setting_order, tidy)

AUDIT_FILES = ["iranian_scores.csv", "iranian_conformal.csv", "telco_scores.csv", "telco_conformal.csv"]
RELEASE_FILES = ["curves.csv", "operating_points.csv", "exact_check.csv", "information_terms.json"]
CHURN_ATTACKERS = ["Logistic Regression", "LightGBM", "CatBoost", "MLP", REFERENCE]
TABLE_NAMES = {"iranian": "Iranian Churn", "telco": "IBM Telco"}
TASK_OF = {"iranian": "age", "telco": "gender"}


def find_churn_inputs(root):
    """Step 4 files under root; the release files sit in a folder named iranian_release."""
    found = {}
    for name in AUDIT_FILES:
        hits = sorted(Path(root).rglob(name))
        found[name] = hits[0] if hits else None
    for name in RELEASE_FILES:
        hits = sorted(p for p in Path(root).rglob(name) if p.parent.name == "iranian_release")
        found["iranian_release/" + name] = hits[0] if hits else None
    return found


class ChurnInputs:
    def __init__(self, found):
        read = lambda n: pd.read_csv(found[n]) if found.get(n) else None
        self.scores = pd.concat([t for t in (read("iranian_scores.csv"), read("telco_scores.csv")) if t is not None])
        self.conformal = pd.concat([t for t in (read("iranian_conformal.csv"), read("telco_conformal.csv"))
                                    if t is not None])
        self.curves = read("iranian_release/curves.csv")
        self.points = read("iranian_release/operating_points.csv")
        self.exact = read("iranian_release/exact_check.csv")
        terms = found.get("iranian_release/information_terms.json")
        self.terms = json.loads(Path(terms).read_text()) if terms else None

    def auc(self):
        frame = self.scores.copy()
        frame["auc_any"] = frame["auc"].where(frame["task"] == "gender", frame.get("auc_ovr"))
        return frame


# ---- the audit on both tables ----

def table_churn_audit(W, C):
    frame = C.auc()
    sets = C.conformal[(C.conformal["alpha"] == 0.10) & C.conformal["per_class"]]
    rows = []
    for data in ("iranian", "telco"):
        part = frame[frame["data"] == data]
        for name in CHURN_ATTACKERS:
            mine = part[part["attacker"] == name]
            if mine.empty:
                continue
            s = sets[(sets["data"] == data) & (sets["attacker"] == name)]
            score = "macro_f1" if TASK_OF[data] == "age" else "accuracy"
            rows.append({"Data": TABLE_NAMES[data], "Private label": TASK_OF[data], "Attacker": ATTACKER_NAMES[name],
                         "AUC": fmt_interval(*mean_interval(mine["auc_any"])),
                         "Macro-F1 or accuracy": fmt(mine[score].mean()), "ECE": fmt(mine["ece"].mean()),
                         "Coverage %": fmt(100 * s["coverage"].mean(), 1),
                         "Mean set size": fmt(s["mean_size"].mean(), 2),
                         "Correct single %": fmt(100 * s["correct_singletons"].mean(), 1)})
    W.table(pd.DataFrame(rows), "table_churn_audit",
            "The audit on the two churn tables, full release (test, mean over five repeats; AUC with its 95\\% "
            "interval). Iranian Churn: age group (five groups), macro-F1 and one-vs-rest AUC. IBM Telco: gender "
            "(simulated), accuracy and AUC. The last three columns describe class-conditional conformal sets at "
            "90\\% promised coverage.", "tab:churn_audit", "lllrrrrrr")


# ---- the release selection on Iranian Churn ----

CHURN_MEASURES = [("n_released", "Attributes released"), ("utility_kept_test", "Utility kept"),
                  ("leakage_test", "Leakage"), ("age_auc", "Age AUC"), ("age_mean_set", "Age set size"),
                  ("age_at_most_two", "At most two groups (%)")]


def table_churn_defences(W, C):
    rows = []
    for defence in DEFENCE_ORDER:
        part = C.points[C.points["defence"] == defence]
        if part.empty:
            continue
        row = {"Defence": DEFENCE_NAMES[defence]}
        for column, label in CHURN_MEASURES:
            if column == "n_released":
                row[label] = fmt(part[column].mean(), 1)
            else:
                row[label] = fmt_interval(*mean_interval(part[column]), percent=column == "age_at_most_two")
        rows.append(row)
    W.table(pd.DataFrame(rows), "table_churn_defences",
            "Iranian Churn: the release each defence chooses on validation (lowest validation leakage among "
            "releases that keep at least 90\\% of the validation utility). Utility is the churn AUC minus 0.5, as a "
            "share of the full release. Test values, mean over five repeats with the 95\\% interval. The last two "
            "columns describe class-conditional 90\\% sets for the age group (five groups).",
            "tab:churn_defences", "l" + "r" * len(CHURN_MEASURES))


def table_churn_paired(W, C):
    frame = C.points.set_index(["defence", "repeat"]).sort_index()
    ours = frame.loc[OURS]
    rows, out = [], {}
    for defence in DEFENCE_ORDER:
        if defence == OURS or defence not in frame.index.get_level_values(0):
            continue
        other = frame.loc[defence]
        common = ours.index.intersection(other.index)
        same = sum(_released(C, r, OURS) == _released(C, r, defence) for r in common)
        for column, label in (("leakage_test", "Leakage"), ("utility_kept_test", "Utility kept"),
                              ("age_auc", "Age AUC")):
            r = paired(ours.loc[common, column], other.loc[common, column])
            r["same_release"] = same
            out[f"{defence} / {column}"] = r
            rows.append({"Compared with": DEFENCE_NAMES[defence], "Measure": label,
                         "Difference": f"{r['difference']:+.3f}",
                         "95 % interval": f"[{r['low']:+.3f}, {r['high']:+.3f}]",
                         "Paired t-test p": fmt(r["p"], 3),
                         "Repeats lower / higher": f"{r['repeats_lower']} / {r['repeats_higher']}",
                         "Same release": f"{same} of {len(common)}"})
    W.table(pd.DataFrame(rows), "table_churn_paired",
            "Iranian Churn: the QUBO with the privacy term minus each other defence, paired over the five repeats. "
            "A negative difference in leakage or age AUC favours the QUBO with the privacy term. The last column "
            "counts the repeats in which both defences release exactly the same attributes.",
            "tab:churn_paired", "llrrrrr")
    W.numbers["churn / paired_with_qubo_privacy_aware"] = out


def _released(C, repeat, defence):
    point = C.points[(C.points["repeat"] == repeat) & (C.points["defence"] == defence)].iloc[0]
    match = C.curves[(C.curves["repeat"] == repeat) & (C.curves["defence"] == defence)
                     & (C.curves["setting"] == point["setting"])]
    if match.empty:
        return None
    return frozenset(str(match.iloc[0]["released"]).split("|"))


def table_churn_releases(W, C):
    rows = []
    for defence in ("QUBO privacy-aware", "remove most revealing", "QUBO utility only"):
        for r in C.points[C.points["defence"] == defence].sort_values("repeat").itertuples():
            released = _released(C, r.repeat, defence) or []
            rows.append({"Defence": DEFENCE_NAMES[defence], "Repeat": int(r.repeat), "Setting": r.setting,
                         "Released attributes": ", ".join(sorted(a.replace("_", " ").lower() for a in released))})
    W.table(pd.DataFrame(rows), "table_churn_releases",
            "Iranian Churn: attributes in the release chosen by the two QUBO selectors and by removing the most "
            "revealing attributes, per repeat.", "tab:churn_releases", "lllp{0.55\\textwidth}")


# ---- one figure: the two audits and the Iranian trade-off ----

def fig_churn(W, C):
    pyplot = plt()
    frame = C.auc()
    fig, axes = pyplot.subplots(1, 3, figsize=(DOUBLE, 62 * MM), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    rows = []
    for ax, data, title in ((axes[0], "iranian", "(a) Iranian Churn, age group"),
                            (axes[1], "telco", "(b) IBM Telco, gender")):
        part = frame[frame["data"] == data]
        names = [a for a in CHURN_ATTACKERS if a in set(part["attacker"])]
        for y, name in enumerate(reversed(names)):
            values = part.loc[part["attacker"] == name, "auc_any"]
            mean, low, high = mean_interval(values)
            reference = name == REFERENCE
            colour = MUTED if reference else BLUE
            ax.plot([values.min(), values.max()], [y, y], color=LIGHT if reference else colour, lw=2,
                    solid_capstyle="round", zorder=1)
            ax.scatter([mean], [y], s=30, zorder=2, color="white" if reference else colour,
                       edgecolor=colour, linewidth=1.2)
            rows.append({"panel": data, "attacker": name, "auc_mean": mean, "auc_low": low, "auc_high": high,
                         "auc_min": values.min(), "auc_max": values.max()})
        ax.axvline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_xlim(0.4, 1.0)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([ATTACKER_NAMES[a] for a in reversed(names)] if data == "iranian" else [])
        ax.set_xlabel("Test AUC (0.5 = guessing)")
        ax.set_title(title, loc="left")
        tidy(ax)

    ax = axes[2]
    means = curve_means(C.curves)
    for defence in DEFENCE_ORDER:
        part = means[means["defence"] == defence]
        if part.empty or defence == "release everything":
            continue
        colour, marker = DEFENCE_STYLE[defence]
        part = part.assign(order=[setting_order(defence, s) for s in part["setting"]]).sort_values("order")
        ax.scatter(part["leakage_test"], part["utility_kept_test"], s=9, marker=marker, color=colour,
                   alpha=0.45, linewidth=0)
        rows += [{"panel": "trade-off", "kind": "candidate", "defence": defence, **r}
                 for r in part.drop(columns="order").to_dict("records")]
    op = C.points.groupby("defence")[["leakage_test", "utility_kept_test"]].mean()
    for defence in DEFENCE_ORDER:
        if defence not in op.index:
            continue
        colour, marker = DEFENCE_STYLE[defence]
        x, y = op.loc[defence]
        ax.scatter([x], [y], s=50, marker=marker, color=colour, edgecolor="white", linewidth=1.2, zorder=4,
                   label=DEFENCE_NAMES[defence])
        rows.append({"panel": "trade-off", "kind": "chosen release", "defence": defence, "leakage_test": x,
                     "utility_kept_test": y})
    ax.axhline(0.90, color=MUTED, ls=":", lw=0.8)
    ax.set_xlabel("Age leakage (1 = full release, 0 = guessing)")
    ax.set_ylabel("Churn utility kept")
    ax.set_title("(c) Iranian Churn, candidate and chosen releases", loc="left")
    ax.set_ylim(max(0.0, means["utility_kept_test"].min() - 0.05), 1.03)
    ax.legend(loc="lower right", fontsize=6, title="Chosen release", title_fontsize=6.5, markerscale=0.7)
    tidy(ax, "both")
    fig.text(0.01, -0.03, "(a, b) Dot: mean over five repeats; bar: lowest to highest repeat; open dot: attacker "
             "trained on shuffled labels. (c) Small marks: every candidate release (mean over repeats).",
             fontsize=6.5, color=MUTED)
    fig.tight_layout()
    W.figure(fig, "fig_churn", rows)


def collect_churn_numbers(W, C):
    n = W.numbers
    frame = C.auc()
    for (data, name), part in frame.groupby(["data", "attacker"]):
        n[f"churn / {data} / {name} / auc"] = mean_interval(part["auc_any"])
        score = "macro_f1" if TASK_OF[data] == "age" else "accuracy"
        n[f"churn / {data} / {name} / {score}"] = mean_interval(part[score])
    sets = C.conformal[(C.conformal["alpha"] == 0.10) & C.conformal["per_class"]]
    for (data, name), part in sets.groupby(["data", "attacker"]):
        for column in ("coverage", "mean_size", "singletons", "correct_singletons", "at_most_two"):
            n[f"churn / {data} / 90% sets / {name} / {column}"] = mean_interval(part[column])
    for defence, part in C.points.groupby("defence"):
        for column, _ in CHURN_MEASURES:
            n[f"churn / chosen release / {defence} / {column}"] = mean_interval(part[column])
    if C.exact is not None:
        n["churn / solver"] = {label: f"{int(C.exact[f'{label}_found_minimum'].sum())} of {len(C.exact)}"
                               for label in ("single_flip", "with_swaps")}


def run_churn(W, found, log=print):
    """Draw the Step 4 part; skipped (with a note) when the Step 4 outputs are not attached."""
    if any(found.get(n) is None for n in ("iranian_scores.csv", "telco_scores.csv")):
        return "Step 4 outputs not found: the churn figure and tables were not drawn."
    C = ChurnInputs(found)
    steps = [("churn audit table", table_churn_audit)]
    if C.points is not None and C.curves is not None:
        steps += [("churn defences table", table_churn_defences), ("churn paired table", table_churn_paired),
                  ("churn releases table", table_churn_releases), ("churn figure", fig_churn)]
    steps.append(("churn numbers", collect_churn_numbers))
    for name, draw in steps:
        draw(W, C)
        log(f"{name} done")
    return None
