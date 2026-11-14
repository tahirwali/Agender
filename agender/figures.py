"""
Step F: the figures and tables of the paper, drawn only from the saved outputs of Steps 2B and 3.

No number is typed in. Every figure is saved as
  <name>.pdf   vector, for the paper
  <name>.png   for a quick look
  <name>.csv   exactly the values that are plotted
every table as <name>.tex (booktabs) and <name>.csv, and every number the text quotes goes to numbers.json.
manifest.json lists the input files (with SHA-256) and what was written.

Spread over repeats: the five repeats use different splits, so a mean over repeats gets a 95 % t interval
(t with 4 degrees of freedom). Two defences are compared on the same five splits, so their difference is a
paired difference; we give its t interval, a paired t-test and how many of the five repeats favour each side.

Style: Elsevier widths (90 mm single column, 190 mm full width), serif text. Colour carries one message
only: orange is the privacy-aware QUBO, blue the QUBO without the privacy term (and the real attacker in
the audit figures), grey everything else, each grey series with its own marker and a direct label. The
no-information reference is a dashed grey line or an open grey marker.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, LIGHT, GRID = "#2b2b2b", "#8a8a85", "#c3c2b7", "#e6e6e3"

REFERENCE = "No information (shuffled labels)"
ATTACKER_ORDER = ["Logistic Regression", "LightGBM", "CatBoost", "MLP", "MTL", REFERENCE]
ATTACKER_NAMES = {"Logistic Regression": "Logistic regression", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
                  "MLP": "MLP", "MTL": "Multi-task network", REFERENCE: "No information"}
DEFENCE_ORDER = ["release everything", "remove most revealing", "coarsen", "add noise",
                 "QUBO utility only", "QUBO privacy-aware"]
DEFENCE_NAMES = {"release everything": "Release everything", "remove most revealing": "Remove most revealing",
                 "coarsen": "Coarsen values", "add noise": "Add noise",
                 "QUBO utility only": "QUBO, no privacy term", "QUBO privacy-aware": "QUBO with privacy term"}
DEFENCE_STYLE = {"release everything": (MUTED, "s"), "remove most revealing": (MUTED, "v"),
                 "coarsen": (MUTED, "D"), "add noise": (MUTED, "^"),
                 "QUBO utility only": (BLUE, "o"), "QUBO privacy-aware": (ORANGE, "o")}
OURS = "QUBO privacy-aware"
AGE_ORDER = ["<18", "18-25", "26-35", "36-45", "46-60", "60+"]

INPUT_FILES = {
    "step2b": ["lat_scores.csv", "lat_conformal.csv", "lat_intervals.csv",
               "copenhagen_scores.csv", "copenhagen_conformal.csv", "copenhagen_intervals.csv"],
    "step3": ["curves.csv", "operating_points.csv", "exact_check.csv", "information_terms.json"],
}


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    pyplot.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 100, "legend.frameon": False})
    return pyplot


def tidy(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- inputs ----

def find_inputs(root):
    """Locate every input file under root (Kaggle may nest the saved outputs in extra folders)."""
    found = {}
    for step, names in INPUT_FILES.items():
        for name in names:
            hits = sorted(Path(root).rglob(name))
            found[name] = hits[0] if hits else None
    return found


class Inputs:
    def __init__(self, found):
        self.found = found
        read = lambda n: pd.read_csv(found[n]) if found.get(n) else None
        self.scores = pd.concat([t for t in (read("lat_scores.csv"), read("copenhagen_scores.csv")) if t is not None])
        self.conformal = pd.concat([t for t in (read("lat_conformal.csv"), read("copenhagen_conformal.csv"))
                                    if t is not None])
        self.curves = read("curves.csv")
        self.points = read("operating_points.csv")
        self.exact = read("exact_check.csv")
        self.terms = json.loads(Path(found["information_terms.json"]).read_text()) if found.get(
            "information_terms.json") else None

    def has_reference_in_points(self):
        return self.points is not None and "gender_no_information_correct_singletons" in self.points


# ---- small statistics ----

def mean_interval(values):
    """Mean and 95 % t interval over repeats."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float(v.mean()) if len(v) else np.nan, np.nan, np.nan
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def paired(a, b):
    """a - b on the same repeats: mean, 95 % t interval, paired t-test p, repeats with a < b and a > b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, low, high = mean_interval(d)
    p = float(stats.ttest_rel(a, b).pvalue) if len(d) > 1 and np.std(d) > 0 else np.nan
    return {"difference": mean, "low": low, "high": high, "p": p,
            "repeats_lower": int((d < 0).sum()), "repeats_higher": int((d > 0).sum()), "repeats": len(d)}


def fmt(value, digits=3):
    return "--" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def fmt_interval(mean, low, high, digits=3, percent=False):
    scale = 100 if percent else 1
    d = 1 if percent else digits
    if np.isnan(low):
        return fmt(mean * scale, d)
    return f"{mean * scale:.{d}f} [{low * scale:.{d}f}, {high * scale:.{d}f}]"


# ---- writing ----

class Writer:
    def __init__(self, out_dir):
        self.out = Path(out_dir)
        for sub in ("figures", "tables"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.written, self.numbers = [], {}

    def figure(self, fig, name, data):
        base = self.out / "figures" / name
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=200)
        pd.DataFrame(data).to_csv(base.with_suffix(".csv"), index=False)
        plt().close(fig)
        self.written.append(f"figures/{name}")

    def table(self, frame, name, caption, label, columns_format=None, note=None):
        """Every table spans both columns (table*) and is scaled to the text width, so wide ones still fit."""
        base = self.out / "tables" / name
        frame.to_csv(base.with_suffix(".csv"), index=False)
        head = " & ".join(escape(c) for c in frame.columns) + r" \\"
        body = [" & ".join(escape(str(v)) for v in row) + r" \\" for row in frame.itertuples(index=False)]
        spec = columns_format or "l" + "r" * (len(frame.columns) - 1)
        lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
                 r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + spec + "}", r"\toprule", head,
                 r"\midrule", *body, r"\bottomrule", r"\end{tabular}}"]
        if note:
            lines += [r"\par\smallskip", r"\footnotesize " + note]
        lines.append(r"\end{table*}")
        base.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.written.append(f"tables/{name}")


def escape(text):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(a, b)
    return text.replace("<", r"$<$").replace(">=", r"$\geq$").replace("+-", r"$\pm$")


# ---- Step 2B: the attackers ----

def attacker_frame(scores):
    frame = scores.copy()
    frame["auc_any"] = frame["auc"].where(frame["task"] == "gender", frame.get("auc_ovr"))
    return frame


def fig_attackers(W, I):
    """Test AUC of every attacker: LAT gender, LAT age (one-vs-rest), Copenhagen gender."""
    pyplot = plt()
    frame = attacker_frame(I.scores)
    panels = [("lat", "gender", "LAT, gender"), ("lat", "age", "LAT, six age groups"),
              ("copenhagen", "gender", "Copenhagen, gender")]
    panels = [p for p in panels if ((frame["data"] == p[0]) & (frame["task"] == p[1])).any()]
    fig, axes = pyplot.subplots(1, len(panels), figsize=(DOUBLE, 52 * MM), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, (data, task, title) in zip(axes, panels):
        part = frame[(frame["data"] == data) & (frame["task"] == task)]
        names = [a for a in ATTACKER_ORDER if a in set(part["attacker"])]
        for y, name in enumerate(reversed(ATTACKER_ORDER)):
            if name not in names:
                continue
            values = part.loc[part["attacker"] == name, "auc_any"]
            mean, low, high = mean_interval(values)
            reference = name == REFERENCE
            colour = MUTED if reference else BLUE
            ax.plot([values.min(), values.max()], [y, y], color=LIGHT if reference else colour, lw=2,
                    solid_capstyle="round", zorder=1)
            ax.scatter([mean], [y], s=30, zorder=2, color="white" if reference else colour,
                       edgecolor=colour, linewidth=1.2)
            rows.append({"data": data, "task": task, "attacker": name, "auc_mean": mean, "auc_low": low,
                         "auc_high": high, "auc_min": values.min(), "auc_max": values.max()})
        ax.axvline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Test AUC (0.5 = guessing)")
        ax.set_xlim(min(0.4, part["auc_any"].min() - 0.02), max(0.75, part["auc_any"].max() + 0.02))
        tidy(ax)
    axes[0].set_yticks(range(len(ATTACKER_ORDER)))
    axes[0].set_yticklabels([ATTACKER_NAMES[a] for a in reversed(ATTACKER_ORDER)])
    fig.text(0.01, -0.04, "Dot: mean over five repeats; bar: lowest to highest repeat. Open dot: attacker trained "
             "on shuffled labels.", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    W.figure(fig, "fig_attackers", rows)


def table_attackers(W, I):
    frame = attacker_frame(I.scores)
    sets = I.conformal[(I.conformal["alpha"] == 0.10) & I.conformal["per_class"]]
    rows = []
    for (data, task), part in frame.groupby(["data", "task"], sort=False):
        for name in ATTACKER_ORDER:
            mine = part[part["attacker"] == name]
            if mine.empty:
                continue
            mine_sets = sets[(sets["data"] == data) & (sets["task"] == task) & (sets["attacker"] == name)]
            row = {"Data": "LAT" if data == "lat" else "Copenhagen", "Task": task, "Attacker": ATTACKER_NAMES[name],
                   "AUC": fmt_interval(*mean_interval(mine["auc_any"])),
                   "ECE": fmt(mine["ece"].mean())}
            if task == "gender":
                row["Accuracy or macro-F1"] = fmt(mine["accuracy"].mean())
                row["90 % sets"] = fmt(100 * mine_sets["correct_singletons"].mean(), 1) + " % correct single"
            else:
                row["Accuracy or macro-F1"] = fmt(mine["macro_f1"].mean())
                row["90 % sets"] = "mean size " + fmt(mine_sets["mean_size"].mean(), 2)
            rows.append(row)
    W.table(pd.DataFrame(rows), "table_attackers",
            "Test results of every attacker on the full release (mean over five repeats; AUC with its 95\\% "
            "interval). Gender: accuracy; age: macro-F1. Last column: class-conditional conformal sets at 90\\% "
            "coverage.", "tab:attackers", "lllllll")


# ---- Step 2B: the conformal sets ----

def chosen_or_reference(table, data, task):
    """Rows of the attacker chosen on validation (per repeat) and of the no-information reference."""
    part = table[(table["data"] == data) & (table["task"] == task)]
    return part[part["chosen_on_validation"]], part[part["attacker"] == REFERENCE]


def fig_conformal(W, I):
    """(a) gender: correct single sets; (b) age: mean set size; both against the target coverage; (c) coverage
    inside each group at 90 %, plain and class-conditional sets."""
    pyplot = plt()
    table = I.conformal[I.conformal["per_class"]]
    fig, axes = pyplot.subplots(1, 3, figsize=(DOUBLE, 55 * MM), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    rows = []
    for ax, task, measure, label in ((axes[0], "gender", "correct_singletons", "Correctly pinned down (%)"),
                                     (axes[1], "age", "mean_size", "Mean set size (of 6 groups)")):
        chosen, reference = chosen_or_reference(table, "lat", task)
        for part, name, colour, face in ((chosen, "Attacker chosen on validation", BLUE, BLUE),
                                         (reference, "No information", MUTED, "white")):
            if part.empty:
                continue
            g = part.groupby("alpha")[measure]
            alphas = sorted(g.groups)
            scale = 100 if task == "gender" else 1
            means = np.array([g.get_group(a).mean() for a in alphas]) * scale
            lows = np.array([g.get_group(a).min() for a in alphas]) * scale
            highs = np.array([g.get_group(a).max() for a in alphas]) * scale
            x = [100 * (1 - a) for a in alphas]
            ax.plot(x, means, color=colour, lw=1.5, ls="-" if face != "white" else "--")
            ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor=colour, lw=0.8, capsize=2)
            ax.scatter(x, means, s=24, color=face, edgecolor=colour, zorder=3, linewidth=1.2)
            ax.annotate(name, (x[0], means[0]), xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                        color=INK, va="center")
            rows += [{"panel": task, "series": name, "coverage_target": xi, "mean": m, "min": lo, "max": hi}
                     for xi, m, lo, hi in zip(x, means, lows, highs)]
        ax.set_xlabel("Promised coverage (%)")
        ax.set_ylabel(label)
        ax.set_xticks([80, 90, 95])
        ax.set_xlim(77, 101)
        tidy(ax, "y")
    axes[0].set_title("(a) LAT, gender", loc="left")
    axes[1].set_title("(b) LAT, six age groups", loc="left")

    ax = axes[2]
    labels, plain, per_class = [], [], []
    for task, order in (("gender", ["male", "female"]), ("age", AGE_ORDER)):
        for by_class, store in ((False, plain), (True, per_class)):
            part = I.conformal[(I.conformal["data"] == "lat") & (I.conformal["task"] == task)
                               & I.conformal["chosen_on_validation"] & (I.conformal["alpha"] == 0.10)
                               & (I.conformal["per_class"] == by_class)]
            groups = [json.loads(s) for s in part["by_group"]]
            store += [np.mean([g[name]["coverage"] for g in groups if name in g]) for name in order]
        labels += order
    x = np.arange(len(labels))
    ax.axhline(90, color=MUTED, ls="--", lw=0.8)
    ax.scatter(x - 0.12, 100 * np.array(plain), s=22, color="white", edgecolor=MUTED, linewidth=1.2,
               label="One threshold for everyone", zorder=3)
    ax.scatter(x + 0.12, 100 * np.array(per_class), s=22, color=BLUE, label="One threshold per group", zorder=3)
    ax.axvline(1.5, color=GRID, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Coverage inside the group (%)")
    ax.set_title("(c) Coverage per group at 90 % (LAT)", loc="left")
    ax.legend(loc="center", bbox_to_anchor=(0.55, 0.42), handletextpad=0.2)
    tidy(ax, "y")
    rows += [{"panel": "per group", "series": s, "group": g, "mean": 100 * v}
             for s, vals in (("plain", plain), ("per class", per_class)) for g, v in zip(labels, vals)]
    fig.tight_layout()
    W.figure(fig, "fig_conformal", rows)


def table_conformal(W, I):
    table = I.conformal[I.conformal["per_class"]]
    rows = []
    for data, task in (("lat", "gender"), ("lat", "age"), ("copenhagen", "gender")):
        chosen, reference = chosen_or_reference(table, data, task)
        if chosen.empty:
            continue
        for alpha in sorted(chosen["alpha"].unique()):
            row = {"Data": "LAT" if data == "lat" else "Copenhagen", "Task": task,
                   "Promised coverage": f"{100 * (1 - alpha):.0f} %"}
            for part, tag in ((chosen, "attacker"), (reference, "no information")):
                p = part[part["alpha"] == alpha]
                if p.empty:
                    continue
                row[f"Coverage ({tag})"] = fmt(100 * p["coverage"].mean(), 1)
                row[f"Mean size ({tag})"] = fmt(p["mean_size"].mean(), 2)
                row[f"Correct single ({tag}) %"] = fmt(100 * p["correct_singletons"].mean(), 1)
            rows.append(row)
    W.table(pd.DataFrame(rows), "table_conformal",
            "Class-conditional conformal sets on the full release: the attacker chosen on validation against the "
            "attacker trained on shuffled labels (mean over five repeats).", "tab:conformal")


# ---- Step 3: the defences ----

def curve_means(curves):
    keep = ["utility_kept_test", "leakage_test", "utility_kept_validation", "leakage_validation", "n_released"]
    return curves.groupby(["defence", "setting"], sort=False)[keep].mean().reset_index()


def setting_order(defence, setting):
    numbers = [float(t) for t in setting.replace("=", " ").replace(",", " ").split() if t.replace(".", "").isdigit()]
    if defence == "add noise":
        return -numbers[0]              # little noise first, so the line runs from high to low utility
    return numbers[0] if numbers else 0.0


def gender_excess(points):
    if "gender_no_information_correct_singletons" in points:
        return points["gender_correct_singletons"] - points["gender_no_information_correct_singletons"]
    # outputs written before the reference attacker existed: plain share of single sets
    return points.get("gender_correct_singletons", points["gender_singletons"])


def fig_tradeoff(W, I):
    """(a) every candidate release, utility kept against leakage, with the operating points;
    (b) gender exposure beyond the no-information reference at each operating point."""
    pyplot = plt()
    means = curve_means(I.curves)
    fig, axes = pyplot.subplots(1, 2, figsize=(DOUBLE, 72 * MM), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    rows = []
    for defence in DEFENCE_ORDER:
        part = means[means["defence"] == defence]
        if part.empty:
            continue
        colour, marker = DEFENCE_STYLE[defence]
        part = part.assign(order=[setting_order(defence, s) for s in part["setting"]]).sort_values("order")
        if defence in ("remove most revealing", "coarsen", "add noise"):
            ax.plot(part["leakage_test"], part["utility_kept_test"], color=colour, lw=0.9, alpha=0.7)
        if defence != "release everything":
            ax.scatter(part["leakage_test"], part["utility_kept_test"], s=9, marker=marker, color=colour,
                       alpha=0.45, linewidth=0)
        rows += [{"kind": "candidate", "defence": defence, **r} for r in part.drop(columns="order").to_dict("records")]
    op = I.points.groupby("defence")[["leakage_test", "utility_kept_test"]].mean()
    for defence in DEFENCE_ORDER:
        if defence not in op.index:
            continue
        colour, marker = DEFENCE_STYLE[defence]
        x, y = op.loc[defence]
        ax.scatter([x], [y], s=55, marker=marker, color=colour, edgecolor="white", linewidth=1.2, zorder=4,
                   label=DEFENCE_NAMES[defence])
        rows.append({"kind": "operating point", "defence": defence, "leakage_test": x, "utility_kept_test": y})
    ax.axhline(0.90, color=MUTED, ls=":", lw=0.8)
    ax.text(0.02, 0.905, "90 % of the utility kept", fontsize=6.5, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_xlabel("Leakage (1 = as much as releasing everything, 0 = guessing)")
    ax.set_ylabel("Utility kept (share of the full release's $R^2$)")
    ax.set_title("(a) Every candidate release (small) and the release each defence chooses (large)", loc="left")
    ax.set_ylim(max(0.0, means["utility_kept_test"].min() - 0.05), 1.03)
    ax.legend(loc="lower right", title="Chosen release", title_fontsize=7, markerscale=0.8)
    tidy(ax, "both")

    ax = axes[1]
    excess = I.points.assign(excess=100 * gender_excess(I.points))
    names = [d for d in DEFENCE_ORDER if d in set(excess["defence"])]
    for y, defence in enumerate(reversed(names)):
        colour, marker = DEFENCE_STYLE[defence]
        values = excess.loc[excess["defence"] == defence, "excess"]
        ax.scatter(values, [y] * len(values), s=10, color=colour, alpha=0.5, linewidth=0)
        ax.scatter([values.mean()], [y], s=40, marker=marker, color=colour, zorder=3, edgecolor="white")
        ax.annotate(f"{values.mean():.1f}", (values.mean(), y), xytext=(0, 6), textcoords="offset points",
                    fontsize=6.5, ha="center", color=INK)
        rows += [{"kind": "gender excess", "defence": defence, "repeat": r, "excess_percent": v}
                 for r, v in zip(excess.loc[excess["defence"] == defence, "repeat"], values)]
    ax.axvline(0, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([DEFENCE_NAMES[d] for d in reversed(names)])
    reference_note = "beyond no information" if I.has_reference_in_points() else "(no reference in input)"
    ax.set_xlabel(f"Gender correctly pinned down at 90 %\n{reference_note} (%)")
    ax.set_ylim(-0.5, len(names) - 0.3)            # room for the label above the top row
    ax.set_title("(b) Gender exposure of the chosen release", loc="left", pad=8)
    tidy(ax)
    fig.tight_layout()
    W.figure(fig, "fig_tradeoff", rows)


POINT_MEASURES = [("n_released", "Attributes released", False), ("utility_kept_test", "Utility kept", False),
                  ("leakage_test", "Leakage", False), ("gender_auc", "Gender AUC", False),
                  ("age_auc", "Age AUC", False), ("gender_excess", "Gender exposure (%)", True),
                  ("age_mean_set", "Age set size", False)]


def point_table(points):
    frame = points.copy()
    frame["gender_excess"] = gender_excess(frame)
    return frame


def table_defences(W, I):
    frame = point_table(I.points)
    rows = []
    for defence in DEFENCE_ORDER:
        part = frame[frame["defence"] == defence]
        if part.empty:
            continue
        row = {"Defence": DEFENCE_NAMES[defence]}
        for column, label, percent in POINT_MEASURES:
            if column == "n_released":
                row[label] = fmt(part[column].mean(), 0)
            else:
                row[label] = fmt_interval(*mean_interval(part[column]), percent=percent)
        rows.append(row)
    W.table(pd.DataFrame(rows), "table_defences",
            "The release each defence chooses on validation (lowest validation leakage among releases that "
            "keep at least 90\\% of the validation utility). Test values, mean over five repeats with the 95\\% "
            "interval. Gender exposure: subscribers whose gender is correctly pinned down by a 90\\% class-"
            "conditional set, minus the same share for the no-information attacker, in \\% of test subscribers.",
            "tab:defences", "l" + "r" * len(POINT_MEASURES))


def table_paired(W, I):
    frame = point_table(I.points).set_index(["defence", "repeat"]).sort_index()
    ours = frame.loc[OURS]
    rows, out = [], {}
    for defence in DEFENCE_ORDER:
        if defence == OURS or defence not in frame.index.get_level_values(0):
            continue
        other = frame.loc[defence]
        common = ours.index.intersection(other.index)
        for column, label in (("leakage_test", "Leakage"), ("utility_kept_test", "Utility kept"),
                              ("gender_excess", "Gender exposure (%)")):
            r = paired(ours.loc[common, column], other.loc[common, column])
            out[f"{defence} / {column}"] = r
            scale, d = (100, 1) if column == "gender_excess" else (1, 3)
            rows.append({"Compared with": DEFENCE_NAMES[defence], "Measure": label,
                         "Difference": f"{r['difference'] * scale:+.{d}f}",
                         "95 % interval": f"[{r['low'] * scale:+.{d}f}, {r['high'] * scale:+.{d}f}]",
                         "Paired t-test p": fmt(r["p"], 4),
                         "Repeats lower / higher": f"{r['repeats_lower']} / {r['repeats_higher']}"})
    W.table(pd.DataFrame(rows), "table_paired",
            "The QUBO with the privacy term minus each other defence, paired over the five repeats (same splits). "
            "A negative difference in leakage or gender exposure favours the QUBO with the privacy term.",
            "tab:paired", "llrrrr")
    W.numbers["paired_with_qubo_privacy_aware"] = out


def table_releases(W, I):
    part = I.points[I.points["defence"].isin(["QUBO utility only", OURS])].sort_values(["defence", "repeat"])
    if "released" in I.curves:
        lookup = I.curves.set_index(["repeat", "defence", "setting"])["released"]
        part = part.assign(released=[lookup.get((r, d, s), "") for r, d, s in
                                     zip(part["repeat"], part["defence"], part["setting"])])
    rows = [{"Selector": DEFENCE_NAMES[r.defence], "Repeat": int(r.repeat), "Setting": r.setting,
             "Released attributes": str(getattr(r, "released", "")).replace("|", ", ")}
            for r in part.itertuples()]
    W.table(pd.DataFrame(rows), "table_releases",
            "Attributes in the release each QUBO selector chooses, per repeat.", "tab:releases",
            "llll")


# ---- Step 3: the attributes and the solver ----

def information_means(terms):
    frame = pd.DataFrame([{"repeat": r, "attribute": a, **v} for r, block in terms.items() for a, v in block.items()])
    return frame.groupby("attribute", sort=False)[["U", "P"]].mean()


def release_counts(I, defence):
    if "released" not in I.curves:
        return {}
    lookup = I.curves.set_index(["repeat", "defence", "setting"])["released"]
    counts = {}
    for r in I.points[I.points["defence"] == defence].itertuples():
        for name in str(lookup.get((r.repeat, r.defence, r.setting), "")).split("|"):
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def fig_information(W, I):
    """What each attribute says about the partner's target (U) and about gender x age group (P)."""
    pyplot = plt()
    info = information_means(I.terms)
    ours, plain = release_counts(I, OURS), release_counts(I, "QUBO utility only")
    fig, ax = pyplot.subplots(figsize=(SINGLE * 1.25, 80 * MM))
    rows = []
    for name, (u, p) in info.iterrows():
        if ours.get(name, 0) >= 3:
            colour, size = ORANGE, 30
        elif plain.get(name, 0) >= 3:
            colour, size = BLUE, 30
        else:
            colour, size = LIGHT, 16
        ax.scatter([u], [p], s=size, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
        rows.append({"attribute": name, "U": u, "P": p, "privacy_aware_repeats": ours.get(name, 0),
                     "utility_only_repeats": plain.get(name, 0)})
    frame = pd.DataFrame(rows)
    labelled = set(frame.nlargest(5, "P")["attribute"]) | set(frame.nlargest(5, "U")["attribute"]) | \
        {n for n, c in ours.items() if c >= 3} | {n for n, c in plain.items() if c >= 3}
    for r in frame[frame["attribute"].isin(labelled)].itertuples():
        ax.annotate(r.attribute.replace("_", " ").lower(), (r.U, r.P), xytext=(4, 2), textcoords="offset points",
                    fontsize=6, color=INK)
    ax.scatter([], [], s=30, color=ORANGE, label="Released by the QUBO with the privacy term")
    ax.scatter([], [], s=30, color=BLUE, label="Released without it, dropped once it is added")
    ax.scatter([], [], s=16, color=LIGHT, label="Other attributes")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_xlabel("Information about data demand, $U_i$")
    ax.set_ylabel("Information about gender and age group, $P_i$")
    tidy(ax, "both")
    fig.tight_layout()
    W.figure(fig, "fig_information", rows)


def fig_solver(W, I):
    """Energy above the exact minimum, for the single-flip annealer and the annealer with swap moves."""
    if I.exact is None:
        return
    pyplot = plt()
    exact = I.exact.copy()
    ks, gammas = sorted(exact["k"].unique()), sorted(exact["gamma"].unique())
    fig, axes = pyplot.subplots(1, 2, figsize=(DOUBLE * 0.75, 52 * MM), sharey=True)
    rows = []
    top = max((exact["single_flip_energy"] - exact["exact_energy"]).max(), 1e-9)
    for ax, (label, title) in zip(axes, (("single_flip", "(a) Single-bit flips"), ("with_swaps", "(b) With swap moves"))):
        gap = exact.pivot(index="k", columns="gamma", values=f"{label}_energy") - \
            exact.pivot(index="k", columns="gamma", values="exact_energy")
        gap = gap.loc[ks, gammas].clip(lower=0)
        image = ax.imshow(gap.values, cmap="Blues", vmin=0, vmax=top, aspect="auto")
        for i in range(len(ks)):
            for j in range(len(gammas)):
                v = gap.values[i, j]
                text = "0" if v < 1e-9 else ("<0.01" if v < 0.005 else f"{v:.2f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=5.5,
                        color="white" if v > 0.6 * top else INK)
                rows.append({"annealer": label, "k": ks[i], "gamma": gammas[j], "energy_gap": v})
        found = int(exact[f"{label}_found_minimum"].sum())
        ax.set_title(f"{title}\nexact minimum found in {found} of {len(exact)} settings", loc="left")
        ax.set_xticks(range(len(gammas)))
        ax.set_xticklabels([f"{g:g}" for g in gammas])
        ax.set_xlabel(r"Privacy weight $\gamma$")
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("Attributes released, $k$")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Energy above the exact minimum")
    W.figure(fig, "fig_solver", rows)


# ---- the numbers the text quotes ----

def collect_numbers(W, I):
    n = W.numbers
    frame = attacker_frame(I.scores)
    for (data, task, name), part in frame.groupby(["data", "task", "attacker"]):
        n[f"{data} / {task} / {name} / auc"] = mean_interval(part["auc_any"])
    sets = I.conformal[(I.conformal["alpha"] == 0.10) & I.conformal["per_class"]]
    for data, task in (("lat", "gender"), ("lat", "age"), ("copenhagen", "gender")):
        chosen, reference = chosen_or_reference(sets, data, task)
        for part, tag in ((chosen, "chosen"), (reference, "no information")):
            if not part.empty:
                for column in ("coverage", "mean_size", "singletons", "correct_singletons", "at_most_two"):
                    n[f"{data} / {task} / 90% sets / {tag} / {column}"] = mean_interval(part[column])
    points = point_table(I.points)
    for defence, part in points.groupby("defence"):
        for column, _, _ in POINT_MEASURES:
            n[f"operating point / {defence} / {column}"] = mean_interval(part[column])
    if I.exact is not None:
        n["solver"] = {label: f"{int(I.exact[f'{label}_found_minimum'].sum())} of {len(I.exact)}"
                       for label in ("single_flip", "with_swaps")}


# ---- driver ----

def run(input_root, out_dir, log=print):
    found = find_inputs(input_root)
    missing = [name for name, path in found.items() if path is None]
    if any(found[n] is None for n in ("lat_scores.csv", "lat_conformal.csv", "curves.csv", "operating_points.csv")):
        raise FileNotFoundError(f"Step 2B or Step 3 outputs not found under {input_root}; missing: {missing}")
    I = Inputs(found)
    W = Writer(out_dir)
    notes = []
    if not I.has_reference_in_points():
        notes.append("operating_points.csv has no no-information columns: gender exposure is shown without "
                     "subtracting the reference. Re-run Step 3 with the current code.")
    for name, draw in (("attackers figure", fig_attackers), ("attackers table", table_attackers),
                       ("conformal figure", fig_conformal), ("conformal table", table_conformal),
                       ("trade-off figure", fig_tradeoff), ("defences table", table_defences),
                       ("paired table", table_paired), ("releases table", table_releases),
                       ("information figure", fig_information), ("solver figure", fig_solver),
                       ("numbers", collect_numbers)):
        draw(W, I)
        log(f"{name} done")
    with open(W.out / "numbers.json", "w") as f:
        json.dump(W.numbers, f, indent=1, default=float)
    manifest = {"inputs": {n: {"path": str(p), "sha256": sha256(p)} for n, p in found.items() if p},
                "missing_inputs": missing, "written": W.written, "notes": notes}
    with open(W.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    return {"written": len(W.written), "missing_inputs": missing, "notes": notes}
