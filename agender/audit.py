"""
Step 0 checks: describe each dataset and list anything odd, without changing the data.

Decisions about odd values (drop, keep, treat as missing) are taken after reading the report, so
these functions only count and describe.
"""
import numpy as np
import pandas as pd

# The six age groups used for the age task.
AGE_EDGES = [-np.inf, 17, 25, 35, 45, 60, np.inf]
AGE_NAMES = ["<18", "18-25", "26-35", "36-45", "46-60", "60+"]


def age_groups(age):
    return pd.cut(age, AGE_EDGES, labels=AGE_NAMES)


def counts(series):
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def numeric_columns(table, skip=()):
    """Columns where every non-empty cell is a number."""
    found = []
    for column in table.columns:
        if column in skip:
            continue
        values = table[column].dropna()
        if len(values) and pd.to_numeric(values, errors="coerce").notna().all():
            found.append(column)
    return found


def spikes(table, columns, min_share=0.005):
    """For each column, the most common non-zero value if it is unusually frequent.

    A continuous measure such as megabytes should almost never repeat exactly; a value that does
    may be a fill value written by the operator's export."""
    found = {}
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").dropna()
        nonzero = values[values != 0]
        if nonzero.empty:
            continue
        top = nonzero.value_counts()
        share = top.iloc[0] / len(values)
        distinct_share = nonzero.nunique() / len(nonzero)
        if share >= min_share and distinct_share > 0.05:
            found[column] = {"value": float(top.index[0]), "rows": int(top.iloc[0]), "share": round(float(share), 4)}
    return found


def equal_pairs(table, columns, min_share=0.5):
    """Column pairs that hold the same number in at least min_share of the rows (both non-zero)."""
    numbers = table[columns].apply(pd.to_numeric, errors="coerce")
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            both = numbers[[a, b]].dropna()
            both = both[(both[a] != 0) | (both[b] != 0)]
            if len(both) == 0:
                continue
            share = float((both[a] == both[b]).mean())
            if share >= min_share:
                pairs.append({"a": a, "b": b, "rows_compared": len(both), "equal_share": round(share, 4)})
    return sorted(pairs, key=lambda p: -p["equal_share"])


def audit_lat(table):
    report = {"rows": len(table), "columns": list(table.columns)}
    report["duplicate_rows"] = int(table.duplicated().sum())
    report["missing"] = {c: int(n) for c, n in table.isna().sum().items() if n}
    report["gender"] = counts(table["GENDER"])

    age = pd.to_numeric(table["AGE"], errors="coerce")
    report["age_summary"] = {k: float(v) for k, v in age.describe().round(2).items()}
    report["age_by_year"] = counts(age)
    report["age_groups"] = counts(age_groups(age))
    report["gender_by_age_group"] = {
        str(g): counts(part["GENDER"]) for g, part in table.groupby(age_groups(age), observed=False)}

    numbers = numeric_columns(table, skip=("AGE",))
    report["numeric_columns"] = numbers
    report["categorical_columns"] = {c: int(table[c].nunique()) for c in table.columns
                                     if c not in numbers and c not in ("AGE", "GENDER")}
    report["zero_share"] = {c: round(float((pd.to_numeric(table[c]) == 0).mean()), 4) for c in numbers}

    sessions = [c for c in table.columns if c.endswith("_SESSIONS")]
    megabytes = [c for c in table.columns if c.endswith("_MB")]
    if sessions:
        values = table[sessions].apply(pd.to_numeric, errors="coerce")
        all_zero = (values == 0).all(axis=1)
        report["all_sessions_zero"] = {
            "rows": int(all_zero.sum()),
            # if these people still used megabytes, a zero count means "not recorded", not "no use"
            "of_these_with_some_megabytes": int((table.loc[all_zero, megabytes]
                                                 .apply(pd.to_numeric, errors="coerce") > 0).any(axis=1).sum()),
            "gender": counts(table.loc[all_zero, "GENDER"]),
            "age_groups": counts(age_groups(age[all_zero])),
        }
    report["spikes"] = spikes(table, numbers)
    report["equal_pairs"] = equal_pairs(table, numbers)
    return report


def audit_copenhagen(tables):
    report = {}
    genders = tables.get("genders")
    if genders is not None:
        report["people_with_gender_row"] = len(genders)
        report["female_column"] = counts(genders["female"])      # 1 female, 0 male, -1 unknown (per catalogue)
        known = set(genders.loc[genders["female"].isin([0, 1]), "user"])
    else:
        known = set()
    for name, a, b in (("calls", "caller", "callee"), ("sms", "sender", "recipient"), ("fb_friends", "user_a", "user_b")):
        table = tables.get(name)
        if table is None:
            continue
        people = set(table[a]) | set(table[b])
        per_person = pd.concat([table[a], table[b]]).value_counts()
        report[name] = {
            "records": len(table),
            "people": len(people),
            "people_with_known_gender": len(people & known),
            "records_per_person": {k: float(v) for k, v in per_person.describe().round(1).items()},
        }
        if "duration" in table.columns:
            report[name]["duration_summary"] = {k: float(v) for k, v in table["duration"].describe().round(1).items()}
            report[name]["negative_durations"] = int((table["duration"] < 0).sum())
    return report


def write_report(summary, path):
    """A plain text version of the summary for reading on the Kaggle output page."""
    lines = ["AGender - Step 0 data report", ""]

    lat = summary.get("lat")
    if lat:
        lines += [f"LAT: {lat['rows']} rows, {len(lat['columns'])} columns, {lat['duplicate_rows']} duplicate rows",
                  f"  gender: {lat['gender']}",
                  f"  age: {lat['age_summary']}",
                  f"  age groups: {lat['age_groups']}",
                  f"  missing cells: {lat['missing']}"]
        if "all_sessions_zero" in lat:
            z = lat["all_sessions_zero"]
            lines.append(f"  all session counts zero: {z['rows']} rows, of which {z['of_these_with_some_megabytes']} "
                         f"still used some megabytes")
        lines.append("  frequent non-zero values (possible fill values):")
        lines += [f"    {c}: {s['value']} in {s['rows']} rows ({100 * s['share']:.1f} %)" for c, s in lat["spikes"].items()]
        lines.append("  column pairs that are often identical:")
        lines += [f"    {p['a']} = {p['b']} in {100 * p['equal_share']:.1f} % of {p['rows_compared']} rows"
                  for p in lat["equal_pairs"]]
        lines.append("")

    copenhagen = summary.get("copenhagen")
    if copenhagen:
        lines.append(f"Copenhagen: gender rows {copenhagen.get('people_with_gender_row')}, "
                     f"female column {copenhagen.get('female_column')}")
        for name in ("calls", "sms", "fb_friends"):
            if name in copenhagen:
                part = copenhagen[name]
                lines.append(f"  {name}: {part['records']} records, {part['people']} people, "
                             f"{part['people_with_known_gender']} with known gender")
        lines.append("")

    lines += ["", "Steps:"]
    lines += [f"  {s['step']}: {s['status']} {s['message']}" for s in summary.get("steps", [])]
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
