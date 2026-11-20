"""Step F on small synthetic outputs of Steps 2B and 3."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender import figures, step2b, step3
from test_step2b3 import synthetic_clean


def test_paired_difference_and_interval():
    a, b = np.array([0.40, 0.35, 0.42, 0.38, 0.41]), np.array([0.56, 0.52, 0.60, 0.55, 0.57])
    r = figures.paired(a, b)
    assert np.isclose(r["difference"], (a - b).mean())
    assert r["low"] < r["difference"] < r["high"] < 0
    assert r["repeats_lower"] == 5 and r["p"] < 0.001
    mean, low, high = figures.mean_interval([1.0, 1.0, 1.0])
    assert mean == low == high == 1.0


def test_latex_escape():
    assert figures.escape("90 % & k_2 <18") == r"90 \% \& k\_2 $<$18"


def test_figures_from_saved_outputs(tmp_path, monkeypatch):
    clean, split_table = synthetic_clean()
    step2b.audit_lat(clean, split_table, tmp_path / "in" / "step2b", repeats=[0, 1],
                     attackers=("Logistic Regression", "LightGBM", figures.REFERENCE))
    monkeypatch.setattr(step3, "K_GRID", (4, 8))
    monkeypatch.setattr(step3, "GAMMA_GRID", (0.0, 8.0))
    monkeypatch.setattr(step3, "COARSE_BINS", (3,))
    monkeypatch.setattr(step3, "NOISE_LEVELS", (1.0,))
    monkeypatch.setattr(step3.attack, "ATTACKERS", ("Logistic Regression", "LightGBM"))
    step3.run(clean, split_table, tmp_path / "in" / "step3", repeats=[0, 1], exact=False)

    result = figures.run(tmp_path / "in", tmp_path / "out", log=lambda message: None)
    out = tmp_path / "out"
    for name in ("fig_attackers", "fig_conformal", "fig_tradeoff", "fig_information"):
        assert (out / "figures" / f"{name}.pdf").exists() and (out / "figures" / f"{name}.csv").exists()
    for name in ("table_attackers", "table_conformal", "table_defences", "table_paired", "table_releases"):
        assert (out / "tables" / f"{name}.tex").read_text().startswith(r"\begin{table*}")
    assert not result["notes"]                     # the reference attacker is present in both inputs
    defences = pd.read_csv(out / "tables" / "table_defences.csv")
    assert defences.iloc[0]["Leakage"].startswith("1.000")
    numbers = json.loads((out / "numbers.json").read_text())
    assert "operating point / QUBO privacy-aware / leakage_test" in numbers
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["inputs"]["curves.csv"]["sha256"]) == 64


def test_churn_part_from_step4_outputs(tmp_path):
    """The Step 4 tables are drawn from their own files, and the LAT step does not pick up iranian_release/."""
    from agender import figures_churn
    attackers = ["Logistic Regression", "LightGBM", figures.REFERENCE]
    rng = np.random.default_rng(0)
    folder = tmp_path / "in" / "step4"
    (folder / "iranian_release").mkdir(parents=True)
    for data, task, score in (("iranian", "age", "auc_ovr"), ("telco", "gender", "auc")):
        rows = [{"data": data, "repeat": r, "task": task, "attacker": a, "chosen_on_validation": a == "LightGBM",
                 "accuracy": 0.5, "macro_f1": 0.4, score: 0.5 if a == figures.REFERENCE else 0.9 - 0.01 * r,
                 "ece": 0.03} for r in range(3) for a in attackers]
        pd.DataFrame(rows).to_csv(folder / f"{data}_scores.csv", index=False)
        sets = [{"data": data, "repeat": r, "task": task, "attacker": a, "chosen_on_validation": a == "LightGBM",
                 "alpha": 0.10, "per_class": True, "coverage": 0.9, "mean_size": 2.0, "singletons": 0.3,
                 "at_most_two": 0.6, "empty": 0.0, "correct_singletons": 0.25, "by_group": "{}"}
                for r in range(3) for a in attackers]
        pd.DataFrame(sets).to_csv(folder / f"{data}_conformal.csv", index=False)
    points, curves = [], []
    for r in range(3):
        for defence in figures.DEFENCE_ORDER:
            released = "A|B" if "QUBO" in defence or defence == "remove most revealing" else "A|B|C"
            leak = 1.0 if defence == "release everything" else 0.5 + 0.1 * rng.random()
            points.append({"repeat": r, "defence": defence, "setting": "k=2, gamma=1.0", "n_released": released.count("|") + 1,
                           "utility_kept_test": 0.9, "leakage_test": leak, "age_auc": 0.8, "age_mean_set": 2.5,
                           "age_at_most_two": 0.5})
            curves.append({"repeat": r, "defence": defence, "setting": "k=2, gamma=1.0", "n_released": 2, "released": released,
                           "utility_kept_test": 0.9, "leakage_test": leak, "utility_kept_validation": 0.9,
                           "leakage_validation": leak})
    pd.DataFrame(points).to_csv(folder / "iranian_release" / "operating_points.csv", index=False)
    pd.DataFrame(curves).to_csv(folder / "iranian_release" / "curves.csv", index=False)

    found = figures_churn.find_churn_inputs(tmp_path / "in")
    assert found["iranian_release/curves.csv"].parent.name == "iranian_release"
    assert figures.find_inputs(tmp_path / "in")["curves.csv"] is None
    W = figures.Writer(tmp_path / "out")
    assert figures_churn.run_churn(W, found, log=lambda message: None) is None
    for name in ("table_churn_audit", "table_churn_defences", "table_churn_paired", "table_churn_releases"):
        assert (tmp_path / "out" / "tables" / f"{name}.tex").exists()
    assert (tmp_path / "out" / "figures" / "fig_churn.pdf").exists()
    paired = W.numbers["churn / paired_with_qubo_privacy_aware"]["remove most revealing / leakage_test"]
    assert paired["same_release"] == 3
    assert figures_churn.run_churn(W, {}, log=lambda message: None).startswith("Step 4 outputs not found")
