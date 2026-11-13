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
        assert (out / "tables" / f"{name}.tex").read_text().startswith(r"\begin{table}")
    assert not result["notes"]                     # the reference attacker is present in both inputs
    defences = pd.read_csv(out / "tables" / "table_defences.csv")
    assert defences.iloc[0]["Leakage"].startswith("1.000")
    numbers = json.loads((out / "numbers.json").read_text())
    assert "operating point / QUBO privacy-aware / leakage_test" in numbers
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["inputs"]["curves.csv"]["sha256"]) == 64
