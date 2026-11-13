"""Offline checks for Step 2B (conformal audit) and Step 3 (privacy-aware release)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender import conformal, lat, qubo, release, splits, step2b, step3
from agender.features import AttributeEncoder


def test_conformal_sets_reach_the_promised_coverage():
    rng = np.random.default_rng(0)
    n, classes = 6000, 4
    logits = rng.normal(size=(n, classes))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(classes, p=row) for row in p])          # labels drawn from the model itself
    cal, test = slice(0, 3000), slice(3000, None)
    for by_class in (False, True):
        q = conformal.thresholds(p[cal], y[cal], 0.10, by_class)
        result = conformal.summary(conformal.prediction_sets(p[test], q), y[test], y[test])
        assert 0.88 <= result["coverage"] <= 0.93
        if by_class:
            assert all(g["coverage"] >= 0.85 for g in result["by_group"].values())


def test_quantile_uses_the_finite_sample_level():
    scores = np.arange(1, 10) / 10              # 0.1 ... 0.9, n = 9
    # ceil(10 * 0.9) / 9 = 1.0 -> the largest score
    assert conformal.quantile(scores, 0.10) == 0.9


def test_pair_synergy_finds_an_xor_pair():
    rng = np.random.default_rng(1)
    a, b, c = rng.integers(0, 2, (3, 4000))
    secret = a ^ b                              # neither a nor b alone says anything; together they say all
    alone = release.mutual_information(secret, a) / release.entropy(secret)
    together = release.mutual_information(secret, release.joint(a, b)) / release.entropy(secret)
    assert alone < 0.01 and together > 0.95
    assert release.mutual_information(secret, release.joint(a, c)) < 0.01


def test_exact_minimum_matches_brute_force():
    rng = np.random.default_rng(2)
    d = 7
    terms = {"U": rng.random(d), "P": rng.random(d) * 0.2, "P_pair": np.triu(rng.random((d, d)) * 0.1, 1),
             "Q": np.triu(rng.random((d, d)), 1)}
    terms["P_pair"] += terms["P_pair"].T
    terms["Q"] += terms["Q"].T
    matrix, constant = release.energy_matrix(terms, 3, gamma=2.0)
    brute = min(qubo.energy(np.array(b), matrix, constant) for b in itertools.product([0, 1], repeat=d))
    x, e = release.exact_minimum(matrix, constant, chunk_bits=4)
    assert np.isclose(e, brute) and np.isclose(qubo.energy(x, matrix, constant), brute)
    released, e_swap, _ = release.qubo_release(terms, list("abcdefg"), 3, 2.0, seed=0)
    assert np.isclose(e_swap, brute) and len(released) == 3


def test_single_flip_annealing_is_unchanged_for_the_replication():
    # the swap option must not change the baseline annealer when it is off
    rng = np.random.default_rng(3)
    m = np.triu(rng.normal(size=(6, 6)))
    assert np.array_equal(qubo.anneal(m, seed=5, iterations=500)[0], qubo.anneal(m, seed=5, iterations=500, swaps=False)[0])


def synthetic_clean(n=900, seed=4):
    """A small table with the cleaned LAT columns; age and gender leak through two attributes."""
    rng = np.random.default_rng(seed)
    age = rng.integers(14, 66, n)
    gender = rng.choice(["F", "M"], n)
    raw = pd.DataFrame({"AGE": age, "GENDER": gender,
                        "PAYMENT_METHOD": rng.choice([f"PMethod {i}" for i in range(12)], n),
                        "N_DEVICES": rng.integers(1, 3, n), "N_SERVICES": rng.integers(0, 3, n),
                        "DEVICE_TYPE": rng.choice(["Phone", "Tablet", None], n),
                        "DEVICE_CAPABILITY": rng.choice(["3G", "LTE"], n),
                        "DEVICE_AGE": rng.choice(["2014 Q2", "2015 Q1", None], n),
                        "CUSTOMER_SERVICE": rng.integers(0, 2, n), "NEED_CUSTOMER_SERVICE": 0,
                        "RECHARGE_WAY": rng.choice([f"Way {i}" for i in range(15)], n),
                        "MAIN_RECHARGE_WAY": rng.choice([f"Way {i}" for i in range(15)], n),
                        "RECHARGE_VALUE": rng.exponential(50, n), "RECHARGES": rng.poisson(3, n),
                        "RECHARGE_AVG": rng.exponential(20, n)})
    for app in ("SOCIAL_NETWORK", "INSTAGRAM", "WHATSAPP", "BANK", "UBER", "INTERNET"):
        raw[f"{app}_MB"] = rng.exponential(100, n)
        raw[f"{app}_SESSIONS"] = rng.poisson(20, n)
    raw["INSTAGRAM_MB"] = rng.exponential(100, n) * (1 + 3 * (age < 30))      # leaks age
    raw["INTERNET_MB"] = raw["SOCIAL_NETWORK_MB"] + raw["WHATSAPP_MB"] + rng.exponential(10, n)
    for name in ("RECEIVED_CALLS", "RECEIVED_CALLS_DURATION", "MADE_CALLS"):
        raw[name] = rng.poisson(30, n)
    raw["MADE_CALLS_DURATION"] = rng.poisson(30, n) + 40 * (gender == "F")     # leaks gender
    clean = lat.clean_lat(raw)
    strata = clean["GENDER"] + "_" + clean["AGE_GROUP"]
    return clean, splits.make_splits(clean["ROW"], strata)


def test_encoder_codes_follow_training_bins():
    clean, _ = synthetic_clean()
    encoder = AttributeEncoder().fit(clean.iloc[:600], lat.feature_columns(clean))
    codes = encoder.codes(clean, "RECHARGE_VALUE", bins=5)
    assert set(np.unique(codes)) <= set(range(5))
    assert encoder.codes(clean, "PAYMENT_METHOD", bins=5).max() <= 4


def test_audit_runs_and_finds_the_leaks(tmp_path):
    clean, split_table = synthetic_clean()
    result = step2b.audit_lat(clean, split_table, tmp_path, repeats=[0],
                              attackers=("Logistic Regression", "LightGBM"))
    assert result["gender / Logistic Regression"]["auc"] > 0.8
    risk = pd.read_csv(tmp_path / "lat_risk.csv")
    assert set(risk["task"]) == {"gender", "age"} and risk["set_size"].min() >= 0


def test_release_curves_and_operating_points(tmp_path, monkeypatch):
    clean, split_table = synthetic_clean()
    monkeypatch.setattr(step3, "K_GRID", (4, 8))
    monkeypatch.setattr(step3, "GAMMA_GRID", (0.0, 8.0))
    monkeypatch.setattr(step3, "COARSE_BINS", (3,))
    monkeypatch.setattr(step3, "NOISE_LEVELS", (1.0,))
    monkeypatch.setattr(step3.attack, "ATTACKERS", ("Logistic Regression", "LightGBM"))
    monkeypatch.setattr(step3.release, "exact_minimum", lambda m, c=0.0: (np.zeros(m.shape[0], int), np.inf))
    summary = step3.run(clean, split_table, tmp_path, repeats=[0], exact=False)
    curve = pd.read_csv(tmp_path / "curves.csv")
    assert {"release everything", "remove most revealing", "QUBO privacy-aware", "coarsen", "add noise"} <= set(curve["defence"])
    full = curve[curve["defence"] == "release everything"].iloc[0]
    assert np.isclose(full["leakage_test"], 1.0) and np.isclose(full["utility_kept_test"], 1.0)
    # the privacy-aware QUBO with a strong privacy weight must avoid the two leaking attributes
    strong = curve[(curve["defence"] == "QUBO privacy-aware") & (curve["setting"] == "k=4, gamma=8.0")].iloc[0]
    assert "MADE_CALLS_DURATION" not in strong["released"] and "INSTAGRAM_MB" not in strong["released"]
    assert "operating_points" in summary
