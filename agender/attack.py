"""
The attackers of Steps 2B and 3: models the partner can train on the labelled subscribers it already
knows (the training part) to infer age group and gender of everyone else.

All settings are chosen on the validation part only. Each attacker returns class probabilities, so the
same models feed the scores, the conformal sets and the privacy measure of Step 3.
"""
import warnings

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier

from agender.mtl import MultiTaskNet

ATTACKERS = ("Logistic Regression", "LightGBM", "CatBoost", "MLP", "MTL")
FAST_ATTACKERS = ("Logistic Regression", "LightGBM")       # used inside the many Step 3 evaluations


def _full(model, X, n_classes):
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = model.predict_proba(X)
    return out


def logistic(X, y, Xv, yv, n_classes, seed):
    best = None
    for c in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=3000).fit(X, y)
        loss = log_loss(yv, _full(model, Xv, n_classes), labels=range(n_classes))
        if best is None or loss < best[0]:
            best = (loss, model)
    return best[1]


def lightgbm(X, y, Xv, yv, n_classes, seed, fast=False):
    best = None
    for leaves in ((15,) if fast else (7, 15, 31)):
        model = LGBMClassifier(n_estimators=300 if fast else 2000, learning_rate=0.1 if fast else 0.03,
                               num_leaves=leaves, min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, eval_set=[(Xv, yv)], callbacks=[early_stopping(30 if fast else 100, verbose=False)])
        loss = log_loss(yv, _full(model, Xv, n_classes), labels=range(n_classes))
        if best is None or loss < best[0]:
            best = (loss, model)
    return best[1]


def catboost(X, y, Xv, yv, n_classes, seed):
    model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, random_seed=seed, verbose=0,
                               early_stopping_rounds=100, thread_count=-1,
                               loss_function="Logloss" if n_classes == 2 else "MultiClass")
    return model.fit(X, y, eval_set=(Xv, yv))


def mlp(X, y, Xv, yv, n_classes, seed):
    best = None
    for alpha in (1e-4, 1e-2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=alpha, early_stopping=True,
                                  max_iter=300, random_state=seed).fit(X, y)
        loss = log_loss(yv, _full(model, Xv, n_classes), labels=range(n_classes))
        if best is None or loss < best[0]:
            best = (loss, model)
    return best[1]


SINGLE_TASK = {"Logistic Regression": logistic, "LightGBM": lightgbm, "CatBoost": catboost, "MLP": mlp}


NO_INFORMATION = "No information (shuffled labels)"


def train(name, X, y, Xv, yv, n_classes, seed, fast=False):
    """A fitted single-task attacker with a `probabilities(X)` method.

    NO_INFORMATION is the reference line: Logistic Regression trained on shuffled training labels, so it
    has the data but no link to the labels. Its conformal sets show what the guarantee gives by itself
    (at 90 %, about 20 % single-gender sets, half of them wrong)."""
    if name == NO_INFORMATION:
        shuffled = np.random.default_rng(seed).permutation(y)
        model = LogisticRegression(C=1.0, max_iter=3000).fit(X, shuffled)
    elif name == "LightGBM":
        model = lightgbm(X, y, Xv, yv, n_classes, seed, fast=fast)
    else:
        model = SINGLE_TASK[name](X, y, Xv, yv, n_classes, seed)
    return lambda Z: _full(model, Z, n_classes)


def train_mtl(X, gender, age, Xv, gender_v, age_v, n_age, seed):
    """The baseline multi-task network as one more attacker (both tasks at once)."""
    net = MultiTaskNet(X.shape[1], n_age, seed=seed).fit(X, gender, age, Xv, gender_v, age_v, n_age)

    def gender_probabilities(Z):
        p = net.predict(Z)[0]
        return np.column_stack([1 - p, p])

    return gender_probabilities, (lambda Z: net.predict(Z)[1])


def auc(p, y):
    """Gender: ordinary AUC. Age: one-vs-rest macro AUC over the classes present."""
    y = np.asarray(y)
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    values = [roc_auc_score(y == c, p[:, c]) for c in range(p.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(values))


def probabilities_for_all(X, tasks, train_rows, validation_rows, attackers, seed, fast=False):
    """Train every attacker on the training rows and return, per task and attacker, the class
    probabilities of all rows. tasks: {"gender": (labels, n_classes), "age": (labels, n_classes)}."""
    tr, va = train_rows, validation_rows
    out = {task: {} for task in tasks}
    for name in attackers:
        if name == "MTL":
            if set(tasks) != {"gender", "age"}:
                continue
            (g, _), (a, n_age) = tasks["gender"], tasks["age"]
            p_gender, p_age = train_mtl(X[tr], g[tr], a[tr], X[va], g[va], a[va], n_age, seed)
            out["gender"]["MTL"], out["age"]["MTL"] = p_gender(X), p_age(X)
            continue
        for task, (y, n_classes) in tasks.items():
            out[task][name] = train(name, X[tr], y[tr], X[va], y[va], n_classes, seed, fast=fast)(X)
    return out
