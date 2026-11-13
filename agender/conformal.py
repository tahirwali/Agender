"""
Split conformal prediction sets (Vovk et al. 2005; Angelopoulos and Bates 2023).

The attacker is trained on the training part. On the calibration part, which the attacker never saw,
we compute the score s = 1 - p(true class). For a target error alpha, the threshold q is the
ceil((n + 1)(1 - alpha)) / n empirical quantile of these scores. A test subscriber's set holds every
class c with 1 - p(c) <= q. If calibration and test subscribers are exchangeable, the true class is in
the set with probability at least 1 - alpha.

"Mondrian" (class-conditional) sets use one threshold per true class, so the guarantee holds inside each
age group (or gender) separately, not only on average.
"""
import numpy as np
import pandas as pd


def quantile(scores, alpha):
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def thresholds(p_cal, y_cal, alpha, by_class=False):
    p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal)
    scores = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    n_classes = p_cal.shape[1]
    if not by_class:
        return np.full(n_classes, quantile(scores, alpha))
    out = np.ones(n_classes)                        # a class missing from calibration is always in the set
    for c in range(n_classes):
        inside = y_cal == c
        if inside.any():
            out[c] = quantile(scores[inside], alpha)
    return out


def prediction_sets(p, q):
    """Boolean matrix: class c is in the set of row i when 1 - p[i, c] <= q[c]."""
    return (1 - np.asarray(p)) <= np.asarray(q)[None, :] + 1e-12


def summary(sets, y, groups=None):
    """Coverage, average set size, share of single-class sets (the subscriber is pinned down), share of
    sets with at most two classes, share of empty sets; optionally the same numbers per group."""
    y = np.asarray(y)
    sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    out = {"coverage": float(covered.mean()), "mean_size": float(sizes.mean()),
           "singletons": float((sizes == 1).mean()), "at_most_two": float((sizes <= 2).mean()),
           "empty": float((sizes == 0).mean()),
           "correct_singletons": float(((sizes == 1) & covered).mean())}
    if groups is not None:
        frame = pd.DataFrame({"group": np.asarray(groups), "covered": covered, "single": sizes == 1, "size": sizes})
        per = frame.groupby("group").agg(n=("covered", "size"), coverage=("covered", "mean"),
                                         singletons=("single", "mean"), mean_size=("size", "mean"))
        out["by_group"] = per.round(4).to_dict("index")
    return out
