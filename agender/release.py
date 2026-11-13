"""
Step 3: which attributes can the operator release?

The baseline QUBO energy (agender/qubo.py), with relevance moved to the partner's task and a privacy
term added:

    E(x) = - alpha sum_i U_i x_i + beta sum_{i<j} Q_ij x_i x_j
           + gamma ( sum_i P_i x_i + sum_{i<j} P_ij x_i x_j ) + lambda (sum_i x_i - k)^2

U_i   information attribute i carries about the partner's target (log data demand), normalised
Q_ij  normalised mutual information between attributes i and j (redundancy, as in the baseline)
P_i   information attribute i carries about the joint gender x age-group label, normalised
P_ij  what attributes i and j reveal together beyond the sum of what they reveal alone (synergy, >= 0)

Mutual information uses the Miller-Madow bias correction, so small cells do not create false synergy.
All terms are estimated on the training rows only.
"""
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from agender import qubo


# ---- information terms ----

def entropy(codes):
    """Miller-Madow corrected entropy (nats) of a vector of integer codes."""
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() + (len(counts) - 1) / (2 * counts.sum()))


def joint(a, b):
    return np.asarray(a, dtype=np.int64) * (int(np.max(b)) + 1) + np.asarray(b, dtype=np.int64)


def mutual_information(a, b):
    return max(entropy(a) + entropy(b) - entropy(joint(a, b)), 0.0)


def information_terms(encoder, table, attributes, secret, target, pair_bins=5):
    """U, P, P_pair and Q for the given rows (the training rows)."""
    single = {name: encoder.codes(table, name, bins=10) for name in attributes}
    coarse = {name: encoder.codes(table, name, bins=pair_bins) for name in attributes}
    h_secret, h_target = entropy(secret), entropy(target)
    d = len(attributes)
    U = np.array([mutual_information(target, single[n]) / h_target for n in attributes])
    P = np.array([mutual_information(secret, single[n]) / h_secret for n in attributes])
    P_alone = np.array([mutual_information(secret, coarse[n]) for n in attributes])
    P_pair, Q = np.zeros((d, d)), np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            both = mutual_information(secret, joint(coarse[attributes[i]], coarse[attributes[j]]))
            P_pair[i, j] = P_pair[j, i] = max(both - P_alone[i] - P_alone[j], 0.0) / h_secret
            Q[i, j] = Q[j, i] = normalized_mutual_info_score(single[attributes[i]], single[attributes[j]])
    return {"U": U, "P": P, "P_pair": P_pair, "Q": Q}


def energy_matrix(terms, k, alpha=1.0, beta=0.05, gamma=0.0, lam=1.5):
    """Upper-triangular M and constant c with E(x) = x^T M x + c (see qubo.qubo_matrix)."""
    d = len(terms["U"])
    pairs = beta * terms["Q"] + gamma * terms["P_pair"] + 2 * lam
    matrix = np.triu(pairs, k=1)
    matrix[np.diag_indices(d)] = -alpha * terms["U"] + gamma * terms["P"] + lam * (1 - 2 * k)
    return matrix, lam * k * k


def exact_minimum(matrix, constant=0.0, chunk_bits=18):
    """Best x over all 2^d subsets, by enumeration (d = 24 gives about 17 million subsets)."""
    d = matrix.shape[0]
    m = matrix.astype(np.float64)
    shifts = np.arange(d, dtype=np.int64)
    best_e, best_code = np.inf, 0
    step = 1 << min(chunk_bits, d)
    for start in range(0, 1 << d, step):
        codes = np.arange(start, start + step, dtype=np.int64)
        bits = ((codes[:, None] >> shifts) & 1).astype(np.float64)
        e = np.einsum("ij,ij->i", bits @ m, bits)
        i = int(np.argmin(e))
        if e[i] < best_e:
            best_e, best_code = float(e[i]), int(codes[i])
    return ((best_code >> shifts) & 1).astype(int), best_e + constant


def qubo_release(terms, attributes, k, gamma, seed, alpha=1.0, beta=0.05, lam=1.5, swaps=True):
    """Attributes chosen by simulated annealing (with swap moves unless swaps=False)."""
    matrix, constant = energy_matrix(terms, k, alpha, beta, gamma, lam)
    x, e = qubo.anneal(matrix, constant, seed=seed, swaps=swaps)
    return [attributes[i] for i in np.flatnonzero(x)], e, (matrix, constant)
