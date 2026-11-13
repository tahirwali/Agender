# Step 3 - which attributes can be released

**Notebook:** `notebooks/step3_release.ipynb` (CPU, inputs `agender-code` and `agender-step1`, about 50-70 min).

## Energy (extends the baseline QUBO of `agender/qubo.py`)

E(x) = - alpha sum U_i x_i + beta sum Q_ij x_i x_j + gamma ( sum P_i x_i + sum P_ij x_i x_j ) + lambda (sum x_i - k)^2

- U_i: mutual information of attribute i with the partner's target (log(1 + INTERNET_MB), 10 quantile bins), / H(target).
- Q_ij: normalised mutual information between attributes (as in the baseline).
- P_i: mutual information with the joint gender x age-group label (12 classes), / H(label).
- P_ij: synergy, max(0, I(label; i, j) - I(label; i) - I(label; j)) / H(label), with 5 bins per attribute.
- Miller-Madow bias correction for every entropy. All terms from training rows only.
- alpha 1, beta 0.05, lambda 1.5 as in the baseline; k in {4, 6, 8, 10, 12, 16, 20}; gamma in {0, 0.5, 1, 2, 4, 8, 16}.

## Solver

Simulated annealing with the baseline schedule (6 restarts x 20,000 moves, T 5 -> 0.01) plus swap moves
(every second move takes one chosen attribute out and puts another in). Reason: with the cardinality
penalty, single-bit flips must pass through k +- 1 attributes, which costs about lambda; once the temperature
is low the search freezes. The notebook checks both annealers in all 49 settings against the exact minimum
over all 2^24 subsets.

## Defences and scoring

Release everything; remove the most revealing attributes (by P_i) one at a time; QUBO with gamma = 0 (the
baseline selector, aimed at the partner's task); privacy-aware QUBO; coarsening every attribute to 2, 3, 5
or 10 levels; Gaussian noise with sigma 0.25 to 4 on the standardised values.

Each candidate release: utility = LightGBM R^2 for log data demand; leakage = mean over gender and age of
(AUC - 0.5) / (AUC of releasing everything - 0.5), with the attacker (Logistic Regression or LightGBM,
retrained on the release) picked on validation. Operating point per defence and repeat, chosen on validation:
lowest leakage among releases that keep at least 90 % of the validation R^2. Each operating point gets the full
Step 2B audit (five attackers, class-conditional 90 % sets).

## Output (`agender_outputs/step3/`)

`curves.csv` (every candidate release), `operating_points.csv` (full audit), `exact_check.csv`,
`qubo_solutions.csv`, `information_terms.json`, `step3_summary.json`.
