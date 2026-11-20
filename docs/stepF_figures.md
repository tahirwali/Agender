# Step F - figures and tables

**Notebook:** `notebooks/stepF_figures.ipynb` (CPU, inputs `agender-code`, `agender-step2b`, `agender-step3`,
`agender-step4`,
under a minute). Code: `agender/figures.py` and, for Step 4, `agender/figures_churn.py`; test in `tests/test_figures.py`.

Nothing is trained and nothing is typed in: every figure and table is drawn from the files Steps 2B and 3
saved (Step 4's Iranian release files are read only from the folder `iranian_release`, so they are never
mistaken for the LAT files of Step 3). `manifest.json` records the SHA-256 of each input file.

## Figures (`figures/`, PDF + PNG + CSV of the plotted values)

| File | What it shows |
|---|---|
| `fig_attackers` | Test AUC of every attacker (LAT gender, LAT age, Copenhagen gender), mean and range over the five repeats, with the attacker trained on shuffled labels as the reference |
| `fig_conformal` | (a) gender correctly pinned down and (b) age set size against the promised coverage, attacker against reference; (c) coverage inside each gender and age group at 90 %, one threshold for everyone against one threshold per group |
| `fig_tradeoff` | (a) every candidate release of every defence, utility kept against leakage, with the release each defence chooses; (b) gender exposure beyond the reference of each chosen release |
| `fig_information` | What each attribute says about data demand (U_i) and about gender and age group (P_i), and which attributes each QUBO releases |
| `fig_churn` | (a) Iranian Churn age-group AUC and (b) IBM Telco gender AUC of every attacker; (c) Iranian Churn candidate and chosen releases |
| `fig_solver` | Energy above the exact minimum for every (k, gamma), single-bit flips against swap moves |

## Tables (`tables/`, LaTeX + CSV)

| File | Content |
|---|---|
| `table_attackers` | AUC (95 % interval), ECE, accuracy or macro-F1, 90 % sets, per attacker |
| `table_conformal` | Coverage, mean set size, correct single sets at 80/90/95 %, attacker against reference |
| `table_defences` | The release each defence chooses, test values with 95 % intervals over repeats |
| `table_paired` | Privacy-aware QUBO minus each other defence, paired over the five repeats: difference, 95 % interval, paired t-test, repeats favouring each side |
| `table_churn_audit` | Both churn tables: AUC (95 % interval), macro-F1 or accuracy, ECE, 90 % class-conditional sets |
| `table_churn_defences` | Iranian Churn: the release each defence chooses, with 95 % intervals |
| `table_churn_paired` | Iranian Churn: privacy-aware QUBO minus each defence, and how often both release the same attributes |
| `table_churn_releases` | Iranian Churn: released attributes per repeat |
| `table_releases` | Attributes in the release each QUBO chooses, per repeat |

## Statistics

Means over the five repeats carry a 95 % t interval (4 degrees of freedom). Two defences use the same five
splits, so they are compared by paired differences. Gender exposure is the share of test subscribers whose
gender a 90 % class-conditional set pins down correctly, minus the same share for the attacker trained on
shuffled labels; it is given in % of test subscribers.
