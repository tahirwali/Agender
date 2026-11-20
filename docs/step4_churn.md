# Step 4 - the two churn tables

**Notebook:** `notebooks/step4_churn.ipynb` (CPU, input `agender-code`, Internet on, about 30-40 min).

## Why two more tables

LAT shows how much age and gender leak from one operator's aggregates. Two churn tables test whether the
findings carry over, and whether the audit can tell a leak from no leak.

| Table | Private label | Partner's task | Role |
|---|---|---|---|
| Iranian Churn (real customers) | age group, five groups | churn | a second operator, a second kind of release |
| IBM Telco (simulated customers) | gender | churn | negative control: gender is simulated, so the audit should find nothing |

## Cleaning (fixed before any model is trained)

Iranian Churn: `Age` (one value per age group) and `Customer Value` (a formula whose coefficients differ
between age groups) are dropped, because both are the label in another form. Exact copies are removed before
and after dropping them (300 rows in all). `Tariff Plan` and `Status` are codes and are encoded as categories.
Ten attributes are released.

IBM Telco: `TotalCharges` is blank for the 11 customers with tenure 0, who have paid nothing yet, so it is 0
for them. `SeniorCitizen` is an age flag and is not released; `customerID` is not released. Seventeen
attributes are released.

## Splits

The same four parts (train 60 %, validation 10 %, calibration 10 %, test 20 %) and five seeds as LAT,
stratified by the private label and churn, so every part keeps both mixes.

## What runs

- **Audit** (Step 2B) on both tables: Logistic Regression, LightGBM, CatBoost, MLP and the no-information
  reference; 90 % conformal sets, plain and class-conditional. The multi-task network needs both labels and is
  not used here.
- **Release selection** (Step 3) on Iranian Churn only. Utility is the churn AUC minus 0.5 of a LightGBM
  model on what is released, relative to releasing everything; leakage is defined as on LAT. The QUBO tries
  k = 4, 6, 8 (k must stay below the ten attributes). The exact check enumerates all 2^10 subsets.
  On IBM Telco the full release leaks almost nothing (AUC close to 0.5), so a leakage score relative to it
  would divide by almost zero; the release selection is not run there.

## Output (`agender_outputs/step4/`)

`iranian_clean.csv`, `telco_clean.csv`, their splits; `iranian_*` and `telco_*` audit files (scores,
conformal sets, per-customer risk, bootstrap intervals, as in Step 2B); `iranian_release/` (curves,
operating points, QUBO solutions, exact check, information terms, as in Step 3); `step4_summary.json`.
