# AGender

How much age and gender information do released mobile-usage aggregates leak, and which aggregates can an
operator release safely?

Telecom operators share summarised subscriber data with partners. Even without names or phone numbers, these
summaries can reveal a person's age and gender. AGender measures this leakage for each subscriber, with a
coverage guarantee, and selects which aggregates to release so that a partner's task still works while
demographic inference falls towards random guessing.

Everything runs on Kaggle's free tier, on the CPU.

## Pipeline

| Step | Notebook | What it does | Hardware |
|---|---|---|---|
| 0 | `step0_data.ipynb` | Downloads LAT, takes the Copenhagen Networks Study from an attached copy (MD5 checked), and reports missing and odd values | CPU |
| 1 | `step1_splits.ipynb` | Applies the cleaning rules, builds one row per Copenhagen student, and makes the fixed four-part splits (checked against `splits/`) | CPU |
| 2B | `step2b_audit.ipynb` | Leakage audit: five attackers, conformal sets per subscriber (LAT age and gender, Copenhagen gender) | CPU |
| 3 | `step3_release.ipynb` | Privacy-aware QUBO release selection against five other defences; privacy-utility curves; exact check of the solver | CPU |
| 4 | `step4_churn.ipynb` | The audit and the release selection on Iranian Churn (age group) and IBM Telco (gender, simulated: a negative control); downloads both tables | CPU |
| F | `stepF_figures.ipynb` | Draws every figure and table of the paper from the saved outputs of Steps 2B, 3 and 4, with 95 % intervals over the repeats and paired comparisons | CPU |

## Data

The datasets are downloaded by the notebooks and are not stored here. See `docs/datasets.md`.

## Running it

1. Zip this folder and upload it to Kaggle as a dataset named `agender-code`.
2. Import a notebook from `notebooks/` and add `agender-code` as an input. Step 1 also needs the saved
   output of Step 0, Steps 2B and 3 need the saved output of Step 1, and
   Step F needs the saved outputs of Steps 2B, 3 and 4 (without Step 4 it skips the churn part). Step 4 needs only `agender-code` and internet.
3. Use **Save Version -> Save & Run All**.

The notebooks write every result table and figure from code; no number is typed by hand. Settings are
chosen on the validation part only, and the test part is used once.

The methodology diagram and the graphical abstract are drawn by `python diagrams/draw_diagrams.py OUTPUT_FOLDER`.

Tests run locally without internet:

```
pip install -r requirements.txt
python -m pytest tests -q
```

## Layout

```
agender/     the Python package
notebooks/   one Kaggle notebook per step
splits/      fixed splits
tests/       offline tests
docs/        notes on each step and on the datasets
diagrams/    the methodology diagram and the graphical abstract
```

## Licence

The code is released under the MIT licence (`LICENSE`).
