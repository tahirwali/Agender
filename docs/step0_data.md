# Step 0 - download the datasets and check them

Notebook: `notebooks/step0_data.ipynb` (CPU, internet on, 1-3 minutes).
Code: `agender/data.py` (download and read), `agender/audit.py` (checks), tests in `tests/test_step0.py`.

## What it does

1. Downloads LAT from PLOS (`journal.pone.0271714.s001`) and the Copenhagen Networks Study files from figshare
   (article 7267433). Each download is written under a temporary name first and its SHA-256 is recorded.
2. Counts and describes, without changing anything:
   - missing cells per column, duplicate rows;
   - age and gender distributions, also in the six age groups used for the age task;
   - rows where every session count is zero, and whether those people used any megabytes;
   - non-zero values that repeat far more often than a continuous measure should (possible fill values);
   - pairs of columns that hold the same number in most rows (possible copies).
3. Writes `step0_summary.json`, `step0_report.txt`, `steps.json`, and keeps the downloaded files in `data/`.

## Output

`/kaggle/working/agender_outputs/step0/`. Save it as the Kaggle dataset `agender-step0` for the next steps.
