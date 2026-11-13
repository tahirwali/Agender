# Datasets

Neither dataset is stored in this repository. The notebooks download them (or read an attached copy) and
check them before use.

| Dataset | What it contains | Labels | Size | Source and licence |
|---|---|---|---|---|
| **LAT** (Kuang et al., PLOS ONE 2022, doi 10.1371/journal.pone.0271714, file S1) | Pre-paid subscribers of a Latin-American operator: payment and recharge, devices, data use (MB and sessions) of social network, Instagram, WhatsApp, bank, Uber and internet; made and received calls | age in years, gender (from a customer survey) | 11,034 subscribers, 29 attributes | Direct download from PLOS (`journal.pone.0271714.s001`, 435 KB zip), CC BY 4.0 |
| **Copenhagen Networks Study** (Sapiezynski et al., Sci. Data 2019, doi 10.1038/s41597-019-0325-x) | Phone calls (time, duration), SMS (time), Bluetooth contacts and Facebook friendships of about 700 students over 4 weeks | gender only (`genders.csv`) | about 700 people | figshare 7267433, MIT licence |

Figshare refuses downloads from Kaggle (HTTP 403). An unchanged copy of the Copenhagen files is therefore
attached as a Kaggle dataset, and Step 0 checks every file against the MD5 sums published on figshare.

## LAT at a glance

- 11,034 rows, 31 columns (AGE, GENDER and 29 attributes).
- Gender: 5,690 M, 5,344 F. Age: 53 distinct values.
- Missing: DEVICE_TYPE 1,782, DEVICE_CAPABILITY 1,651, DEVICE_AGE 1,870 empty cells.
- 913 rows have 0 in all six `*_SESSIONS` columns and no data use; the per-app session counts copy each other
  in many rows (see `docs/step1_splits.md` for how this is handled).
