# Step 1 - cleaning rules and fixed splits

Notebook: `notebooks/step1_splits.ipynb` (CPU, under a minute). Inputs: `agender-code`, `agender-step0`.
Code: `agender/lat.py`, `agender/copenhagen.py`, `agender/splits.py`, `agender/step1.py`; tests in `tests/test_step1.py`.

## Threat model

An operator releases usage aggregates of all its subscribers to a partner. The partner knows age and gender of a
small group (for example from its own customer survey, as Kuang et al. did). It trains on that group and guesses
age and gender for everyone else. The training split plays the role of the partner's labelled group.

## LAT cleaning rules (fixed in advance, nothing fitted on data)

| Rule | Reason |
|---|---|
| Six per-app session counts -> one `SESSIONS` = the largest of the six | They copy each other (bank = Uber in 80 % of rows); a sum would count the same sessions several times |
| Drop `NEED_CUSTOMER_SERVICE`, keep `CUSTOMER_SERVICE` | Near copy (89 % agreement on the 92 non-zero rows); the count holds more |
| `DEVICE_AGE` "2014 Q2" -> years before the newest quarter in the file, plus a missing flag | Numeric, ordered |
| Missing `DEVICE_TYPE`, `DEVICE_CAPABILITY` -> category `unknown` | Missing may itself carry information |
| Keep the 913 subscribers with no data use; flag `NO_DATA_USE` | Zero sessions and zero megabytes together: real non-users, mostly older |
| `INTERNET_MB` is the partner's task, never a released attribute | Otherwise the partner's task is trivial |

Encodings that learn from data (category frequencies, scaling) are fitted later on the training split only.

## Copenhagen person table

582 students with known gender (452 male, 130 female) who appear in the call or SMS log. Features: calls made and
received, missed calls, call seconds, median call length, call contacts, days with calls, SMS sent and received,
SMS contacts, days with SMS, share of received SMS answered within one hour, Facebook friends. No time-of-day or
weekday features: the release does not say at what clock time the timestamps start.

## Splits

train 60 % / validation 10 % / calibration 10 % / test 20 %, stratified (LAT: gender x six age groups, 12 strata;
Copenhagen: gender), five repeats with the seeds 42, 7, 123, 2024 and 99.
LAT repeat sizes: train 6,620, validation 1,105, calibration 1,103, test 2,206.
Files: `splits/lat_splits.csv` (id = row in the PLOS S1 file), `splits/copenhagen_splits.csv` (id = user), each
with a `*_info.json` of counts per stratum and part. The notebook rebuilds them and checks they are identical.
