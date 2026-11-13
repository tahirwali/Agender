# Fixed splits

- `lat_splits.csv`: one row per LAT subscriber; `id` is the row number in the PLOS S1 file
  (`minimal-dataset/age-gender-pred-dataset.csv`, first data row = 0); `repeat_0` ... `repeat_4` give the part
  (train, validation, calibration, test) for the seeds 42, 7, 123, 2024, 99.
- `copenhagen_splits.csv`: the same for the 582 Copenhagen students with known gender; `id` is the user number.
- `*_info.json`: counts per stratum and part, and the SHA-256 of the LAT source file.

Made by `agender/step1.py`; see `docs/step1_splits.md`. LAT is published with Kuang et al. (2022), PLOS ONE,
under CC BY 4.0; the Copenhagen Networks Study data is MIT-licensed.
