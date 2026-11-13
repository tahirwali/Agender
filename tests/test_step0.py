"""Offline checks for Step 0 (data import and audit). No internet needed."""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender import audit, data


def small_lat():
    return pd.DataFrame({
        "AGE": [15, 22, 30, 40, 50, 63],
        "GENDER": ["F", "M", "F", "M", "F", "M"],
        "DEVICE_TYPE": ["Phone", None, "Phone", "Tablet", "Phone", None],
        "BANK_SESSIONS": [3, 0, 5, 7, 0, 2],
        "UBER_SESSIONS": [3, 0, 5, 7, 0, 9],
        "BANK_MB": [0.5, 0, 1.2, 0.3, 0, 0.1],
        "UBER_MB": [0.1, 0, 0.0, 0.2, 0, 0.4],
    })


def test_read_lat_keeps_empty_cells_as_missing(tmp_path):
    zip_path = tmp_path / "lat.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(data.LAT_MEMBER, "AGE,GENDER,DEVICE_TYPE\n30,F,Phone\n41,M,\n")
    table = data.read_lat(zip_path)
    assert table.shape == (2, 3)
    assert table["DEVICE_TYPE"].isna().sum() == 1


def test_age_groups_use_the_six_bins():
    groups = audit.age_groups(pd.Series([17, 18, 25, 26, 60, 61]))
    assert list(groups.astype(str)) == ["<18", "18-25", "18-25", "26-35", "46-60", "60+"]


def test_lat_audit_finds_zero_rows_and_copied_columns():
    report = audit.audit_lat(small_lat())
    assert report["rows"] == 6
    assert report["missing"] == {"DEVICE_TYPE": 2}
    assert report["all_sessions_zero"]["rows"] == 2
    assert report["all_sessions_zero"]["of_these_with_some_megabytes"] == 0
    pair = report["equal_pairs"][0]
    assert {pair["a"], pair["b"]} == {"BANK_SESSIONS", "UBER_SESSIONS"}
    assert pair["equal_share"] == 0.75          # 3 of the 4 rows where either is non-zero


def test_copenhagen_audit_counts_people_with_gender(tmp_path):
    folder = tmp_path / "copenhagen"
    folder.mkdir()
    (folder / "genders.csv").write_text("# user,female\n1,0\n2,1\n3,-1\n")
    (folder / "calls.csv").write_text("timestamp,caller,callee,duration\n10,1,2,30\n20,2,3,-1\n30,1,9,5\n")
    report = audit.audit_copenhagen(data.read_copenhagen(tmp_path))
    assert report["female_column"] == {"-1": 1, "0": 1, "1": 1}
    assert report["calls"]["people"] == 4
    assert report["calls"]["people_with_known_gender"] == 2
    assert report["calls"]["negative_durations"] == 1


def test_attached_copenhagen_files_are_checked_against_figshare(tmp_path, monkeypatch):
    attached = tmp_path / "input" / "copenhagen-networks-study"
    attached.mkdir(parents=True)
    for name in data.COPENHAGEN_FILES:
        (attached / name).write_text(f"content of {name}\n")
    fake_sums = {name: data.file_hash(attached / name, "md5") for name in data.COPENHAGEN_FILES}
    monkeypatch.setattr(data, "COPENHAGEN_MD5", fake_sums)

    found = data.find_copenhagen(tmp_path / "input")
    records = data.copy_copenhagen(found, tmp_path / "data")
    assert len(records) == 4 and (tmp_path / "data" / "copenhagen" / "calls.csv").exists()

    (attached / "calls.csv").write_text("changed\n")
    with pytest.raises(RuntimeError, match="differs"):
        data.copy_copenhagen(found, tmp_path / "data")


def test_reading_copenhagen_without_files_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.read_copenhagen(tmp_path)


def test_report_is_written(tmp_path):
    summary = {"lat": audit.audit_lat(small_lat()), "steps": [{"step": "a", "status": "ok", "message": ""}]}
    text = audit.write_report(summary, tmp_path / "report.txt")
    assert "LAT: 6 rows" in text and "a: ok" in text


def test_read_lat_also_reads_the_unpacked_csv(tmp_path):
    # Kaggle unpacks lat_s1.zip when the Step 0 output becomes a dataset
    csv = tmp_path / "age-gender-pred-dataset.csv"
    csv.write_text("AGE,GENDER,DEVICE_TYPE\n30,F,\n41,M,Phone\n", encoding="utf-8")
    table = data.read_lat(csv)
    assert table.shape == (2, 3) and table["DEVICE_TYPE"].isna().sum() == 1
