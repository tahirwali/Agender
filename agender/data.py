"""
Download and read the two datasets.

LAT is downloaded from PLOS. The Copenhagen Networks Study is taken from an attached Kaggle dataset
(an unchanged copy, checked against the MD5 sums figshare publishes), because figshare refuses
downloads from Kaggle machines; outside Kaggle it is downloaded from figshare directly.
"""
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

LAT_URL = ("https://journals.plos.org/plosone/article/file?type=supplementary"
           "&id=10.1371/journal.pone.0271714.s001")
LAT_MEMBER = "minimal-dataset/age-gender-pred-dataset.csv"

COPENHAGEN_ARTICLE = "https://api.figshare.com/v2/articles/7267433"
# MD5 sums as published by figshare for article 7267433
COPENHAGEN_MD5 = {
    "genders.csv": "eb033aae23fff859684a82d00ffd8099",
    "calls.csv": "50a7c5be85d7e3eaf5fabd267a87008d",
    "sms.csv": "3d24cef3717881b3d9086584332dd740",
    "fb_friends.csv": "b1f96bae3e79949ed4278e90d0753843",
}
COPENHAGEN_FILES = tuple(COPENHAGEN_MD5)


def file_hash(path, kind="sha256"):
    digest = hashlib.new(kind)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, timeout=120):
    """Download to a temporary name first, so a broken transfer never looks like a finished file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            for block in response.iter_content(1 << 20):
                f.write(block)
    partial.replace(target)
    return {"file": target.name, "url": url, "bytes": target.stat().st_size, "sha256": file_hash(target)}


# ---- LAT (Kuang et al. 2022, PLOS ONE, supporting file S1) ----

def fetch_lat(folder):
    return download(LAT_URL, Path(folder) / "lat_s1.zip")


def read_lat(path):
    """Read the LAT table from the PLOS zip, or from the csv inside it. Kaggle unpacks zip files when
    a notebook output is saved as a dataset, so later steps usually see the csv."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            data = z.read(LAT_MEMBER)
    else:
        data = path.read_bytes()
    # keep_default_na=False so that empty cells stay "" and we decide later what they mean
    return pd.read_csv(io.BytesIO(data), keep_default_na=False, na_values=[""])


# ---- Copenhagen Networks Study (Sapiezynski et al. 2019, figshare 7267433) ----

def copenhagen_links(timeout=60):
    response = requests.get(COPENHAGEN_ARTICLE, timeout=timeout)
    response.raise_for_status()
    return {f["name"]: f["download_url"] for f in response.json()["files"]}


def fetch_copenhagen(folder):
    links = copenhagen_links()
    missing = [name for name in COPENHAGEN_FILES if name not in links]
    if missing:
        raise RuntimeError(f"figshare no longer lists {missing}; files there now: {sorted(links)}")
    return [download(links[name], Path(folder) / "copenhagen" / name) for name in COPENHAGEN_FILES]


def find_copenhagen(root="/kaggle/input"):
    """Return {file name: path} for the Copenhagen files found under root (attached Kaggle dataset)."""
    found = {}
    root = Path(root)
    if root.exists():
        for path in root.rglob("*.csv"):
            if path.name in COPENHAGEN_MD5 and path.name not in found:
                found[path.name] = path
    return found


def copy_copenhagen(found, folder):
    """Copy the attached files into our data folder, refusing any file whose MD5 differs from figshare's."""
    if set(found) != set(COPENHAGEN_FILES):
        raise RuntimeError(f"attached Copenhagen files incomplete: found {sorted(found)}, need {list(COPENHAGEN_FILES)}")
    records = []
    for name, path in found.items():
        md5 = file_hash(path, "md5")
        if md5 != COPENHAGEN_MD5[name]:
            raise RuntimeError(f"{name}: MD5 {md5} differs from figshare's {COPENHAGEN_MD5[name]}")
        target = Path(folder) / "copenhagen" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records.append({"file": name, "source": str(path), "bytes": target.stat().st_size,
                        "md5": md5, "sha256": file_hash(target)})
    return records


def read_copenhagen(folder):
    paths = sorted((Path(folder) / "copenhagen").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no Copenhagen files in {Path(folder) / 'copenhagen'}")
    tables = {}
    for path in paths:
        table = pd.read_csv(path)
        # headers look like "# user,female"; drop the comment mark
        table.columns = [c.strip().lstrip("#").strip() for c in table.columns]
        tables[path.stem] = table
    return tables


def save_manifest(records, path):
    with open(path, "w") as f:
        json.dump(records, f, indent=1)
