from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .pipeline import load_chennai, load_melbourne, validate_daily_index

CHENNAI_URLS = [
    "https://raw.githubusercontent.com/anubhav199/Water-Level-Prediction-in-Chennai/master/chennai_reservoir_levels.csv",
]
MELBOURNE_URLS = [
    "https://data-melbournewater.opendata.arcgis.com/api/download/v1/items/ab54f1679c3d4dd6b0f4030634fce861/csv?layers=0",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "A-BLAST-MethodsX-Reproducibility/3.2"})
    return s


def download_first(urls: List[str], path: Path) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    sess = _session()
    for url in urls:
        try:
            with sess.get(url, timeout=120, stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                if path.stat().st_size < 100:
                    raise RuntimeError("downloaded file is unexpectedly small")
                return {
                    "url": url,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "etag": r.headers.get("ETag"),
                    "last_modified": r.headers.get("Last-Modified"),
                    "content_type": r.headers.get("Content-Type"),
                }
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
    raise RuntimeError("All download sources failed:\n" + "\n".join(errors))




def download_chennai_with_fallback(path: Path) -> Dict[str, object]:
    try:
        return download_first(CHENNAI_URLS, path)
    except Exception as first_error:
        api = "https://api.github.com/repos/anubhav199/Water-Level-Prediction-in-Chennai/contents/chennai_reservoir_levels.csv?ref=master"
        sess = _session()
        r = sess.get(api, timeout=60)
        r.raise_for_status()
        payload = r.json()
        dl = payload.get("download_url")
        if not dl:
            raise RuntimeError(f"Chennai direct download failed and GitHub API returned no download_url: {first_error}")
        meta = download_first([dl], path)
        meta["fallback_via"] = api
        return meta


def download_melbourne_with_fallback(path: Path) -> Dict[str, object]:
    try:
        return download_first(MELBOURNE_URLS, path)
    except Exception as first_error:
        # CKAN datastore fallback for the same DataVic resource.
        base = "https://discover.data.vic.gov.au/api/3/action/datastore_search"
        resource_id = "2f386033-f8c3-4e47-b206-29d433cc7e24"
        sess = _session()
        offset = 0
        limit = 10000
        rows = []
        fields = None
        while True:
            r = sess.get(base, params={"resource_id": resource_id, "limit": limit, "offset": offset}, timeout=120)
            r.raise_for_status()
            payload = r.json()
            if not payload.get("success"):
                raise RuntimeError(f"DataVic datastore fallback failed: {payload}")
            result = payload["result"]
            if fields is None:
                fields = [f["id"] for f in result.get("fields", [])]
            batch = result.get("records", [])
            rows.extend(batch)
            offset += len(batch)
            if not batch or offset >= int(result.get("total", offset)):
                break
        if not rows:
            raise RuntimeError(f"Melbourne direct download failed and CKAN fallback returned no rows: {first_error}")
        df = pd.DataFrame(rows)
        # Drop CKAN internal id only; retain supplied_ML and source columns.
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        df.to_csv(path, index=False)
        return {
            "url": f"{base}?resource_id={resource_id}",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "etag": None,
            "last_modified": None,
            "content_type": "application/json converted to CSV",
            "fallback_via": "DataVic CKAN datastore API",
        }

def dataset_profile(name: str, y: pd.Series, source_meta: Dict[str, object], stl_period: int) -> Dict[str, object]:
    index_check = validate_daily_index(y, name)
    if index_check["n_missing_calendar_days"] != 0:
        raise ValueError(f"{name}: expected daily contiguous observations; found {index_check['n_missing_calendar_days']} missing days")
    if name.lower() == "chennai":
        target = "Total reservoir water level (sum of POONDI, CHOLAVARAM, REDHILLS, CHEMBARAMBAKKAM)"
        unit = "million cubic feet (mcft)"
        source_label = "Kaggle Chennai Water Management dataset; reproducibility mirror: anubhav199/Water-Level-Prediction-in-Chennai"
    else:
        target = "Daily bulk water supply volume (supplied_ML)"
        unit = "megalitres (ML)"
        source_label = "Melbourne Water / DataVic official CSV resource"
    return {
        "dataset": name,
        "source_label": source_label,
        "source_url": source_meta["url"],
        "raw_path": source_meta["path"],
        "sha256": source_meta["sha256"],
        "size_bytes": source_meta["size_bytes"],
        "etag": source_meta.get("etag"),
        "last_modified": source_meta.get("last_modified"),
        "target_variable": target,
        "unit": unit,
        "start": index_check["start"],
        "end": index_check["end"],
        "n": int(len(y)),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)),
        "missing_target_values": int(y.isna().sum()),
        "missing_calendar_days": int(index_check["n_missing_calendar_days"]),
        "sampling_frequency": "Daily",
        "stl_period": int(stl_period),
        "missing_handling": "None; protocol aborts if target values or calendar dates are missing",
    }


def acquire_datasets(data_dir: Path, stl_period: int):
    data_dir.mkdir(parents=True, exist_ok=True)
    ch_path = data_dir / "chennai_reservoir_levels.csv"
    mel_path = data_dir / "melbourne_water_daily_supply.csv"
    ch_meta = download_chennai_with_fallback(ch_path)
    mel_meta = download_melbourne_with_fallback(mel_path)
    ch = load_chennai(str(ch_path))
    mel = load_melbourne(str(mel_path))
    profiles = {
        "Chennai": dataset_profile("Chennai", ch, ch_meta, stl_period),
        "Melbourne": dataset_profile("Melbourne", mel, mel_meta, stl_period),
    }
    with open(data_dir / "download_manifest.json", "w") as f:
        json.dump({"Chennai": ch_meta, "Melbourne": mel_meta}, f, indent=2)
    with open(data_dir / "dataset_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    return {"Chennai": ch, "Melbourne": mel}, profiles
