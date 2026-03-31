"""
load_hub.py
-----------
Loader for any Hubverse-format CDC forecast hub (Flu, COVID, RSV).
Downloads forecast submission CSVs and target-data truth files from GitHub,
saves them as parquet to the hub-specific data directory.

Usage (from root dir):
    python -m src.load_hub --hub covid
    python -m src.load_hub --hub rsv
    python -m src.load_hub --hub flu
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from hub_config import HUBS, HubConfig


def _api_tree_url(hub: HubConfig) -> str:
    return (
        f"https://api.github.com/repos/{hub.owner}/{hub.repo}"
        f"/git/trees/{hub.branch}?recursive=1"
    )


def _raw_base_url(hub: HubConfig) -> str:
    return f"https://raw.githubusercontent.com/{hub.owner}/{hub.repo}/{hub.branch}"


def get_repo_tree(hub: HubConfig) -> list[str]:
    """Return all file paths in the repo."""
    resp = requests.get(_api_tree_url(hub), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [item["path"] for item in data["tree"] if item["type"] == "blob"]


def raw_url(hub: HubConfig, path: str) -> str:
    return f"{_raw_base_url(hub)}/{path}"


def read_csv_from_github(hub: HubConfig, path: str) -> pd.DataFrame:
    url = raw_url(hub, path)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), dtype={"location": str})


def read_parquet_from_github(hub: HubConfig, path: str) -> pd.DataFrame:
    url = raw_url(hub, path)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def parse_reference_date_from_filename(path: str) -> pd.Timestamp | None:
    """
    Expected forecast file names look like:
    model-output/<model_id>/YYYY-MM-DD-<model_id>.csv
    """
    name = Path(path).name
    if len(name) < 10:
        return None
    date_str = name[:10]
    try:
        return pd.to_datetime(date_str, format="%Y-%m-%d")
    except ValueError:
        return None


def keep_forecast_file(hub: HubConfig, path: str) -> bool:
    if not path.startswith("model-output/"):
        return False
    if not path.endswith(".csv"):
        return False

    ref_date = parse_reference_date_from_filename(path)
    if ref_date is None:
        return False

    return hub.season_start <= ref_date <= hub.season_end


def find_target_data_files(paths: Iterable[str]) -> list[str]:
    return [
        p
        for p in paths
        if p.startswith("target-data/") and (p.endswith(".csv") or p.endswith(".parquet"))
    ]


def load_forecasts(hub: HubConfig, paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for i, path in enumerate(paths, start=1):
        print(f"[{i}/{len(paths)}] Reading forecast file: {path}")
        df = read_csv_from_github(hub, path)

        if "model_id" not in df.columns:
            parts = Path(path).parts
            if len(parts) >= 3:
                df["model_id"] = parts[1]

        if "reference_date" not in df.columns:
            ref_date = parse_reference_date_from_filename(path)
            df["reference_date"] = ref_date

        frames.append(df)

    if not frames:
        raise RuntimeError("No forecast files found in the specified date range.")

    out = pd.concat(frames, ignore_index=True)

    for col in ["reference_date", "target_end_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # Hubs with categorical forecasts (FluSight) have string output_type_id values
    # (e.g. "increase") mixed with numeric quantile levels (e.g. 0.01).
    # Cast to str so parquet serialisation has a uniform type.
    if "output_type_id" in out.columns:
        out["output_type_id"] = out["output_type_id"].astype(str)

    return out


def load_truth(hub: HubConfig, paths: list[str]) -> pd.DataFrame:
    csv_paths     = [p for p in paths if p.endswith(".csv")]
    parquet_paths = [p for p in paths if p.endswith(".parquet")]
    frames: list[pd.DataFrame] = []

    for i, path in enumerate(csv_paths, start=1):
        print(f"[{i}/{len(csv_paths)}] Reading target-data CSV: {path}")
        df = read_csv_from_github(hub, path)
        df["source_file"] = path
        frames.append(df)

    for i, path in enumerate(parquet_paths, start=1):
        print(f"[{i}/{len(parquet_paths)}] Reading target-data parquet: {path}")
        df = read_parquet_from_github(hub, path)
        df["source_file"] = path
        frames.append(df)

    if not frames:
        raise RuntimeError("No target-data files found.")

    out = pd.concat(frames, ignore_index=True)

    # Normalise to a consistent schema
    # Some hubs use "observation" instead of "value" --> fill row-by-row
    if "observation" in out.columns:
        if "value" not in out.columns:
            out = out.rename(columns={"observation": "value"})
        else:
            out["value"] = out["value"].fillna(out["observation"])

    # RSV oracle-output files store truth in "oracle_value" rather than "value".
    # When both files are concatenated, fill NaN "value" rows from "oracle_value".
    if "oracle_value" in out.columns:
        if "value" not in out.columns:
            out = out.rename(columns={"oracle_value": "value"})
        else:
            out["value"] = out["value"].fillna(out["oracle_value"])

    # Some hubs already have "target_end_date"; others only have "date".
    # When both are present (mixed oracle-output + time-series), fill NaN
    # target_end_date values row-by-row from "date" rather than renaming.
    if "date" in out.columns:
        if "target_end_date" not in out.columns:
            out = out.rename(columns={"date": "target_end_date"})
        else:
            out["target_end_date"] = out["target_end_date"].fillna(out["date"])
            out = out.drop(columns=["date"])

    for col in ["target_end_date", "reference_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    return out


def main(hub: HubConfig) -> None:
    hub.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hub: {hub.name} ({hub.owner}/{hub.repo})")
    print(f"Season: {hub.season_start.date()} → {hub.season_end.date()}")
    print("Fetching repo tree...")
    paths = get_repo_tree(hub)

    forecast_paths = [p for p in paths if keep_forecast_file(hub, p)]
    target_paths = find_target_data_files(paths)

    print(f"Found {len(forecast_paths)} forecast files in date window.")
    print(f"Found {len(target_paths)} target-data files.")

    forecasts = load_forecasts(hub, forecast_paths)
    truth = load_truth(hub, target_paths)

    print(f"Forecast rows: {len(forecasts):,}")
    print(f"Truth rows: {len(truth):,}")

    forecasts.to_parquet(hub.forecasts_path, index=False)
    truth.to_parquet(hub.truth_path, index=False)

    print(f"Saved forecasts to: {hub.forecasts_path}")
    print(f"Saved truth to: {hub.truth_path}")

    print("\nForecast columns:")
    print(sorted(forecasts.columns.tolist()))

    print("\nTruth columns:")
    print(sorted(truth.columns.tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download hub forecast + truth data.")
    parser.add_argument(
        "--hub",
        choices=list(HUBS),
        required=True,
        help="Which hub to load: covid | rsv | flu",
    )
    args = parser.parse_args()
    main(HUBS[args.hub])
