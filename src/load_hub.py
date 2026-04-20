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


def _get_subtree_paths(hub: HubConfig) -> list[str]:
    """
    For monorepos (Google internal): fetch only files within hub.subtree_path.

    Uses the GitHub Contents API to resolve the SHA of the subtree directory,
    then fetches its recursive tree.  Returns paths with the subtree prefix
    prepended so that raw_url() works.
    """
    # Fetch the parent directory listing to get the SHA of the target subdirectory.
    parent = hub.subtree_path.rsplit("/", 1)[0]
    child_name = hub.subtree_path.rsplit("/", 1)[1]
    parent_url = (
        f"https://api.github.com/repos/{hub.owner}/{hub.repo}"
        f"/contents/{parent}?ref={hub.branch}"
    )
    resp = requests.get(parent_url, timeout=60)
    resp.raise_for_status()
    entries = resp.json()
    subtree_sha = next(
        e["sha"] for e in entries if e["name"] == child_name and e["type"] == "dir"
    )
    tree_url = (
        f"https://api.github.com/repos/{hub.owner}/{hub.repo}"
        f"/git/trees/{subtree_sha}?recursive=1"
    )
    resp2 = requests.get(tree_url, timeout=60)
    resp2.raise_for_status()
    data = resp2.json()
    prefix = hub.subtree_path + "/"
    return [prefix + item["path"] for item in data["tree"] if item["type"] == "blob"]


def get_repo_tree(hub: HubConfig) -> list[str]:
    """Return all file paths in the repo (or subtree for monorepos)."""
    if hub.subtree_path:
        return _get_subtree_paths(hub)
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
    if hub.model_output_dir == "":
        # Google internal hub: paths prefixed with subtree_path
        if hub.subtree_path and not path.startswith(hub.subtree_path + "/"):
            return False
    else:
        if not path.startswith(hub.model_output_dir + "/"):
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
            if hub.model_output_dir == "" and hub.subtree_path:
                # Google hub: model_id right after subtree prefix
                subtree_depth = len(Path(hub.subtree_path).parts)
                if len(parts) > subtree_depth:
                    df["model_id"] = parts[subtree_depth]
            elif len(parts) >= 3:
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


def _manifest_path(data_dir: Path, kind: str) -> Path:
    return data_dir / f"{kind}_manifest.txt"


def _load_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(path.read_text().splitlines())


def _save_manifest(path: Path, entries: set[str]) -> None:
    path.write_text("\n".join(sorted(entries)) + "\n")


def _bootstrap_forecast_manifest(hub: HubConfig, forecast_paths: list[str]) -> set[str]:
    """Infer which forecast files are already in the parquet via (reference_date, model_id) pairs."""
    if not hub.forecasts_path.exists():
        return set()
    existing = pd.read_parquet(hub.forecasts_path, columns=["reference_date", "model_id"])
    loaded = set(
        zip(
            pd.to_datetime(existing["reference_date"]).dt.strftime("%Y-%m-%d"),
            existing["model_id"].astype(str),
        )
    )
    result = set()
    for path in forecast_paths:
        ref_date = parse_reference_date_from_filename(path)
        parts = Path(path).parts
        if hub.model_output_dir == "" and hub.subtree_path:
            subtree_depth = len(Path(hub.subtree_path).parts)
            model_id = parts[subtree_depth] if len(parts) > subtree_depth else None
        else:
            model_id = parts[1] if len(parts) >= 2 else None
        if ref_date is not None and model_id is not None:
            if (ref_date.strftime("%Y-%m-%d"), model_id) in loaded:
                result.add(path)
    return result


def main(hub: HubConfig, incremental: bool = False) -> None:
    hub.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hub: {hub.name} ({hub.owner}/{hub.repo})")
    print(f"Season: {hub.season_start.date()} → {hub.season_end.date()}")
    print("Fetching repo tree...")
    paths = get_repo_tree(hub)

    forecast_paths = [p for p in paths if keep_forecast_file(hub, p)]

    print(f"Found {len(forecast_paths)} forecast files in date window.")

    if hub.truth_source_hub_name:
        print(f"Truth data will be loaded from CDC hub '{hub.truth_source_hub_name}' during scoring.")
        target_paths: list[str] = []
    else:
        target_paths = find_target_data_files(paths)
        print(f"Found {len(target_paths)} target-data files.")

    if incremental:
        fcast_mpath = _manifest_path(hub.data_dir, "forecasts")

        if not fcast_mpath.exists() and hub.forecasts_path.exists():
            print("No forecast manifest found; bootstrapping from existing parquet...")
            forecast_manifest = _bootstrap_forecast_manifest(hub, forecast_paths)
            _save_manifest(fcast_mpath, forecast_manifest)
            print(f"Bootstrapped {len(forecast_manifest)} forecast files into manifest.")
        else:
            forecast_manifest = _load_manifest(fcast_mpath)

        new_forecast_paths = [p for p in forecast_paths if p not in forecast_manifest]
        new_target_paths = target_paths  # always refresh target data
        print(f"Incremental mode: {len(new_forecast_paths)} new forecast files, {len(new_target_paths)} target-data files (target data always reloaded).")
    else:
        forecast_manifest = set()
        new_forecast_paths = forecast_paths
        new_target_paths = target_paths

    if new_forecast_paths:
        forecasts = load_forecasts(hub, new_forecast_paths)
        if incremental and hub.forecasts_path.exists():
            existing = pd.read_parquet(hub.forecasts_path)
            if "output_type_id" in existing.columns:
                existing["output_type_id"] = existing["output_type_id"].astype(str)
            forecasts = pd.concat([existing, forecasts], ignore_index=True)
        print(f"Forecast rows: {len(forecasts):,}")
        forecasts.to_parquet(hub.forecasts_path, index=False)
        _save_manifest(_manifest_path(hub.data_dir, "forecasts"), forecast_manifest | set(new_forecast_paths))
        print(f"Saved forecasts to: {hub.forecasts_path}")
        print("\nForecast columns:")
        print(sorted(forecasts.columns.tolist()))
    else:
        print("No new forecast files to load.")

    if new_target_paths:
        truth = load_truth(hub, new_target_paths)
        print(f"Truth rows: {len(truth):,}")
        truth.to_parquet(hub.truth_path, index=False)
        print(f"Saved truth to: {hub.truth_path}")
        print("\nTruth columns:")
        print(sorted(truth.columns.tolist()))
    else:
        print("No new target data files to load.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download hub forecast + truth data.")
    parser.add_argument(
        "--hub",
        choices=list(HUBS),
        required=True,
        help="Which hub to load: covid | rsv | flu",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help="Only load files not already downloaded; merge with existing parquet.",
    )
    args = parser.parse_args()
    main(HUBS[args.hub], incremental=args.incremental)
