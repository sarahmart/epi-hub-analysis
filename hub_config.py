from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class HubConfig:
    """All hub-specific parameters."""

    name: str              # short key used in directory names: "covid" / "rsv" / "flu"
    owner: str             # GitHub organisation
    repo: str              # GitHub repository name
    branch: str            # branch to pull from (usually "main")
    target_name: str       # target string in the forecasts, e.g. "wk inc covid hosp"
    season_start: pd.Timestamp
    season_end: pd.Timestamp
    ensemble_id: str       # model_id of the hub-generated ensemble
    baseline_id: str       # model_id of the hub-generated baseline

    # Derived paths

    @property
    def data_dir(self) -> Path:
        return Path("data/processed") / self.name

    @property
    def forecasts_path(self) -> Path:
        return self.data_dir / "forecasts.parquet"

    @property
    def truth_path(self) -> Path:
        return self.data_dir / "truth.parquet"

    @property
    def scores_path(self) -> Path:
        return self.data_dir / "scores.parquet"

    @property
    def outputs_dir(self) -> Path:
        return Path("outputs") / self.name

    @property
    def label(self) -> str:
        """Human-readable hub label for plot titles."""
        return self.name.upper()


# Hub instances
# Update season_start / season_end here for current analysis window. 

COVID_HUB = HubConfig(
    name="covid",
    owner="CDCgov",
    repo="covid19-forecast-hub",
    branch="main",
    target_name="wk inc covid hosp",
    season_start=pd.Timestamp("2025-12-10"),
    season_end=pd.Timestamp("2026-05-31"),
    ensemble_id="CovidHub-ensemble",
    baseline_id="CovidHub-baseline",
)

RSV_HUB = HubConfig(
    name="rsv",
    owner="CDCgov",
    repo="rsv-forecast-hub",
    branch="main",
    target_name="wk inc rsv hosp",
    season_start=pd.Timestamp("2026-01-01"),
    season_end=pd.Timestamp("2026-05-31"),
    ensemble_id="RSVHub-ensemble",
    baseline_id="RSVHub-baseline",
)

FLU_HUB = HubConfig(
    name="flu",
    owner="cdcepi",
    repo="FluSight-forecast-hub",
    branch="main",
    target_name="wk inc flu hosp",
    season_start=pd.Timestamp("2025-11-22"),
    season_end=pd.Timestamp("2026-05-31"),
    ensemble_id="FluSight-ensemble",
    baseline_id="FluSight-baseline",
)

HUBS: dict[str, HubConfig] = {
    "covid": COVID_HUB,
    "rsv": RSV_HUB,
    "flu": FLU_HUB,
}
