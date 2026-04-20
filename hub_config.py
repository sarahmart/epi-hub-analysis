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
    ensemble_id: str = ""                     # model_id of the primary hub ensemble
    baseline_id: str = ""                     # model_id of the hub-generated baseline
    google_id: str = ""                       # model_id of the Google submission
    extra_ensemble_ids: tuple[str, ...] = ()  # additional hub-generated ensembles (for Flu)
    # Google internal hubs
    subtree_path: str | None = None           # path prefix within repo (for monorepo)
    model_output_dir: str = "model-output"    # directory name containing model submissions
    truth_source_hub_name: str | None = None  # score using this hub's truth data (internal hubs don't have truth data)

    # Derived paths

    @property
    def hub_model_ids(self) -> tuple[str, ...]:
        """All hub-managed model IDs: primary ensemble, baseline, and any extra ensembles."""
        return (self.ensemble_id, self.baseline_id) + self.extra_ensemble_ids

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
    google_id="Google_SAI-Ensemble",
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
    google_id="Google_SAI-RSVEns",
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
    baseline_id="FluSight-baseline", # there are also multiple baselines, currently not yet implemented
    google_id="Google_SAI-FluEns",
    extra_ensemble_ids=(
        "FluSight-HJudge_ensemble",
        "FluSight-lop_norm",
        "FluSight-trained_mean",
        # "FluSight-trained_median",
    ),
)

GOOGLE_COVID_HUB = HubConfig(
    name="google_covid",
    owner="google-research",
    repo="google-research",
    branch="master",
    target_name="wk inc covid hosp",
    season_start=COVID_HUB.season_start,
    season_end=COVID_HUB.season_end,
    subtree_path="epi_forecasts/covid_hub/model_output",
    model_output_dir="",
    truth_source_hub_name="covid",
)

GOOGLE_FLU_HUB = HubConfig(
    name="google_flu",
    owner="google-research",
    repo="google-research",
    branch="master",
    target_name="wk inc flu hosp",
    season_start=FLU_HUB.season_start,
    season_end=FLU_HUB.season_end,
    subtree_path="epi_forecasts/flu_hub/model_output",
    model_output_dir="",
    truth_source_hub_name="flu",
)

GOOGLE_RSV_HUB = HubConfig(
    name="google_rsv",
    owner="google-research",
    repo="google-research",
    branch="master",
    target_name="wk inc rsv hosp",
    season_start=RSV_HUB.season_start,
    season_end=RSV_HUB.season_end,
    subtree_path="epi_forecasts/rsv_hub/model_output",
    model_output_dir="",
    truth_source_hub_name="rsv",
)

HUBS: dict[str, HubConfig] = {
    "covid": COVID_HUB,
    "rsv": RSV_HUB,
    "flu": FLU_HUB,
    "google_covid": GOOGLE_COVID_HUB,
    "google_flu": GOOGLE_FLU_HUB,
    "google_rsv": GOOGLE_RSV_HUB,
}
