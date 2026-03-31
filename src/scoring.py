"""
scoring.py
----------
Compute per-task WIS / log-WIS / coverage scores for any CDC forecast hub.

Usage:
    python scoring.py --hub covid
    python scoring.py --hub rsv
    python scoring.py --hub flu
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hub_config import HUBS, HubConfig


def _wis(y: float, taus: np.ndarray, qvals: np.ndarray) -> float:
    """
    Weighted Interval Score (Bracher et al. 2021).

        WIS = 1/(K + 0.5) * [ 0.5*|y - m|  +  sum_k (alpha_k/2) * IS_{alpha_k} ]

    where IS_alpha(l, u, y) = (u - l)
                              + (2/alpha) * (l - y) * 1(y < l)
                              + (2/alpha) * (y - u) * 1(y > u)

    K and the interval set are derived from the quantiles actually present in
    the submission: for each lower_tau < 0.5, look for its symmetric partner
    (1 - lower_tau).

    Parameters
    ----------
    y     : observed value
    taus  : 1-D array of quantile levels (sorted ascending)
    qvals : corresponding quantile values
    """
    tau_to_q = dict(zip(taus.tolist(), qvals.tolist()))

    median = tau_to_q.get(0.5, np.nan)
    if np.isnan(median):
        return np.nan

    total = 0.5 * abs(y - median)
    k = 0

    for lower_tau, q_lower in sorted(tau_to_q.items()):
        if lower_tau >= 0.5:
            break
        upper_tau = round(1.0 - lower_tau, 10)  # avoid float precision drift
        q_upper = tau_to_q.get(upper_tau)
        if q_upper is None:
            continue
        alpha = 2.0 * lower_tau
        interval_score = (
            (q_upper - q_lower)
            + (2.0 / alpha) * (q_lower - y) * (y < q_lower)
            + (2.0 / alpha) * (y - q_upper) * (y > q_upper)
        )
        total += (alpha / 2.0) * interval_score
        k += 1

    return total / (k + 0.5)


def score_one_forecast(group: pd.DataFrame) -> pd.Series:
    """
    Score one forecast task: one model × reference_date × horizon × location.
    """
    y = float(group["truth_value"].iloc[0])
    log_y = np.log1p(y)

    qs = group[["output_type_id", "value"]].dropna().sort_values("output_type_id")
    taus = qs["output_type_id"].to_numpy(dtype=float)
    qvals = qs["value"].to_numpy(dtype=float)
    log_qvals = np.log1p(np.maximum(qvals, 0.0))

    tau_to_q = dict(zip(taus.tolist(), qvals.tolist()))

    # ---- WIS ----
    wis = _wis(y, taus, qvals)

    # ---- log WIS ----
    log_wis = _wis(log_y, taus, log_qvals)

    # ---- Absolute error of median ----
    median = tau_to_q.get(0.5, np.nan)
    ae_median = abs(y - median) if not np.isnan(median) else np.nan

    # ---- Interval coverage ----
    l50 = tau_to_q.get(0.25)
    u50 = tau_to_q.get(0.75)
    cov_50 = float(l50 <= y <= u50) if (l50 is not None and u50 is not None) else np.nan

    l95 = tau_to_q.get(0.025)
    u95 = tau_to_q.get(0.975)
    cov_95 = float(l95 <= y <= u95) if (l95 is not None and u95 is not None) else np.nan

    return pd.Series(
        {
            "truth_value": y,
            "n_quantiles": len(qs),
            "median": median,
            "ae_median": ae_median,
            "wis": wis,
            "log_wis": log_wis,
            "cov_50": cov_50,
            "cov_95": cov_95,
        }
    )


def pairwise_relative_wis(scores_df, baseline_model, score_col="wis"):
    """
    Pairwise relative WIS following Bracher et al. 2021.

    For each ordered pair of models (i, j):
      - A_ij = tasks where both i and j submitted (reference_date x horizon x location)
      - theta_ij = mean_score_i(A_ij) / mean_score_j(A_ij)

    For each model i:
      - theta_i = geometric mean of {theta_ij} across all j != i

    Final score = theta_i / theta_baseline  (so baseline model gets rel_wis = 1).

    Parameters
    ----------
    scores_df      : Per-task scores DataFrame with columns model_id,
                     reference_date, horizon, location, and score_col.
    baseline_model : Model ID to use as the scaling reference (rel_wis = 1).
    score_col      : Score column to use ('wis' or 'log_wis').

    Returns
    -------
    DataFrame with columns: model_id, rel_wis
    """
    task_key = ["reference_date", "horizon", "location"]
    models = scores_df["model_id"].unique()

    # Build per-model Series indexed by task tuple for fast intersection
    model_scores = {}
    for m, grp in scores_df.groupby("model_id"):
        model_scores[m] = grp.set_index(task_key)[score_col].dropna()

    # Compute pairwise log-ratios log(theta_ij) for each model i
    log_ratios = {m: [] for m in models}
    for i in models:
        si = model_scores[i]
        for j in models:
            if i == j:
                continue
            sj = model_scores[j]
            common = si.index.intersection(sj.index)
            if len(common) == 0:
                continue
            mean_i = si.loc[common].mean()
            mean_j = sj.loc[common].mean()
            if mean_i > 0 and mean_j > 0:
                log_ratios[i].append(np.log(mean_i / mean_j))

    # Geometric mean across all opponents
    geo_means = {}
    for m in models:
        if log_ratios[m]:
            geo_means[m] = np.exp(np.mean(log_ratios[m]))
        else:
            geo_means[m] = float("nan")

    baseline_geo = geo_means.get(baseline_model, float("nan"))

    return pd.DataFrame(
        [{"model_id": m, "rel_wis": v / baseline_geo} for m, v in geo_means.items()]
    )


def score_hub(hub: HubConfig) -> None:
    print(f"Scoring hub: {hub.name} (target: {hub.target_name})")

    forecasts = pd.read_parquet(hub.forecasts_path)
    truth = pd.read_parquet(hub.truth_path)

    # Hospital admissions, quantile output only
    forecasts = forecasts.loc[
        (forecasts["target"] == hub.target_name)
        & (forecasts["output_type"] == "quantile")
    ].copy()

    # Normalise truth
    # Mixed-source truth files need row-by-row column fills.
    if "date" in truth.columns:
        if "target_end_date" not in truth.columns:
            truth = truth.rename(columns={"date": "target_end_date"})
        elif truth["target_end_date"].isna().any():
            truth["target_end_date"] = truth["target_end_date"].fillna(truth["date"])
    if "oracle_value" in truth.columns:
        truth["value"] = truth["value"].fillna(truth["oracle_value"])
    if "observation" in truth.columns:
        truth["value"] = truth["value"].fillna(truth["observation"])

    # Filter to the relevant target when truth covers multiple targets (RSV).
    if "target" in truth.columns:
        truth = truth[truth["target"] == hub.target_name].copy()

    # Deduplicate to one truth row per (location, target_end_date)
    # oracle-output files contain one row per output_type for the same observation.
    truth = truth.drop_duplicates(subset=["location", "target_end_date"])

    truth = truth.rename(columns={"value": "truth_value"})
    merged = forecasts.merge(
        truth[["location", "target_end_date", "truth_value"]],
        on=["location", "target_end_date"],
        how="inner",
    )

    print(f"Forecast rows after target filter: {len(forecasts):,}")
    print(f"Rows after joining to truth: {len(merged):,}")

    group_cols = [
        "model_id",
        "reference_date",
        "horizon",
        "target_end_date",
        "location",
        "target",
    ]

    scores = (
        merged.groupby(group_cols, as_index=False, sort=False)
        .apply(score_one_forecast)
        .reset_index(drop=True)
    )

    hub.data_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(hub.scores_path, index=False)

    print(f"\nSaved scores to {hub.scores_path}")
    print("\nScore columns:")
    print(scores.columns.tolist())

    print("\nTop models by WIS:")
    summary = (
        scores.groupby("model_id", as_index=False)
        .agg(
            n_tasks=("wis", "size"),
            mean_wis=("wis", "mean"),
            mean_log_wis=("log_wis", "mean"),
            mean_ae_median=("ae_median", "mean"),
            cov_50=("cov_50", "mean"),
            cov_95=("cov_95", "mean"),
        )
        .sort_values("mean_wis")
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score hub forecast submissions.")
    parser.add_argument(
        "--hub",
        choices=list(HUBS),
        required=True,
        help="Which hub to score: covid | rsv | flu",
    )
    args = parser.parse_args()
    score_hub(HUBS[args.hub])
