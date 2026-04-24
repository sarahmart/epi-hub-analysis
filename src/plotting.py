"""
plotting.py
-----------
Shared matplotlib plot functions used by all hub analysis notebooks.

Each function accepts pre-computed DataFrames plus style metadata, and
calls plt.show() at the end. Should work identically in each notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.colouring import GOOGLE_PINK

# Coverage heatmap
# TODO: sort models somehow? alphabetically or by coverage?

def plot_coverage_heatmap(
    loc_cov: pd.DataFrame,
    model_order: list[str],
    all_dates_hm: list,
    all_horizons: list[int],
    n_locs: int,
    hub_label: str = "",
) -> None:
    """
    Heatmap of submission coverage: proportion of locations submitted per
    (model, reference week, horizon).

    Parameters
    ----------
    loc_cov      : DataFrame with columns model_id, reference_date, horizon,
                   prop_locs (already computed as nunique_locs / n_locs).
    model_order  : Model IDs in desired row order (top = most covered).
    all_dates_hm : Sorted list of reference_date Timestamps to show on x-axis.
    all_horizons : Sorted list of horizon ints for panel titles.
    n_locs       : Total number of locations (used only for labelling).
    hub_label    : Short string for the figure suptitle, e.g. "COVID".
    """
    date_labels = [d.strftime("%b %d") for d in all_dates_hm]
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("whitesmoke")

    ncols = 2
    nrows = int(np.ceil(len(all_horizons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 7 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    title = "Submission coverage: proportion of locations per (model, reference week)"
    if hub_label:
        title = f"{hub_label} — {title}"
    fig.suptitle(f"{title}\nGrey = no submission", fontsize=11)

    im = None
    for i, h in enumerate(all_horizons):
        ax = axes[i]
        sub = loc_cov[loc_cov["horizon"] == h]
        pivot = (
            sub.pivot(index="model_id", columns="reference_date", values="prop_locs")
            .reindex(index=model_order, columns=all_dates_hm)
        )
        im = ax.imshow(
            pivot.values, aspect="auto", vmin=0, vmax=1,
            cmap=cmap, interpolation="nearest",
        )
        ax.set_title(f"Horizon {h:.0f}", fontsize=11)
        ax.set_xticks(range(len(all_dates_hm)))
        ax.set_xticklabels(date_labels, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order, fontsize=8)
        ax.grid(False)

    for j in range(len(all_horizons), len(axes)):
        axes[j].set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes[:len(all_horizons)], shrink=0.5,
                     label="Proportion of locations submitted")
    plt.show()


# Season-average bar plots

def plot_season_bars(
    summary: pd.DataFrame,
    model_colours: dict,
    model_hatches: dict | None = None,
    model_labels: dict | None = None,
    include_n_tasks: bool = True,
    bar_height: float = 0.8,
    inches_per_bar: float = 0.32,
    min_fig_height: float = 3.5,
    top_bottom_pad: float = 0.35,
) -> None:
    n = len(summary)
    fig_height = max(min_fig_height, n * inches_per_bar)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, fig_height),
        constrained_layout=True,
    )

    y = np.arange(n)

    for ax, metric, label in [
        (axes[0], "mean_wis", "Mean WIS"),
        (axes[1], "mean_log_wis", "Mean log WIS"),
    ]:
        s = summary.sort_values(metric, ascending=True).reset_index(drop=True)
        bar_colours = [model_colours.get(m, "0.5") for m in s["model_id"]]

        bars = ax.barh(
            y,
            s[metric],
            color=bar_colours,
            height=bar_height,
            align="center",
        )

        if model_hatches is not None:
            for bar, model_id in zip(bars, s["model_id"]):
                bar.set_hatch(model_hatches.get(model_id, ""))

        ax.set_yticks(y)
        ax.set_yticklabels([model_labels.get(m, m) if model_labels else m for m in s["model_id"]])
        ax.set_ylim(n - 0.5 + top_bottom_pad, -0.5 - top_bottom_pad)

        if include_n_tasks:
            xpad = 0.01 * s[metric].max()
            for bar, (_, row) in zip(bars, s.iterrows()):
                ax.text(
                    bar.get_width() + xpad,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={row['n_tasks']:,}",
                    va="center",
                    fontsize=10,
                )

        ax.set_xlabel(label)
        ax.set_title(f"Season-average {label}")
        ax.set_xlim(0, s[metric].max() * 1.2)
        ax.grid(True, axis="x", alpha=0.4)

    plt.show()

# WIS vs log WIS scatter (not super informative -- removed from notebooks)

def plot_wis_vs_logwis(
    all_summary: pd.DataFrame,
    model_colours: dict,
    eligibility_threshold: float,
) -> None:
    """
    Scatter plot of season-average WIS vs log WIS for all models.

    Parameters
    ----------
    all_summary            : DataFrame with columns model_id, mean_wis,
                             mean_log_wis, is_eligible.
    model_colours          : dict mapping model_id → colour.
    eligibility_threshold  : float, used in the legend label.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for _, row in all_summary.iterrows():
        elig = row["is_eligible"]
        ax.scatter(
            row["mean_wis"], row["mean_log_wis"],
            color=model_colours[row["model_id"]],
            marker="o" if elig else "s",
            s=90 if elig else 55,
            alpha=1.0 if elig else 0.5,
            zorder=3,
        )
        ax.annotate(
            row["model_id"],
            (row["mean_wis"], row["mean_log_wis"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
            alpha=0.9 if elig else 0.55,
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="dimgray",
               markersize=9, label=f"Eligible  (≥{eligibility_threshold:.0%} tasks)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="dimgray",
               markersize=8, alpha=0.6, label="Ineligible"),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel("Season-average WIS")
    ax.set_ylabel("Season-average log WIS")
    ax.set_title("WIS vs log WIS by model")
    plt.tight_layout()
    plt.show()


# Performance by horizon 

def plot_by_horizon(
    by_hor_wis: pd.DataFrame,
    by_hor_log: pd.DataFrame,
    all_horizons: list[int],
    model_colours: dict,
    hub_models: set | None = None,
    top_n: int = 15,
    hub_label: str = "",
    main_model: str | None = None,
    main_model_colour: str | None = None,
) -> None:
    """
    Two-panel line plot: mean WIS by horizon (left) and mean log WIS (right).
    Hub models are always plotted; top_n counts only eligible non-hub models.

    Parameters
    ----------
    by_hor_wis        : DataFrame indexed by model_id, columns = horizon ints.
    by_hor_log        : Same but for log WIS.
    all_horizons      : List of horizon ints for x-axis ticks.
    model_colours     : dict mapping model_id -> colour.
    hub_models        : Set of hub model IDs.
    top_n             : Number of top eligible (non-hub) models to plot.
    hub_label         : Short hub name for titles.
    main_model        : Model ID to highlight separately in the plot/legend.
    main_model_colour : Optional override colour for main_model.
    """
    hub_set = hub_models or set()

    non_hub_wis = by_hor_wis[~by_hor_wis.index.isin(hub_set)]
    non_hub_log = by_hor_log[~by_hor_log.index.isin(hub_set)]
    top_by_wis = non_hub_wis.mean(axis=1).nsmallest(top_n).index.tolist()
    top_by_log = non_hub_log.mean(axis=1).nsmallest(top_n).index.tolist()

    hub_in_wis = [m for m in by_hor_wis.index if m in hub_set]
    hub_in_log = [m for m in by_hor_log.index if m in hub_set]

    prefix = f"{hub_label}: " if hub_label else ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Legend colours from inputs only
    submitted_colour = next(
        (
            model_colours[m]
            for m in model_colours
            if m != main_model and m not in hub_set
        ),
        "0.5",
    )
    main_colour = (
        main_model_colour
        if main_model_colour is not None
        else model_colours.get(main_model, submitted_colour)
    )
    hub_colour = next(
        (model_colours[m] for m in hub_set if m in model_colours),
        "black",
    )

    for ax, data, top_models, hub_present, ylabel, title in [
        (axes[0], by_hor_wis, top_by_wis, hub_in_wis, "Mean WIS",     "Mean WIS by horizon"),
        (axes[1], by_hor_log, top_by_log, hub_in_log, "Mean log WIS", "Mean log WIS by horizon"),
    ]:
        for m in top_models:
            is_main = (m == main_model)
            ax.plot(
                all_horizons, data.loc[m],
                marker="o",
                linewidth=2,
                color=main_colour if is_main else model_colours.get(m, submitted_colour),
                zorder=10 if is_main else 3,
            )

        for m in hub_present:
            if m not in top_models:
                is_main = (m == main_model)
                ax.plot(
                    all_horizons, data.loc[m],
                    marker="o",
                    linewidth=2,
                    color=main_colour if is_main else model_colours.get(m, hub_colour),
                    linestyle="--",
                    zorder=10 if is_main else 3,
                )

        legend_handles = [
            Line2D([0], [0], color=submitted_colour, lw=2, marker="o", label="Submitted models"),
        ]
        if main_model is not None:
            legend_handles.append(
                Line2D([0], [0], color=main_colour, lw=2, marker="o", label=main_model)
            )
        legend_handles.append(
            Line2D([0], [0], color=hub_colour, lw=2, marker="o", linestyle="--", label="Hub models")
        )

        ax.set_xlabel("Horizon (weeks ahead)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  ")
        ax.set_xticks(all_horizons)
        ax.legend(handles=legend_handles, loc="upper left")

    plt.suptitle(f"{prefix}top {top_n} eligible + hub models")
    plt.show()


# Weekly performance over time 

def plot_weekly_scores(
    weekly: pd.DataFrame,
    plot_models: list[str],
    model_colours: dict,
    hub_models: set,
    hub_label: str = "",
    main_model: str | None = None,
    main_model_colour: str | None = None,
) -> None:
    """
    Two-panel time series: weekly mean WIS (top) and mean log WIS (bottom),
    one line per model. Hub models are dashed and thicker.

    Parameters
    ----------
    weekly            : DataFrame with columns model_id, reference_date, wis, log_wis.
    plot_models       : Model IDs to include (eligible + hub models).
    model_colours     : dict mapping model_id -> colour.
    hub_models        : Set of hub model IDs.
    hub_label         : Short hub name for titles.
    main_model        : Model ID to highlight separately in the plot/legend.
    main_model_colour : Optional override colour for main_model.
    """
    prefix = f"{hub_label}: " if hub_label else ""

    # Legend colours from inputs only
    submitted_colour = next(
        (
            model_colours[m]
            for m in plot_models
            if m != main_model and m not in hub_models
        ),
        "0.5",
    )
    main_colour = (
        main_model_colour
        if main_model_colour is not None
        else model_colours.get(main_model, submitted_colour)
    )
    hub_colour = next(
        (model_colours[m] for m in plot_models if m in hub_models),
        "black",
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)

    for ax, metric, ylabel, title in [
        (axes[0], "wis",     "Mean WIS",     "WIS"),
        (axes[1], "log_wis", "Mean log WIS", "log WIS"),
    ]:
        for m in plot_models:
            sub = weekly[weekly["model_id"] == m].sort_values("reference_date")
            is_hub = m in hub_models
            is_main = m == main_model

            ax.plot(
                sub["reference_date"],
                sub[metric],
                color=main_colour if is_main else model_colours[m],
                linestyle="--" if is_hub else "-",
                linewidth=2.2 if is_hub else 1.5,
                zorder=10 if is_main else (3 if is_hub else 2),
                marker="o",
                markersize=5,
            )

        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[1].tick_params(axis="x", rotation=0)

    legend_handles = [
        Line2D(
            [0], [0],
            color=submitted_colour,
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=5,
            label="Submitted models",
        )
    ]

    if main_model is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                color=main_colour,
                linestyle="-",
                linewidth=1.5,
                marker="o",
                markersize=5,
                label=main_model,
            )
        )

    legend_handles.append(
        Line2D(
            [0], [0],
            color=hub_colour,
            linestyle="--",
            linewidth=2.2,
            marker="o",
            markersize=5,
            label="Hub models",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        # fontsize=12,
        bbox_to_anchor=(0.5, -0.14),
    )
    plt.suptitle(f"{prefix}Weekly Mean Performance over Reference Dates")
    plt.show()


# Forecast fan vs observed truth
## Overall season performance per state

def plot_forecast_fans(
    forecasts: pd.DataFrame,
    truth: pd.DataFrame,
    model_display: dict,
    hub_label: str = "",
    location_names: dict | None = None,
    locations: list[str] | None = None,
    exclude_locations: set[str] | None = None,
    ncols: int = 6,
    inner_ci: tuple[float, float] = (0.25, 0.75),
    outer_ci: tuple[float, float] = (0.025, 0.975),
    subplot_height: float = 2.5,
    subplot_width: float = 3.2,
    plot_every_n: int = 1,
) -> None:
    """
    Fan chart of quantile forecast intervals vs observed truth for every location.

    One subplot per location. For each reference date and each enabled model,
    draws an outer quantile band, an inner quantile band, and a median line.
    The true observations are shown as a continuous black line.

    Parameters
    ----------
    forecasts         : DataFrame with columns reference_date, horizon,
                        target_end_date, location, output_type, output_type_id,
                        value, model_id.
    truth             : DataFrame with columns target_end_date, location, value
                        (one observed value per date × location).
    model_display     : dict of {model_id: config_dict}. Config keys:
                          "enabled"    : bool  — plot this model? (default False)
                          "color"      : str   — fan/line colour (default "0.5")
                          "label"      : str   — legend entry (default = model_id)
                          "alpha_outer": float — opacity of outer CI band (default 0.15)
                          "alpha_inner": float — opacity of inner CI band (default 0.35)
                          "zorder"     : float — drawing order; higher = on top (default 2)
    hub_label         : Infection name (for titles & labels).
    location_names    : Optional dict mapping location code → display name.
    locations         : Ordered list of location codes to plot. Defaults to all
                        locations in truth (after applying exclude_locations),
                        sorted alphabetically by display name.
    exclude_locations : Set of location codes to omit from the default location
                        list. Defaults to {"US"} (national aggregate excluded).
                        Pass an empty set to include all locations.
                        Has no effect when locations is supplied explicitly.
    ncols             : Number of subplot columns in the grid.
    inner_ci          : (low, high) quantile pair for the inner (darker) CI band.
    outer_ci          : (low, high) quantile pair for the outer (lighter) CI band.
    subplot_height    : Height of each subplot in inches.
    subplot_width     : Width of each subplot in inches.
    plot_every_n      : Only draw fans for every nth reference date (sorted
                        chronologically). The observed truth line is always
                        shown in full. Default 1 (all dates).
    """
    enabled_models = [m for m, cfg in model_display.items() if cfg.get("enabled", False)]
    if not enabled_models:
        raise ValueError("No models enabled in model_display.")

    # Needed quantile levels (round to avoid floating-point comparison issues)
    needed_q = {round(q, 4) for q in {inner_ci[0], inner_ci[1], outer_ci[0], outer_ci[1], 0.5}}
    lo_out = round(outer_ci[0], 4)
    hi_out = round(outer_ci[1], 4)
    lo_in = round(inner_ci[0], 4)
    hi_in = round(inner_ci[1], 4)

    # Filter forecasts: quantile output type, enabled models, non-negative horizons only
    fc_q = forecasts[
        (forecasts["output_type"] == "quantile")
        & forecasts["model_id"].isin(enabled_models)
        & (forecasts["horizon"] >= 0)
    ].copy()
    fc_q["output_type_id"] = fc_q["output_type_id"].astype(float).round(4)
    fc_q = fc_q[fc_q["output_type_id"].isin(needed_q)]

    # Truth: one row per (date, location)
    truth_clean = (
        truth[["target_end_date", "location", "value"]]
        .drop_duplicates(["target_end_date", "location"])
    )

    # Determine location list (only applies when locations is not provided explicitly)
    _exclude = exclude_locations if exclude_locations is not None else {"US"}
    if locations is None:
        all_locs = [l for l in truth_clean["location"].unique() if l not in _exclude]

        def _sort_key(x: str) -> tuple:
            if x == "US":
                return ("", "")
            name = location_names.get(x, x) if location_names else x
            return ("a", name)

        locations = sorted(all_locs, key=_sort_key)

    nrows = int(np.ceil(len(locations) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(subplot_width * ncols, subplot_height * nrows),
        constrained_layout=True,
        sharex=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, loc in enumerate(locations):
        ax = axes_flat[idx]
        loc_name = location_names.get(loc, loc) if location_names else loc

        # Observed truth line
        loc_truth = truth_clean[truth_clean["location"] == loc].sort_values("target_end_date")
        if not loc_truth.empty:
            ax.plot(
                loc_truth["target_end_date"], loc_truth["value"],
                color="black", linewidth=1.5, marker="o", markersize=2.5, zorder=10,
            )

        # Forecast fans per model, per reference date
        for model_id in enabled_models:
            cfg = model_display[model_id]
            color = cfg.get("color", "0.5")
            alpha_outer = cfg.get("alpha_outer", 0.15)
            alpha_inner = cfg.get("alpha_inner", 0.35)
            zorder = float(cfg.get("zorder", 2))

            model_fc = fc_q[(fc_q["model_id"] == model_id) & (fc_q["location"] == loc)]
            if model_fc.empty:
                continue

            all_ref_dates = sorted(model_fc["reference_date"].unique())
            for ref_date in all_ref_dates[::plot_every_n]:
                chunk = model_fc[model_fc["reference_date"] == ref_date]
                pivot = (
                    chunk.pivot_table(
                        index="target_end_date",
                        columns="output_type_id",
                        values="value",
                        aggfunc="first",
                    )
                    .sort_index()
                )
                dates = pivot.index

                if lo_out in pivot.columns and hi_out in pivot.columns:
                    ax.fill_between(
                        dates, pivot[lo_out], pivot[hi_out],
                        alpha=alpha_outer, color=color, linewidth=0, zorder=zorder,
                    )
                if lo_in in pivot.columns and hi_in in pivot.columns:
                    ax.fill_between(
                        dates, pivot[lo_in], pivot[hi_in],
                        alpha=alpha_inner, color=color, linewidth=0, zorder=zorder,
                    )
                if 0.5 in pivot.columns:
                    ax.plot(
                        dates, pivot[0.5],
                        color=color, linewidth=1.0, marker="o", markersize=1.8,
                        zorder=zorder + 0.5,
                    )

        ax.set_title(loc_name, pad=2)
        ax.tick_params(axis="both")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(locations), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend: observed + one entry per enabled model
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.5, marker="o", markersize=5, label="Observed"),
    ]
    for model_id in enabled_models:
        cfg = model_display[model_id]
        legend_handles.append(
            Line2D(
                [0], [0],
                color=cfg.get("color", "0.5"), lw=2, marker="o", markersize=5,
                label=f"{cfg.get('label', model_id)}  (median + 50%/95% PI)",
            )
        )

    fig.supylabel(f"Incident Weekly {hub_label} Hospital Admissions")
    fig.supxlabel("Time (2025-26 season)")

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.03),
    )

    title = f"Incident weekly {hub_label} hospital admissions and GoogleSAI forecasts"
    fig.suptitle(title)

    plt.show()


# Combined season-average bar chart across all three infections

def plot_combined_season_bars(
    flu_summary: pd.DataFrame,
    covid_summary: pd.DataFrame,
    rsv_summary: pd.DataFrame,
    flu_colours: dict,
    covid_colours: dict,
    rsv_colours: dict,
    flu_hatches: dict | None = None,
    covid_hatches: dict | None = None,
    rsv_hatches: dict | None = None,
    model_labels: dict | None = None,
    metric: str = "mean_log_wis",
    bar_height: float = 0.7,
    inches_per_bar: float = 0.30,
    panel_gap: float = 1.5,
) -> None:
    """
    Single-figure horizontal bar comparison across Flu, COVID-19 and RSV.

    Flu occupies the full left column. COVID and RSV are stacked in the right
    column, separated by panel_gap inches. Every bar row has the same physical
    height (inches_per_bar) across all three panels.

    Parameters
    ----------
    flu_summary, covid_summary, rsv_summary
        DataFrames with columns model_id and the chosen metric column, one row
        per model. Pre-filter to the desired model set (e.g. eligible + hub
        models) before passing. Sorted ascending by metric inside the function.
    flu_colours, covid_colours, rsv_colours
        dict mapping model_id → colour string.
    metric        : Column to plot — "mean_log_wis" (default) or "mean_wis".
    bar_height    : Fractional bar height within each row (0–1).
    inches_per_bar: Vertical inches allocated per model row.
    panel_gap     : Inches of whitespace between the COVID-19 and RSV panels.
    """
    n_flu   = len(flu_summary)
    n_covid = len(covid_summary)
    n_rsv   = len(rsv_summary)

    metric_label = "Mean log WIS" if "log" in metric else "Mean WIS"

    _title_h = 0.75   # inches reserved for suptitle above plot area
    _xlab_h  = 0.55   # inches reserved for x-label / tick labels below axes

    # Content height: tallest column (bars only) + gap between COVID and RSV
    n_right = n_covid + n_rsv
    content_h = max(n_flu, n_right) * inches_per_bar + panel_gap
    fig_height = max(4.5, content_h + _title_h + _xlab_h)

    fig = plt.figure(figsize=(14, fig_height))

    # Column x-positions (figure fractions)
    # Reproduces GridSpec(left=0.16, right=0.97, wspace=0.5) geometry:
    #   col_w = (0.97 - 0.16) / 2.5;  gap = 0.5 * col_w
    _lft, _rgt = 0.16, 0.97
    _col_w = (_rgt - _lft) / 2.5          # ≈ 0.324
    _lc_l  = _lft                          # left column left edge
    _lc_w  = _col_w
    _rc_l  = _lft + 1.5 * _col_w          # right column left edge  ≈ 0.646
    _rc_w  = _rgt - _rc_l

    # Row y-positions (figure fractions, measured from bottom)
    _top = (fig_height - _title_h) / fig_height   # top of plot area

    # Flu: full left column, anchored at top
    _flu_h   = n_flu   * inches_per_bar / fig_height
    _flu_top = _top
    _flu_bot = _flu_top - _flu_h

    # COVID: right column, anchored at top
    _cov_h   = n_covid * inches_per_bar / fig_height
    _cov_top = _top - panel_gap / fig_height
    _cov_bot = _cov_top - _cov_h

    # RSV: below COVID, separated by panel_gap
    _rsv_h   = n_rsv   * inches_per_bar / fig_height
    _rsv_top = _cov_bot - panel_gap / fig_height
    _rsv_bot = _rsv_top - _rsv_h

    ax_flu   = fig.add_axes([_lc_l, _flu_bot, _lc_w, _flu_h])
    ax_covid = fig.add_axes([_rc_l, _cov_bot, _rc_w, _cov_h])
    ax_rsv   = fig.add_axes([_rc_l, _rsv_bot, _rc_w, _rsv_h])

    # Draw bars
    for ax, summary, colours, hatches, label in [
        (ax_flu,   flu_summary,   flu_colours,   flu_hatches,   "FluSight Forecast Hub"),
        (ax_covid, covid_summary, covid_colours, covid_hatches, "COVIDHub"),
        (ax_rsv,   rsv_summary,   rsv_colours,   rsv_hatches,   "RSVHub"),
    ]:
        s = summary.sort_values(metric, ascending=False).reset_index(drop=True)
        n = len(s)
        y = np.arange(n)
        colors = [colours.get(m, "0.5") for m in s["model_id"]]

        bars = ax.barh(y, s[metric], color=colors, height=bar_height, align="center")

        if hatches is not None:
            for bar, model_id in zip(bars, s["model_id"]):
                bar.set_hatch(hatches.get(model_id, ""))
        ax.set_yticks(y)
        ax.set_yticklabels([model_labels.get(m, m) if model_labels else m for m in s["model_id"]])
        ax.set_ylim(-0.75, n - 0.25)
        ax.set_title(label, pad=5)
        ax.set_xlim(0, s[metric].max() * 1.15)
        ax.set_xlabel(metric_label)
        ax.grid(True, axis="x", alpha=0.4)

    fig.suptitle(f"Season-average {metric_label}")
    plt.show()


# Cross-hub relative WIS bar chart

def plot_crosshub_rel_bars(
    summaries: dict,
    colours: dict,
    hatches: dict | None = None,
    model_labels: dict | None = None,
    legend_entries: list | None = None,
    metric: str = "rel_log_wis",
    metric_label: str = "Relative log WIS  (< 1 = better than baseline)",
    title: str = "Cross-hub: Relative log WIS vs CDC ensemble (on common tasks)",
    diseases: list | None = None,
    bar_height: float = 0.7,
    inches_per_bar: float = 0.25,
    panel_gap: float = 1.0,
) -> None:
    """
    Horizontal bar chart of a relative WIS metric across three diseases.

    Layout mirrors plot_combined_season_bars: the first disease occupies the
    full left column; the second and third are stacked in the right column,
    separated by panel_gap inches. Bar physical height is equal across panels.

    Parameters
    ----------
    summaries      : dict disease → DataFrame containing model_id and metric.
    colours        : dict disease → {model_id: colour}.
    hatches        : dict disease → {model_id: hatch pattern}.
    model_labels   : Optional {model_id: display_name} applied to y-tick labels.
    legend_entries : Optional list of (facecolor, hatch, label) tuples used to
                     build the shared figure legend. Edit labels freely here —
                     colours and hatches are set independently of the bars, so
                     entries that share a colour can carry distinct labels.
    metric         : Column name to plot (default "rel_log_wis").
    metric_label   : x-axis label.
    title          : Figure suptitle.
    diseases       : Ordered list of three disease keys — [left, top-right, bot-right].
                     Defaults to the order of summaries.keys().
    bar_height     : Fractional bar height within each row (0–1).
    inches_per_bar : Physical inches allocated per model row.
    panel_gap      : Vertical inches between the two right-column panels.
    """
    if diseases is None:
        diseases = list(summaries.keys())
    d_l, d_tr, d_br = diseases[0], diseases[1], diseases[2]

    cleaned = {
        d: (summaries[d]
            .dropna(subset=[metric])
            .sort_values(metric, ascending=False)
            .reset_index(drop=True))
        for d in diseases
    }
    n = {d: len(cleaned[d]) for d in diseases}

    _title_h = 0.75
    _xlab_h  = 1.1 if legend_entries else 0.55

    content_h  = max(n[d_l], n[d_tr] + n[d_br]) * inches_per_bar + panel_gap
    fig_height = max(4.5, content_h + _title_h + _xlab_h)

    fig = plt.figure(figsize=(14, fig_height))

    # Column geometry — identical to plot_combined_season_bars
    _lft, _rgt = 0.16, 0.97
    _col_w = (_rgt - _lft) / 2.5
    _lc_l, _lc_w = _lft, _col_w
    _rc_l, _rc_w = _lft + 1.5 * _col_w, _rgt - (_lft + 1.5 * _col_w)

    _top = (fig_height - _title_h) / fig_height

    _l_h    = n[d_l]  * inches_per_bar / fig_height
    _tr_h   = n[d_tr] * inches_per_bar / fig_height
    _br_h   = n[d_br] * inches_per_bar / fig_height
    _gap    = panel_gap / fig_height

    ax_l  = fig.add_axes([_lc_l, _top - _l_h,                           _lc_w, _l_h])
    ax_tr = fig.add_axes([_rc_l, _top - _gap - _tr_h,                  _rc_w, _tr_h])
    ax_br = fig.add_axes([_rc_l, _top - _gap - _tr_h - _gap - _br_h,   _rc_w, _br_h])

    for ax, d in [(ax_l, d_l), (ax_tr, d_tr), (ax_br, d_br)]:
        df  = cleaned[d]
        y   = np.arange(n[d])
        col = colours.get(d, {})
        hat = (hatches or {}).get(d, {})

        bar_colours = [col.get(m, "0.5") for m in df["model_id"]]
        bars = ax.barh(y, df[metric], color=bar_colours, height=bar_height, align="center")

        for bar, m in zip(bars, df["model_id"]):
            bar.set_hatch(hat.get(m, ""))

        ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [model_labels.get(m, m) if model_labels else m for m in df["model_id"]],
            fontsize=8,
        )
        ax.set_ylim(-0.75, n[d] - 0.25)
        ax.set_xlabel(metric_label, fontsize=9)
        ax.set_title(d.upper())
        ax.grid(True, axis="x", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if legend_entries:
        handles = [
            Patch(facecolor=fc, hatch=h, label=lbl)
            for fc, h, lbl in legend_entries
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 6),
            fontsize=9,
            bbox_to_anchor=(0.5, 0.08),
        )

    fig.suptitle(title)
    plt.show()


# Per-jurisdiction performance plot

def plot_by_location(
    by_loc_wis: pd.DataFrame,
    by_loc_log: pd.DataFrame,
    model_colours: dict,
    hub_models: set | None = None,
    top_n: int = 100,
    hub_label: str = "",
    main_model: str | None = None,
    main_model_colour: str | None = None,
    inches_per_row: float = 0.25,
) -> None:
    """
    Model performance per jurisdiction.
    One row per location, one dot per model. Locations sorted alphabetically
    (A at top).

    Parameters
    ----------
    by_loc_wis, by_loc_log
        DataFrames indexed by model_id, columns = location names or codes.
        Pre-filter to eligible + hub models before passing.
    model_colours     : dict mapping model_id → colour.
    hub_models        : Set of hub model IDs (always included, distinct style).
    top_n             : Max non-hub eligible models to include.
    hub_label         : Short hub name for the figure suptitle.
    main_model        : Model ID to emphasise (large diamond, distinct colour).
    main_model_colour : Optional colour override for main_model.
    inches_per_row    : Figure height per location row.
    """
    hub_set = hub_models or set()

    # Select top_n non-hub models by overall mean log WIS, plus all hub models
    non_hub_log = by_loc_log.loc[~by_loc_log.index.isin(hub_set)]
    top_eligible = non_hub_log.mean(axis=1).nsmallest(top_n).index.tolist()
    hub_in = [m for m in by_loc_log.index if m in hub_set]
    plot_models = list(dict.fromkeys(top_eligible + hub_in))
    if not plot_models:
        return

    _main_colour = (
        main_model_colour if main_model_colour is not None
        else model_colours.get(main_model, GOOGLE_PINK) if main_model else GOOGLE_PINK
    )

    # Alphabetical order
    sorted_locs = sorted(by_loc_log.columns, reverse=True)
    n_locs = len(sorted_locs)
    y_map = {loc: i for i, loc in enumerate(sorted_locs)}

    fig_height = max(5.0, n_locs * inches_per_row + 1.8)
    prefix = f"{hub_label}: " if hub_label else ""

    fig, axes = plt.subplots(
        1, 2, figsize=(13, fig_height),
        constrained_layout=True, sharey=True,
    )

    for ax, data, xlabel, panel_title in [
        (axes[0], by_loc_wis, "Mean WIS",     "Mean WIS by jurisdiction"),
        (axes[1], by_loc_log, "Mean log WIS", "Mean log WIS by jurisdiction"),
    ]:
        avail = [loc for loc in sorted_locs if loc in data.columns]
        sub = data.loc[[m for m in plot_models if m in data.index], avail]

        # Grey range lines (min → max across all models per location)
        for loc in avail:
            vals = sub[loc].dropna()
            if len(vals) >= 2:
                ax.hlines(
                    y_map[loc], vals.min(), vals.max(),
                    color="0.82", linewidth=0.7, zorder=1,
                )

        # Dots drawn back-to-front: other eligible → hub → main_model
        draw_order = (
            [m for m in top_eligible if m != main_model and m in sub.index]
            + [m for m in hub_in if m != main_model and m in sub.index]
            + ([main_model] if main_model and main_model in sub.index else [])
        )

        for m in draw_order:
            is_main = (m == main_model)
            is_hub  = (m in hub_set)
            color   = _main_colour if is_main else model_colours.get(m, "0.5")
            marker  = "D" if is_main else ("s" if is_hub else "o")
            size    = 55  if is_main else (35  if is_hub else 30)
            zorder  = 5   if is_main else (4   if is_hub else 2)
            alpha   = 1.0 if (is_main or is_hub) else 0.75

            valid = [
                (y_map[loc], sub.loc[m, loc])
                for loc in avail
                if not pd.isna(sub.loc[m, loc])
            ]
            if not valid:
                continue
            ys, xs = zip(*valid)
            ax.scatter(
                xs, ys, color=color, s=size, zorder=zorder,
                alpha=alpha, marker=marker,
                edgecolors="white" if is_main else "none",
                linewidths=0.5,
            )

        ax.set_xlabel(xlabel)
        ax.set_title(panel_title, pad=5)
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(range(n_locs))
    axes[0].set_yticklabels(sorted_locs)
    axes[0].set_ylim(-0.7, n_locs - 0.3)

    # Legend
    legend_handles = []
    if main_model:
        legend_handles.append(Line2D(
            [0], [0], marker="D", color="w", markerfacecolor=_main_colour,
            markersize=8, markeredgecolor="white", markeredgewidth=0.5,
            label=main_model,
        ))
    for m in hub_in:
        if m == main_model:
            continue
        legend_handles.append(Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=model_colours.get(m, "black"),
            markersize=8, label=m,
        ))
    others = [m for m in top_eligible if m != main_model]
    if others:
        _oc = next((model_colours[m] for m in others if m in model_colours), "0.5")
        legend_handles.append(Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=_oc,
            markersize=8, alpha=0.75,
            label=f"Top {min(top_n, len(others))} eligible models",
        ))

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 5),
            bbox_to_anchor=(0.5, -0.04),
        )

    fig.suptitle(f"{prefix}Model performance by jurisdiction")
    plt.show()


# Standardised rank distribution (from https://www.pnas.org/doi/10.1073/pnas.2113561119, Fig. 2)

def plot_rank_distribution(
    scores: pd.DataFrame,
    eligible_models: list[str],
    model_colours: dict,
    score_col: str = "log_wis",
    top_n: int = 10,
    main_model: str | None = None,
    hub_label: str = "",
    show_n_tasks: bool = True,
    kde_max_height: float = 1,
    inches_per_row: float = 0.55,
) -> None:
    """
    Plot of each model's distribution of standardised log WIS ranks,
    inspired by Fig. 2 of Cramer et al. (2022, PNAS).

    For every task (reference_date × horizon × location), all models with
    a score in that group are ranked by `score_col` (ascending; lower is better).
    The standardised rank maps rank 1 (best) → 1.0 and rank n (worst) → 0.0.
    The kernel density of each model's rank distribution is drawn as a filled
    ridgeline, coloured by quartile. Models are ordered by Q1 (descending) so
    those that most consistently avoid low ranks (bad) appear at the top.

    Parameters
    ----------
    scores          : Scores DataFrame with columns model_id, reference_date,
                      horizon, location, wis, log_wis.
    eligible_models : Model IDs to include.
    model_colours   : dict model_id → colour; used for main_model highlight.
    score_col       : Column to rank on — "log_wis" (default) or "wis".
    top_n           : Maximum number of models to display, selected by Q1 of
                      the standardised rank distribution (descending).
    main_model      : Model ID to highlight with a coloured outline and label.
    hub_label       : Short hub name for the figure title.
    show_n_tasks    : If True, annotate each ridge with the model's task count.
    kde_max_height  : Maximum ridge height in row units. Values > 1.0 produce
                      overlapping ridges (ridgeline effect).
    inches_per_row  : Figure height allocated per model row.
    """
    _Q_COLORS = ["#440154", "#31688e", "#7ebe1e", "#35b779"]  # Q1→Q4
    _main_color = model_colours.get(main_model, GOOGLE_PINK) if main_model else GOOGLE_PINK

    # Standardised ranks across ALL models present in scores
    # Ranking includes every model that submitted for each group
    _grp = ["reference_date", "horizon", "location"]
    ranked = scores[scores[score_col].notna()].copy()
    ranked["_raw"] = ranked.groupby(_grp)[score_col].rank(
        method="average", ascending=True
    )
    ranked["_n"] = ranked.groupby(_grp)["model_id"].transform("count")
    ranked = ranked[ranked["_n"] > 1].copy()  # need ≥2 models to define a rank
    ranked["std_rank"] = (ranked["_n"] - ranked["_raw"]) / (ranked["_n"] - 1)

    # Select top-N eligible models by Q1
    elig = ranked[ranked["model_id"].isin(eligible_models)]
    q1_series = (
        elig.groupby("model_id")["std_rank"]
        .quantile(0.25)
        .sort_values(ascending=False)
    )
    model_order = q1_series.index[:top_n].tolist()  # [best, …, worst]
    n_models = len(model_order)
    if n_models == 0:
        return

    elig_top = elig[elig["model_id"].isin(model_order)]
    ranks_map = {
        m: elig_top.loc[elig_top["model_id"] == m, "std_rank"].values
        for m in model_order
    }
    n_tasks_map = {m: len(v) for m, v in ranks_map.items()}

    # Draw ridgelines
    fig_height = max(3.5, n_models * inches_per_row + 1.6)
    fig, ax = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
    right_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    for i, model in enumerate(model_order):
        y_base = float(n_models - 1 - i)  # i=0 (best) at the top
        data = ranks_map[model]
        if len(data) < 2:
            continue

        # Quartile cuts from actual data
        q25, q50, q75 = np.percentile(data, [25, 50, 75])

        # Build x grid that includes exact quartile positions so band
        # boundaries are pixel-perfect with no gaps between colours
        x_eval = np.sort(np.unique(np.concatenate([
            np.linspace(0, 1, 300), [q25, q50, q75],
        ])))

        # KDE using Scott's rule, numpy only (no scipy dependency)
        h = max(1.06 * np.std(data) * len(data) ** -0.2, 1e-3)
        diff = (x_eval[:, None] - data[None, :]) / h
        y_raw = np.exp(-0.5 * diff ** 2).mean(axis=1) / (h * np.sqrt(2 * np.pi))
        if y_raw.max() == 0:
            continue
        y_kde = y_raw / y_raw.max() * kde_max_height + y_base

        # Quartile-coloured fill
        for x_lo, x_hi, color in [
            (0.0, q25, _Q_COLORS[0]),
            (q25, q50, _Q_COLORS[1]),
            (q50, q75, _Q_COLORS[2]),
            (q75, 1.0, _Q_COLORS[3]),
        ]:
            mask = (x_eval >= x_lo) & (x_eval <= x_hi)
            if mask.sum() > 1:
                ax.fill_between(
                    x_eval[mask], y_base, y_kde[mask],
                    color=color, linewidth=0, zorder=2,
                )

        # Ridge outline — closed polygon that drops back to y_base at both
        # edges so the curve doesn't float open at x=0 and x=1
        x_closed = np.concatenate([[0.0], x_eval, [1.0]])
        y_closed = np.concatenate([[y_base], y_kde, [y_base]])
        is_main = (model == main_model)
        ax.plot(
            x_closed, y_closed,
            color=_main_color if is_main else "black",
            lw=2.0 if is_main else 0.8,
            zorder=3 + int(is_main),
        )

        # White vertical line at the median
        med_idx = int(np.argmin(np.abs(x_eval - q50)))
        ax.plot(
            [q50, q50], [y_base, y_kde[med_idx]],
            color="white", lw=1.2, zorder=5,
        )

        # Thin baseline separator — drawn on top so it's visible through overlap
        ax.plot([0, 1], [y_base, y_base], color="black", lw=0.5, zorder=6)

        # n_tasks annotation just outside the right axis
        if show_n_tasks:
            ax.text(
                1.01, y_base + kde_max_height * 0.45,
                f"n={n_tasks_map[model]:,}",
                transform=right_trans,
                va="center", ha="left", color="0.4",
            )

    # 4. Axes
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_order[::-1])  # y=0 = worst, y=n-1 = best
    for tick, label in zip(ax.get_yticklabels(), model_order[::-1]):
        if label == main_model:
            tick.set_color(_main_color)
            tick.set_fontweight("bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_models - 1 + kde_max_height + 0.3)
    ax.set_xlabel("standardized rank")
    ax.set_ylabel("")
    ax.grid(False)
    ax.xaxis.grid(True, alpha=0.3)

    # Legend
    _q_labels = ["Q1 (bottom 25%)", "Q2 (25–50%)", "Q3 (50–75%)", "Q4 (top 25%)"]
    legend_handles = [
        Patch(facecolor=c, label=l) for c, l in zip(_Q_COLORS, _q_labels)
    ]
    _ax_h = max(2.0, n_models * inches_per_row)
    ax.legend(
        handles=legend_handles, title="Rank Quartile",
        loc="upper center", bbox_to_anchor=(0.5, -0.42 / _ax_h),
        ncol=4, frameon=True,
    )

    metric_label = "log WIS" if score_col == "log_wis" else "WIS"
    prefix = f"{hub_label}: " if hub_label else ""
    ax.set_title(
        f"{prefix}Distribution of standardized {metric_label} ranks."
        # f"Rank 1.0 = best {metric_label} for that location × horizon × week  |  "
        # f"Models ordered by Q1 (↑ = rarely ranked poorly)",
    )

    plt.show()
