"""
plotting.py
-----------
Shared matplotlib plot functions used by all hub analysis notebooks.

Each function accepts pre-computed DataFrames plus style metadata, and
calls plt.show() at the end. Should work identically in each notebook.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.colouring import GOOGLE_PINK, HUB_BLACK, build_legend_entries

# Coverage heatmap
# TODO: sort models somehow? alphabetically or by coverage?

def plot_coverage_heatmap(
    loc_cov: pd.DataFrame,
    model_order: list[str],
    all_dates_hm: list,
    all_horizons: list[int],
    n_locs: int,
    hub_label: str = "",
    model_labels: dict | None = None,
    save_path: str | None = None,
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
    fig.suptitle(f"{title}\nGrey = no submission")

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
        ax.set_title(f"Horizon {h:.0f}")
        ax.set_xticks(range(len(all_dates_hm)))
        ax.set_xticklabels(date_labels, rotation=90)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(
            [model_labels.get(m, m) if model_labels else m for m in model_order],
        )
        ax.grid(False)

    for j in range(len(all_horizons), len(axes)):
        axes[j].set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes[:len(all_horizons)], shrink=0.5,
                     label="Proportion of locations submitted")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# Season-average bar plots

def plot_season_bars(
    summary: pd.DataFrame,
    model_colours: dict,
    model_hatches: dict | None = None,
    model_labels: dict | None = None,
    legend_entries: list | None = None,
    include_n_tasks: bool = True,
    bar_width: float = 0.8,
    inches_per_bar: float = 0.25,
    min_fig_width: float = 6.0,
    save_path: str | None = None,
) -> None:
    n = len(summary)

    fig_width = max(min_fig_width, n * inches_per_bar * 2)

    # Scale height and fonts for small-n plots
    fig_height = min(6.0, max(4.2, 0.14 * n + 3.4))
    small_plot = n <= 8

    title_fs = 11 if small_plot else 14
    label_fs = 10 if small_plot else 12
    tick_fs = 8 if small_plot else 9
    legend_fs = 9 if small_plot else 10
    task_fs = 6 if small_plot else 7

    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
    )

    for ax, metric, label in [
        (axes[1], "mean_wis",     "Mean WIS"),
        (axes[0], "mean_log_wis", "Mean log WIS"),
    ]:
        s = summary.sort_values(metric, ascending=True).reset_index(drop=True)
        x = np.arange(len(s))

        bar_colours = [model_colours.get(m, "0.5") for m in s["model_id"]]

        bars = ax.bar(
            x,
            s[metric],
            color=bar_colours,
            width=bar_width,
            align="center",
            edgecolor="white",
            linewidth=0.5,
        )

        if model_hatches is not None:
            for bar, model_id in zip(bars, s["model_id"]):
                bar.set_hatch(model_hatches.get(model_id, ""))

        ax.set_xticks(x)
        ax.set_xticklabels(
            [model_labels.get(m, m) if model_labels else m for m in s["model_id"]],
            rotation=45,
            ha="right",
            fontsize=tick_fs,
        )

        if include_n_tasks:
            ypad = 0.01 * s[metric].max()
            for bar, (_, row) in zip(bars, s.iterrows()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ypad,
                    f"n={row['n_tasks']:,}",
                    ha="center",
                    va="bottom",
                    fontsize=task_fs,
                )

        ax.set_ylabel(label, fontsize=label_fs)
        ax.set_title(f"Season-average {label}", fontsize=title_fs, pad=8)
        ax.set_ylim(0, s[metric].max() * 1.25)
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if legend_entries is None:
        legend_entries = build_legend_entries(model_colours, model_hatches)

    if legend_entries:
        handles = [
            Patch(facecolor=fc, hatch=h, edgecolor="white", label=lbl)
            for fc, h, lbl in legend_entries
        ]

        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 4 if small_plot else 6),
            bbox_to_anchor=(0.5, -0.08),
            fontsize=legend_fs,
        )

    # More bottom room for rotated tick labels and legend
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.34 if small_plot else 0.28,
        wspace=0.28,
    )

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# WIS vs log WIS scatter (not super informative -- removed from notebooks)

def plot_wis_vs_logwis(
    all_summary: pd.DataFrame,
    model_colours: dict,
    eligibility_threshold: float,
    save_path: str | None = None,
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
            xytext=(4, 4), textcoords="offset points",
            alpha=0.9 if elig else 0.55,
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="dimgray",
               markersize=9, label=f"Eligible  (≥{eligibility_threshold:.0%} tasks)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="dimgray",
               markersize=8, alpha=0.6, label="Ineligible"),
    ]
    ax.legend(handles=handles)
    ax.set_xlabel("Season-average WIS")
    ax.set_ylabel("Season-average log WIS")
    ax.set_title("WIS vs log WIS by model")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
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
    save_path: str | None = None,
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
        HUB_BLACK,
    )

    for ax, data, top_models, hub_present, ylabel, title in [
        (axes[1], by_hor_wis, top_by_wis, hub_in_wis, "Mean WIS",     "Mean WIS by horizon"),
        (axes[0], by_hor_log, top_by_log, hub_in_log, "Mean log WIS", "Mean log WIS by horizon"),
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

    n_shown = len(top_by_wis)
    plt.suptitle(f"{prefix}top {n_shown} eligible + hub models")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
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
    save_path: str | None = None,
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
        HUB_BLACK,
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)

    for ax, metric, ylabel, title in [
        (axes[1], "wis",     "Mean WIS",     "WIS"),
        (axes[0], "log_wis", "Mean log WIS", "log WIS"),
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
                alpha=1 if is_main else (1 if is_hub else 0.5),
                marker="o",
                markersize=6 if is_main else (5 if is_hub else 4),
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
        bbox_to_anchor=(0.5, -0.08),
    )
    plt.suptitle(f"{prefix}Weekly Mean Performance over Reference Dates")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
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
    save_path: str | None = None,
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

    if save_path:
        folder = os.path.dirname(save_path)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
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
    legend_entries: list | None = None,
    metric: str = "mean_log_wis",
    bar_width: float = 0.7,
    inches_per_bar: float = 0.25,
    save_path: str | None = None,
) -> None:
    """
    Single-figure vertical bar comparison across Flu, COVID-19 and RSV.

    Flu spans the full top row. COVID-19 and RSV are shown below, side by side,
    with width proportional to the number of models per infection. Models are
    sorted ascending by metric (lowest/best at left).

    Parameters
    ----------
    flu_summary, covid_summary, rsv_summary
        DataFrames with columns model_id and the chosen metric column, one row
        per model. Pre-filter to the desired model set (e.g. eligible + hub
        models) before passing. Sorted ascending by metric inside the function.
    flu_colours, covid_colours, rsv_colours
        dict mapping model_id → colour string.
    legend_entries : Optional list of (facecolor, hatch, label) tuples for a
                     shared figure legend.
    metric        : Column to plot — "mean_log_wis" (default) or "mean_wis".
    bar_width     : Fractional bar width within each column (0–1).
    inches_per_bar: Horizontal inches allocated per model column.
    """
    n_flu   = len(flu_summary)
    n_covid = len(covid_summary)
    n_rsv   = len(rsv_summary)
    n_total = n_flu + n_covid + n_rsv

    metric_label = "Mean log WIS" if "log" in metric else "Mean WIS"

    fig_width  = max(10, n_total * inches_per_bar + 2)
    fig_height = 10 + (5 if legend_entries else 0)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[n_covid, n_rsv],
    )

    axes = [
        fig.add_subplot(gs[0, :]),  # flu spans full width
        fig.add_subplot(gs[1, 0]),  # covid bottom-left
        fig.add_subplot(gs[1, 1]),  # rsv bottom-right
    ]

    for ax, summary, colours, hatches, label in [
        (axes[0], flu_summary,   flu_colours,   flu_hatches,   "FluSight Forecast Hub"),
        (axes[1], covid_summary, covid_colours, covid_hatches, "COVIDHub"),
        (axes[2], rsv_summary,   rsv_colours,   rsv_hatches,   "RSVHub"),
    ]:
        s = summary.sort_values(metric, ascending=True).reset_index(drop=True)
        n = len(s)
        x = np.arange(n)
        colors = [colours.get(m, "0.5") for m in s["model_id"]]

        bars = ax.bar(
            x, s[metric],
            color=colors, width=bar_width, align="center",
            edgecolor="white", linewidth=0.5,
        )

        if hatches is not None:
            for bar, model_id in zip(bars, s["model_id"]):
                bar.set_hatch(hatches.get(model_id, ""))

        ax.set_xticks(x)
        ax.set_xticklabels(
            [model_labels.get(m, m) if model_labels else m for m in s["model_id"]],
            rotation=45, ha="right",
        )
        ax.set_title(label, pad=5)
        ax.set_ylim(0, s[metric].max() * 1.15)
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(metric_label)
    axes[1].set_ylabel(metric_label)
    axes[2].set_ylabel("")

    if legend_entries is None:
        _all_colours = {**flu_colours, **covid_colours, **rsv_colours}
        _all_hatches = {**(flu_hatches or {}), **(covid_hatches or {}), **(rsv_hatches or {})}
        legend_entries = build_legend_entries(_all_colours, _all_hatches)
    if legend_entries:
        handles = [
            Patch(facecolor=fc, hatch=h, edgecolor="white", label=lbl)
            for fc, h, lbl in legend_entries
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 6),
            bbox_to_anchor=(0.5, -0.03),
        )

    fig.suptitle(f"Season-average {metric_label}")
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# Cross-hub relative WIS bar chart

def plot_crosshub_rel_bars(
    summaries: dict,
    colours: dict,
    hatches: dict | None = None,
    model_labels: dict | None = None,
    legend_entries: list | None = None,
    metric: str = "rel_log_wis",
    metric_label: str = "Relative log WIS", #   (< 1 = better than FluSight ensemble)
    title: str = "Relative log WIS vs CDC ensemble (on common tasks)",
    diseases: list | None = None,
    bar_width: float = 0.7,
    inches_per_bar: float = 0.25,
    save_path: str | None = None,
) -> None:
    """
    Vertical bar chart of a relative WIS metric across three diseases.

    Three side-by-side panels with width proportional to number of models per
    disease. A horizontal dashed reference line marks relative WIS = 1 (parity
    with baseline). Models are sorted ascending by metric (lowest/best at left).

    Parameters
    ----------
    summaries      : dict disease → DataFrame containing model_id and metric.
    colours        : dict disease → {model_id: colour}.
    hatches        : dict disease → {model_id: hatch pattern}.
    model_labels   : Optional {model_id: display_name} applied to x-tick labels.
    legend_entries : Optional list of (facecolor, hatch, label) tuples used to
                     build the shared figure legend.
    metric         : Column name to plot (default "rel_log_wis").
    metric_label   : y-axis label.
    title          : Figure suptitle.
    diseases       : Ordered list of three disease keys.
                     Defaults to the order of summaries.keys().
    bar_width      : Fractional bar width within each column (0–1).
    inches_per_bar : Physical inches allocated per model column.
    """
    if diseases is None:
        diseases = list(summaries.keys())
    d_0, d_1, d_2 = diseases[0], diseases[1], diseases[2]

    cleaned = {
        d: (summaries[d]
            .dropna(subset=[metric])
            .sort_values(metric, ascending=True)
            .reset_index(drop=True))
        for d in diseases
    }
    n = {d: len(cleaned[d]) for d in diseases}
    n_total = sum(n.values())

    fig_width  = max(10, n_total * inches_per_bar + 2)
    fig_height = 10 + (5 if legend_entries else 0)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[n[d_1], n[d_2]],
    )

    axes = [
        fig.add_subplot(gs[0, :]),  # flu spans full width
        fig.add_subplot(gs[1, 0]),  # covid bottom-left
        fig.add_subplot(gs[1, 1]),  # rsv bottom-right
    ]

    for ax, d in zip(axes, diseases):
        df  = cleaned[d]
        x   = np.arange(n[d])
        col = colours.get(d, {})
        hat = (hatches or {}).get(d, {})

        bar_colours = [col.get(m, "0.5") for m in df["model_id"]]
        bars = ax.bar(
            x, df[metric],
            color=bar_colours, width=bar_width, align="center",
            edgecolor="white", linewidth=0.5,
        )

        for bar, m in zip(bars, df["model_id"]):
            bar.set_hatch(hat.get(m, ""))
        
        if d in (d_0, d_1):
            ax.set_ylabel(metric_label)
        else:
            ax.set_ylabel("")

        ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--")
        ax.set_xticks(x)
        _lbl_d = (model_labels or {}).get(d)
        _lbl = _lbl_d if isinstance(_lbl_d, dict) else (model_labels or {})
        ax.set_xticklabels(
            [_lbl.get(m, m) for m in df["model_id"]],
            rotation=45, ha="right",
        )
        ax.set_ylim(0, df[metric].max() * 1.2)
        ax.set_ylabel(metric_label)
        ax.set_title(d.upper())
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if legend_entries is None:
        _all_colours = {m: c for d_c in colours.values() for m, c in d_c.items()}
        _all_hatches = {m: h for d_h in (hatches or {}).values() for m, h in d_h.items()}
        legend_entries = build_legend_entries(_all_colours, _all_hatches)
    if legend_entries:
        handles = [
            Patch(facecolor=fc, hatch=h, edgecolor="white", label=lbl)
            for fc, h, lbl in legend_entries
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 6),
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# Per-jurisdiction performance plot

def plot_by_location(
    by_loc_wis: pd.DataFrame,
    by_loc_log: pd.DataFrame,
    model_colours: dict,
    hub_models: set | None = None,
    top_n: int = 100,
    hub_label: str = "",
    hub_legend_label: str = "Hub models (ensemble + baseline)",
    main_model: str | None = None,
    main_model_colour: str | None = None,
    inches_per_row: float = 0.25,
    save_path: str | None = None,
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
    hub_legend_label  : Shared legend label for all hub models.
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
        (axes[1], by_loc_wis, "Mean WIS",     "Mean WIS by jurisdiction"),
        (axes[0], by_loc_log, "Mean log WIS", "Mean log WIS by jurisdiction"),
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
            size    = 55  if is_main else (30  if is_hub else 25)
            zorder  = 5   if is_main else (4   if is_hub else 2)
            alpha   = 1.0 if (is_main or is_hub) else 0.65

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

    hub_others = [m for m in hub_in if m != main_model]
    if hub_others:
        _hc = next((model_colours[m] for m in hub_others if m in model_colours), "black")
        legend_handles.append(Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=_hc,
            markersize=8,
            label=hub_legend_label,
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
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
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
    save_path: str | None = None,
) -> None:
    """
    Plot of each model's distribution of standardised log WIS ranks,
    inspired by Fig. 2 of Cramer et al. (2022, PNAS).

    For every task (reference_date × horizon × location), all models with
    a score are ranked by `score_col` (ascending; lower is better).
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

        x_eval = np.sort(np.unique(np.concatenate([
            np.linspace(0, 1, 300), [q25, q50, q75],
        ])))

        # KDE using Scott's rule
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

        # Ridge outline — closed polygon
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
        loc="upper center", bbox_to_anchor=(0.5, -0.45 / _ax_h),
        ncol=4, frameon=True,
    )

    metric_label = "log WIS" if score_col == "log_wis" else "WIS"
    prefix = f"{hub_label}: " if hub_label else ""
    ax.set_title(
        f"{prefix}Distribution of standardized {metric_label} ranks."
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
