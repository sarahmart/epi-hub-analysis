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
from matplotlib.lines import Line2D


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
) -> None:
    """
    Two horizontal bar plots side-by-side: mean WIS (left) and mean log WIS
    (right), sorted best → top.

    Parameters
    ----------
    summary       : DataFrame with columns model_id, mean_wis, mean_log_wis,
                    n_tasks. Rows = eligible models only.
    model_colours : dict mapping model_id → matplotlib colour.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    for ax, metric, label in [
        (axes[0], "mean_wis",     "Mean WIS"),
        (axes[1], "mean_log_wis", "Mean log WIS"),
    ]:
        s = summary.sort_values(metric, ascending=False)
        bar_colours = [model_colours.get(m, "0.5") for m in s["model_id"]]

        bars = ax.barh(s["model_id"], s[metric], color=bar_colours)

        for bar, (_, row) in zip(bars, s.iterrows()):
            ax.text(
                bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"n={row['n_tasks']:,}", va="center", fontsize=7,
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

    prefix = f"{hub_label} — " if hub_label else ""
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
        (axes[0], by_hor_wis, top_by_wis, hub_in_wis, "Mean WIS",     f"{prefix}Mean WIS by horizon"),
        (axes[1], by_hor_log, top_by_log, hub_in_log, "Mean log WIS", f"{prefix}Mean log WIS by horizon"),
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
        ax.set_title(f"{title}  (top {top_n} eligible + hub models)")
        ax.set_xticks(all_horizons)
        ax.legend(handles=legend_handles, fontsize=7.5, loc="upper left")

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
    prefix = f"{hub_label} — " if hub_label else ""

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
        (axes[0], "wis",     "Mean WIS",     f"{prefix}Weekly mean WIS over reference dates"),
        (axes[1], "log_wis", "Mean log WIS", f"{prefix}Weekly mean log WIS over reference dates"),
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

    axes[1].tick_params(axis="x", rotation=40)

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
        fontsize=8,
        bbox_to_anchor=(0.5, -0.14),
    )
    plt.show()
