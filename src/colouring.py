import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

"""
OVERALL HUB APPEARANCES
* Google_SAI ensemble : darker blue
* Google internal models: 
    * Adapted : solid blue
    * Hybrid  : diagonal blue //
    * Novel   : cross-hatch blue (both diagonals)
* Hub-generated models (ensembles and baselines): red diagonal hatch //
* Hub-submitting models : red solid
"""


# HUB COLOURS — match previous paper colour + hatching scheme
HUB = "#c0392b"              # models from external hubs (submitting and baselines)
GOOGLE_INTERNAL = "#2471a3"  # Google internal models
SAI_ENSEMBLE = "#011773"     # main SAI ensemble

GREEN = "#17b248"
GOOGLE_PINK = SAI_ENSEMBLE   # main SAI hub submission; alias kept for notebook compatibility
HUB_BLACK   = HUB            # kept for notebook compatibility
OTHER_GREY  = "#c0392b"

HUB_HATCH = "//"             # hatch pattern for hub-generated models (ensembles / baselines)


def median_gradient(col, cmap="RdYlGn_r"):
    """Per-column colour scale where the column median maps to the midpoint (yellow).
    Text colour adapts to background luminance for readability."""
    cm = plt.get_cmap(cmap)
    med = col.median()
    lo, hi = col.min(), col.max()

    def norm(v):
        if v <= med:
            return 0.0 if med == lo else 0.5 * (v - lo) / (med - lo)
        else:
            return 1.0 if hi == med else 0.5 + 0.5 * (v - med) / (hi - med)

    def luminance(r, g, b):
        def f(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

    styles = []
    for v in col:
        if pd.isna(v):
            bg = "#D6D6D6"
            fg = "#000000"
        else:
            rgba = cm(norm(v))
            bg = mcolors.to_hex(rgba)
            fg = "#000000" if luminance(rgba[0], rgba[1], rgba[2]) > 0.179 else "#f1f1f1"
        styles.append(f"background-color: {bg}; color: {fg}")
    return styles


# INTERNAL HUB ANALYSIS — Google model types, all GOOGLE_INTERNAL blue, differentiated by hatch
TYPE_COLOURS = {
    "Adapted": GOOGLE_INTERNAL,
    "Hybrid":  GOOGLE_INTERNAL,
    "Novel":   GOOGLE_INTERNAL,
}

TYPE_HATCHES = {
    "Adapted": "",    # solid
    "Hybrid":  "//",  # diagonal (bottom-left → top-right)
    "Novel":   "x",   # cross-hatch (both diagonals)
}

# Canonical (colour, hatch) → display label mapping used for auto-legend generation
STYLE_LABELS: dict[tuple[str, str], str] = {
    (SAI_ENSEMBLE,    ""):   "Google SAI Ensemble",
    (GOOGLE_INTERNAL, ""):   "Adapted",
    (GOOGLE_INTERNAL, "//"): "Hybrid",
    (GOOGLE_INTERNAL, "x"):  "Novel",
    (HUB,             ""):   "Hub-submitted models",
    (HUB,             "//"): "Hub ensemble / baseline",
    (OTHER_GREY,      ""):   "Other submitted models",
}

_rdylgn = plt.colormaps["RdYlGn"]

# Diverging colormap for relative WIS tables:
#   green (much better) → white (= baseline) → yellow → red (much worse)
REL_WIS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rel_wis",
    [(0.00, "#227c49"), (0.25, "#13cb63"), (0.50, "white"), (0.75, "#fcd059"), (1.00, "#d73027")],
)


def rel_wis_style(s: pd.Series, q_lim: float = 0.90) -> list[str]:
    """Style a relative-WIS column using a log-scale diverging colormap.

    Maps log(rel_wis) linearly onto REL_WIS_CMAP, with the midpoint (white)
    anchored at rel_wis = 1 (= log 0). Scale limits are set symmetrically at
    the q_lim quantile of |log(rel_wis)| so outliers don't wash out the range.
    """
    log_vals = np.log(pd.to_numeric(s, errors="coerce").replace(0, np.nan))
    lim = float(np.nanquantile(np.abs(log_vals), q_lim))
    if lim == 0:
        lim = 1.0

    styles = []
    for lv in log_vals:
        if pd.isna(lv):
            styles.append("")
        else:
            norm = max(0.0, min(1.0, 0.5 + lv / (2 * lim)))
            rgba = REL_WIS_CMAP(norm)
            bg = mcolors.to_hex(rgba)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            fg = "black" if lum > 0.40 else "white"
            styles.append(f"background-color: {bg}; color: {fg}")
    return styles

def build_legend_entries(
    model_colours: dict,
    model_hatches: dict | None = None,
) -> list[tuple[str, str, str]]:
    """Return deduplicated (facecolor, hatch, label) tuples for a legend.

    Iterates over unique (colour, hatch) pairs present in model_colours /
    model_hatches and looks up display labels from STYLE_LABELS.  Pairs not in
    STYLE_LABELS are silently skipped so unknown colours don't pollute the legend.
    Order follows STYLE_LABELS insertion order so the legend is consistent.
    """
    present = {
        (c, (model_hatches or {}).get(m, ""))
        for m, c in model_colours.items()
    }
    return [
        (c, h, lbl)
        for (c, h), lbl in STYLE_LABELS.items()
        if (c, h) in present
    ]


def is_hub_generated_model(model_id: str) -> bool:
    ml = str(model_id).lower()
    return (
        ("ensemble" in ml or "baseline" in ml or "flusight" in ml)
        and "google" not in ml
        and "sai" not in ml
    )


def make_model_colours(
    df: pd.DataFrame,
    hub_generated_models: set | None = None,
) -> dict:
    hub_generated_models = hub_generated_models or set()

    colours = {
        m: TYPE_COLOURS.get(t, HUB)
        for m, t in df.groupby("model_id")["model_type"].first().items()
    }

    for m in colours:
        if m in hub_generated_models or is_hub_generated_model(m):
            colours[m] = HUB

    return colours


def make_model_hatches(
    df: pd.DataFrame,
    hub_generated_models: set | None = None,
) -> dict:
    hub_generated_models = hub_generated_models or set()

    hatches = {
        m: TYPE_HATCHES.get(t, "")
        for m, t in df.groupby("model_id")["model_type"].first().items()
    }

    for m in hatches:
        if m in hub_generated_models or is_hub_generated_model(m):
            hatches[m] = HUB_HATCH

    return hatches


def model_type_style(s):
    return [f"color: {TYPE_COLOURS.get(v, 'black')}; font-weight: bold" for v in s]
