import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


# HUB COLOURS

GREEN = "#17b248"
GOOGLE_PINK = "#e91e8c"
HUB_BLACK = "#000000"
OTHER_GREY = "#9a9a9a"


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


# INTERNAL HUB ANALYSIS
## Google internal (by type) -- hatched
## grey = CDC submitted models
## black = CDC ensemble
## pink = Google CDC submission

# Color scheme: by model type
TYPE_COLOURS = {
    "Adapted":  "#FDE8F3", #"#CBA6DE",
    "Hybrid":   "#F9BEDE", #"#B0CB92",
    "Novel":    "#F494C9", #"#78ADB8",
}

TYPE_HATCHES = {
    "Adapted": "//",
    "Hybrid":  "\\\\",
    "Novel":   "..",
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

def make_model_colours(df: pd.DataFrame) -> dict:
    return {
        m: TYPE_COLOURS[t]
        for m, t in df.groupby("model_id")["model_type"].first().items()
    }

def make_model_hatches(df: pd.DataFrame) -> dict:
    return {
        m: TYPE_HATCHES.get(t, "")
        for m, t in df.groupby("model_id")["model_type"].first().items()
    }

def model_type_style(s):
    return [f"color: {TYPE_COLOURS.get(v, 'black')}; font-weight: bold" for v in s]
