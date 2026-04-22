import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


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
