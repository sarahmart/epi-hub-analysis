"""Utilities for exporting styled DataFrames to LaTeX.

Usage in each notebook:

    from src.export import show_table
    EXPORT_TABLES = False          # set True to write .tex files
    TABLE_PREFIX  = "covid"        # short prefix to namespace output filenames

    # replace display(styled) with:
    show_table(styled, "wis_by_horizon", prefix=TABLE_PREFIX, export=EXPORT_TABLES,
               caption="Mean log WIS by model and horizon.", label="tab:covid_wis_horizon")

LaTeX preamble requirements (only needed when exporting):
    \\usepackage[table]{xcolor}
    \\usepackage{colortbl}
    \\usepackage{booktabs}
"""

from __future__ import annotations

import re
import pathlib
from IPython.display import display

TABLES_DIR = pathlib.Path(__file__).parent.parent / "tables"


def show_table(
    styler,
    name: str,
    *,
    prefix: str = "",
    export: bool = False,
    caption: str = "",
    label: str = "",
) -> None:
    """Display a pandas Styler in the notebook and optionally export to LaTeX.

    Parameters
    ----------
    styler:
        A pandas ``Styler`` object (the result of ``df.style...``).
    name:
        Short identifier for this table (e.g. ``"wis_by_horizon"``).
        Combined with *prefix* to form the output filename.
    prefix:
        Notebook-level prefix (e.g. ``"covid"``, ``"flu"``).  Set once per
        notebook via a top-level ``TABLE_PREFIX`` variable.
    export:
        When ``True``, write a ``.tex`` file to ``tables/``.  Controlled by
        a top-level ``EXPORT_TABLES`` variable in each notebook.
    caption:
        LaTeX table caption (optional).
    label:
        LaTeX ``\\label{}`` key (optional).  Defaults to
        ``tab:{prefix}_{name}`` when omitted.
    """
    display(styler)

    if not export:
        return

    TABLES_DIR.mkdir(exist_ok=True)

    stem = f"{prefix}_{name}" if prefix else name

    latex = styler.to_latex(
        convert_css=True,
        hrules=True,
    )

    latex = re.sub(r'(?<!\\)_', r'\\_', latex)
    latex = re.sub(r'(?<!\\)%', r'\\%', latex)

    out_path = TABLES_DIR / f"{stem}.tex"
    out_path.write_text(latex, encoding="utf-8")
    print(f"Exported → {out_path.relative_to(pathlib.Path(__file__).parent.parent)}")
