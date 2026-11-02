# Epidemiological Forecast Hub Performance Analysis

Code for evaluating the performance of forecasting models submitted to the U.S. CDC epidemiological forecast hubs for COVID-19, influenza (FluSight), and RSV. Forecasts are scored using the Weighted Interval Score (WIS). This repository accompanies [Martinson et al. (2026)](https://arxiv.org/abs/2605.16238).

## Installation

**Requirements:** Python ≥ 3.11. A Jupyter environment is also needed to run analysis notebooks.

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd performance-analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   .venv\Scripts\activate.ps1       # Windows (PowerShell)
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. *(Table image export only)* `dataframe_image` uses Playwright to render coloured tables as images. If you need this, install the browser backend after the step above:
   ```bash
   playwright install chromium
   ```
   This step can be skipped if you only need figures and in-notebook table output.

**Typical install time:** approximately 2–4 minutes on a standard laptop (excluding the optional Playwright browser download, which may add a further 2–3 minutes).

## Demo

The repository includes pre-processed forecast and scoring data for all hubs, so figures can be reproduced immediately without re-downloading data from the hubs.

To run the FluSight analysis:

1. Activate the virtual environment (see above) and open `flu_analysis.ipynb` in a Jupyter environment.
2. Run all cells (`Run` → `Run All Cells`, or `Shift+Enter` through each cell).
3. **Expected outputs:**
   - Figures saved to `figures/hubs/flu/`
   - Summary tables printed in-notebook and exported as `.tex` files to `tables/`
4. **Typical runtime:** approximately 2–4 minutes on a standard laptop.

The same procedure applies to `covid_analysis.ipynb` and `rsv_analysis.ipynb` for COVID-19 and RSV analyses, and to `hub_season_analysis.ipynb` and `internal_hub_analysis.ipynb` for combined cross-hub analyses.

Note: coloured tables do not render in GitHub's notebook preview but display correctly when run locally.

## Usage

### Reproducing the full analysis pipeline

To regenerate all processed data and re-run the analysis from scratch for any of the COVID (`covid`), FluSight (`flu`), or RSV (`rsv`) hubs:

1. **Download forecast data** from the hub:
   ```bash
   python -m src.load_hub --hub <hub> --incremental
   ```
   Downloads forecast submissions from the corresponding CDC GitHub repository and saves them as `data/processed/<hub>/forecasts.parquet`. Ground truth data are always reloaded. The `--incremental` flag skips files already downloaded.

2. **Score forecasts:**
   ```bash
   python -m src.scoring --hub <hub>
   ```
   Computes WIS and log WIS and saves results to `data/processed/<hub>/scores.parquet`.

3. **Run the analysis notebook:**
   Open `<hub>_analysis.ipynb` and run all cells to generate all plots and tables.

### Google Internal Hub

To run analysis for the [GoogleHub](https://github.com/google-research/google-research/tree/master/epi_forecasts) hosted on the Google Research public GitHub page (`google_covid`, `google_flu`, `google_rsv`):

1. Load the corresponding CDC hub data first (the internal hubs reuse CDC ground truth, this step can be skipped if all above listed steps have already been run):
   ```bash
   python -m src.load_hub --hub <hub> --incremental
   ```
2. Load Google internal forecasts for each pathogen:
   ```bash
   python -m src.load_hub --hub google_<hub> --incremental
   ```
3. Score each internal hub:
   ```bash
   python -m src.scoring --hub google_<hub>
   ```
4. Open `internal_hub_analysis.ipynb` and run all cells for a combined analysis.

## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{martinson2026prospectivemultipathogendiseaseforecasting,
      title={Prospective multi-pathogen disease forecasting using autonomous LLM-guided tree search},
      author={Sarah Martinson and Michael P. Brenner and Martyna Plomecka and Brian P. Williams and Nicholas G. Reich and Zahra Shamsi},
      year={2026},
      eprint={2605.16238},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.16238},
}
```
