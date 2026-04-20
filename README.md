To run analysis for any of the COVID (`covid`), FluSight (`flu`) or RSV (`rsv`) hubs, from the root directory run:

1. `python -m src.load_hub --hub <hub> --incremental`
    * This downloads the forecast data from the hub, creates `data/processed` and two parquet files, one with current forecasts and one with current ground truth.
    * Use the flag `--incremental` to load only new forecast files that haven't already been pulled.
    * Note: ground truth data are always reloaded.
2. `python -m src.scoring  --hub <hub>`
    * This computes WIS and log WIS values for the chosen hub. 
3. Open `<hub>_analysis.ipynb` and run all to generate all plots and analyses. 

Note: coloured tables do not show in GitHub previews of notebooks, but should be visible if the notebooks are run locally.

## Google Internal Hub

To run analysis for the Google Research internal hub (`google_covid`, `google_flu`, `google_rsv`):

1. Make sure the corresponding CDC hub data is already loaded (the internal hubs reuse CDC ground truth data):
    * `python -m src.load_hub --hub <hub> --incremental`
2. Load Google internal forecasts for each disease:
    * `python -m src.load_hub --hub google_<hub> --incremental`
3. Score each hub (uses the corresponding CDC hub's truth data):
    * `python -m src.scoring --hub google_<hub>`
4. Open `internal_hub_analysis.ipynb` and run all cells for a combined three-disease analysis.