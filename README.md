To run analysis for any of the COVID (`covid`), FluSight (`flu`) or RSV (`rsv`) hubs, from the root directory run:

1. `python -m src.load_hub --hub <hub> --incremental`
    * This downloads the forecast data from the hub, creates `data/processed` and two parquet files, one with current forecasts and one with current ground truth.
    * Use the flag `--incremental` to load only new forecast files that haven't already been pulled.
    * Note: ground truth data are always reloaded.
2. `python -m src.scoring  --hub <hub>`
    * This computes WIS and log WIS values for the chosen hub. 
3. Open `<hub>_analysis.ipynb` and run all to generate all plots and analyses. 