* python load_covid_hub.py 
    * this creates data/processed and two parquet files, one with current forecasts and one with current ground truth

```
# from root directory, run:
python -m src.load_hub --hub rsv      # download + save parquets
python -m src.scoring  --hub rsv      # compute WIS scores
# open rsv_analysis.ipynb and run
```