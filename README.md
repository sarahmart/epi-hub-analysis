* python load_covid_hub.py 
    * this creates data/processed and two parquet files, one with current forecasts and one with current ground truth

```
python load_hub.py --hub rsv      # download + save parquets
python scoring.py  --hub rsv      # compute WIS scores
# open rsv_analysis.ipynb and run
```