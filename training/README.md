# Training notebooks

These 15 notebooks fit and saved the 15 models the app uses, one for each category and sector. They are my 2025
notebooks, cleaned up only so they run from this folder (the changes are listed at the end).

| Category | Rural | Urban | Combined |
|---|---|---|---|
| Fruits | [fruits_rural](fruits_rural.ipynb) | [fruits_urban](fruits_urban.ipynb) | [fruits_combined](fruits_combined.ipynb) |
| Vegetables | [vegetables_rural](vegetables_rural.ipynb) | [vegetables_urban](vegetables_urban.ipynb) | [vegetables_combined](vegetables_combined.ipynb) |
| Spices | [spices_rural](spices_rural.ipynb) | [spices_urban](spices_urban.ipynb) | [spices_combined](spices_combined.ipynb) |
| Fuel and light | [fuel_and_light_rural](fuel_and_light_rural.ipynb) | [fuel_and_light_urban](fuel_and_light_urban.ipynb) | [fuel_and_light_combined](fuel_and_light_combined.ipynb) |
| Health | [health_rural](health_rural.ipynb) | [health_urban](health_urban.ipynb) | [health_combined](health_combined.ipynb) |

## What each notebook does

1. Loads the CPI table (`../Resources/Revised Final Dataset.csv`) and keeps one sector.
2. Fills the few missing months by linear interpolation.
3. Looks at the series: a plot, an STL decomposition, an ADF test and the ACF and PACF plots.
4. Holds out the last 12 months (August 2024 to July 2025) as the test set.
5. Fits a first seasonal ARIMA. Then it searches between 72 and 324 settings (81 in most notebooks) with time series
   cross validation over 3 folds of 12 months and ranks them by RMSE.
6. Fits the chosen setting on the training months, reports its 12 month test RMSE and saves the model with joblib.

## Run them

```bash
pip install -r ../requirements.txt scikit-learn tqdm jupyter
jupyter notebook
```

Most notebooks run in a few minutes. Because of their bigger searches, fruits_urban, vegetables_combined and
fruits_rural take longer (up to about 40 minutes). The saved `.pkl` file lands in this folder (git ignores it). The app loads compressed copies of the same models from `../Resources/Time Series Models/`.

## Checked in November 2026

- Refitting each notebook's chosen setting on the training months gives exactly the parameters of the saved model in
  `../Resources/Time Series Models/` for all 15.
- The 12 month test RMSE matches the white paper's table ([section 3.2](../docs/whitepaper.md#32-reported-model-results))
  to two decimals for 14 of the 15 models. Health Rural comes out at 0.43 against the reported 0.40.

## Worth knowing

- Some chosen settings were picked by hand after the search. A few sit outside the searched range, for example the
  seasonal differencing of 4 in Fruits Rural.
- The models are fit on the training months only and end in July 2024. Before each forecast the app adds the newer
  months without changing the fitted settings.
- There is no simple seasonal baseline yet.

## Changes from the 2025 originals

- The data path points at `../Resources/` instead of my own folder.
- The `pip install` lines are gone. The packages are listed above.
- Chart titles: "YoY" is gone because the series are index levels, not year on year changes. One title typo is fixed
  and the ACF and PACF titles use a colon.
- Fruits Rural: the final fit uses the training months, as it did when the saved model was made. An unfinished line
  in its first model cell is completed.
- The two Health notebooks that had no save line (health_rural and health_combined) end with one now.
- fruits_combined: an early copy of the error cell that ran before its forecast existed is gone. So are five cells
  that only printed or inspected dates and were not part of training.
