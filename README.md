## Data findings (FD001)

- 6 sensors are constant across all engines and carry no information: <your list>
- Strongest wear signals (correlation with RUL): <your top 4–5 sensors>
- Sensors stay flat early in life; degradation appears in roughly the last 100–150 cycles,
  so RUL will be capped (~125) during feature engineering.
- Validation must split by engine. Random-row split RMSE: <yours>, engine split RMSE: <yours>.

## Features (FD001)

- RUL capped at 125 (sensors show no wear signal above this).
- For each of 15 sensors: raw value, 10-cycle rolling mean, 10-cycle rolling std (per engine,
  past-only) → 45 features.
- Random forest, engine split, capped RUL: raw sensors RMSE <yours>, engineered RMSE <yours>.