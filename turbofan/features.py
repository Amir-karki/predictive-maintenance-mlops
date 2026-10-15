"""Feature engineering for turbofan Remaining Useful Life (RUL) prediction.
The same functions are used for training, validation, testing and live API, so features are always computed automatically.
"""
import pandas as pd

FEATURE_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
    "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",]
 
WINDOW= 10 # # of past cycles each rolling features look at
RUL_CAP = 125 # sensors show no wear above this, so RUL is capped here

def cap_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:

    """Cap the RUL target: any RUL above 'cap' becomes `cap`"""

    df = df.copy()
    df["rul"] = df["rul"].clip(upper=cap)
    return df

def add_rolling_features(df: pd.DataFrame, sensors: list[str] = FEATURE_SENSORS, window: int = WINDOW) -> pd.DataFrame:

    """Add a rolling mean and rolling std for each sensor, per engine.
    Windows only look at the current and previous cycles (never the future), and restart for every engine so engines never mix.
    """

    df = df.copy()
    df = df.sort_values(["unit", "cycle"]) # our training data is already sorted, but data arriving in real production mightn't be. Without sorting, we would silently average the wrong rows. Defensive programming basically

    for sensor in sensors:
        by_engine = df.groupby("unit")[sensor]

        df[f"{sensor}_mean"] = by_engine.transform(
            lambda s: s.rolling(window=WINDOW, min_periods=1).mean()
        )

        df[f"{sensor}_std"] = by_engine.transform(
            lambda s: s.rolling(window=WINDOW, min_periods=1).std().fillna(0)
        )
    return df

def get_feature_columns(sensors: list[str] = FEATURE_SENSORS) -> list[str]:
    """Return the model's input column names, in a fixed order."""
    means = [f"{s}_mean" for s in sensors]
    stds = [f"{s}_std" for s in sensors]
    return sensors + means + stds