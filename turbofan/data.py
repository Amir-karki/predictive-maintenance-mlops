"""Loading and labeling the NASA C-MAPSS turbofan dataset."""

from pathlib import Path
import pandas as pd

# The raw files have no headers, so we need to spedify the column names.

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS

RAW_DIR = Path("data/raw")

def load_cmapss(path: Path) -> pd.DataFrame:
    """Load the NASA C-MAPSS turbofan dataset from a text file.
    Args:
        path (Path): Path to the text file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add the remaining useful life (RUL) column to the dataset.
    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The dataset with the RUL column added.
    """
    # Calculate the maximum cycle for each unit
    df = df.copy() # never modify the caller's data in place
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last_cycle - df["cycle"]
    return df