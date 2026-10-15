"""Loading and labeling the NASA C-MAPSS turbofan dataset."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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

def split_by_engine(
    df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation sets by engine, not by row.

    All cycles of an engine go to the same side. Splitting by row would
    leak information: the model would see neighbouring cycles of the
    same engine in training, making validation scores too optimistic.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["unit"]))
    return df.iloc[train_idx], df.iloc[val_idx]