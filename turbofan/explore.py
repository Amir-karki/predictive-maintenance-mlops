"""Quick Sanity check of the FD001 training data."""

from turbofan.data import RAW_DIR, load_cmapss, add_rul

train = load_cmapss(RAW_DIR / "train_FD001.txt")
train = add_rul(train)

print("Shape:", train.shape)
print("Number of engines:", train["unit"].nunique())

life = train.groupby("unit")["cycle"].max()
print(f"Engine lifetimes: min={life.min()}, max={life.max()}, mean={life.mean():.1f}")

print("\nFirst rows of engine 1:")
print(train[train["unit"] == 1][["unit", "cycle", "sensor_2", "sensor_7", "rul"]].head())

print("\nSensors that never change (useless for prediction):")
print([c for c in train.columns if train[c].std() < 1e-4])