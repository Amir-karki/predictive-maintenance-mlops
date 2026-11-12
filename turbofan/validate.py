from turbofan.data import COLUMNS, RAW_DIR, load_cmapss
from pathlib import Path
import json

SENSOR_COLUMNS = [c for c in COLUMNS if c.startswith("sensor_")]

def check_no_missing(df):
    missing = df.isna().sum().sum()
    if missing > 0:
        raise ValueError(f"data has {missing} missing values.")
    return True

def check_range(df, column, low, high, margin=0.1):
    pad = (high - low) * margin
    low = low - pad
    high = high + pad
    too_low = (df[column] < low).sum()
    too_high = (df[column] > high).sum()
    bad = too_low + too_high
    if bad > 0:
        raise ValueError(f"{column}: {bad} values outside {low} to {high}")
    return True

def build_profile(df, columns):
    profile = {}
    for column in columns:
        profile[column] = (float(df[column].min()), float(df[column].max()))
    return profile

def check_all_ranges(df, profile):
    problems = []
    for column, (low, high) in profile.items():
        try:
            check_range(df, column, low, high)
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError("range check failed:\n" + "\n".join(problems))
    return True

PROFILE_PATH = Path("data_profile.json")
REPORT_PATH = Path("validation_report.json")

def save_profile(profile, path=PROFILE_PATH):
    path.write_text(json.dumps(profile, indent=2))

def load_profile(path=PROFILE_PATH):
    return json.loads(path.read_text())

def main():
    train = load_cmapss(RAW_DIR / "train_FD001.txt")
    check_no_missing(train)

    if PROFILE_PATH.exists():
        check_all_ranges(train, load_profile())
        print("data matches the saved profile.")
    else:
        save_profile(build_profile(train, SENSOR_COLUMNS))
        print(f"no profile found, created {PROFILE_PATH}")

    print("data quality checks passed.")

    report = {"rows": train.shape[0], "status": "passed"}
    REPORT_PATH.write_text(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()