from evidently import Report
import pandas as pd
from evidently.presets import DataDriftPreset

from turbofan.features import FEATURE_SENSORS
from turbofan.data import RAW_DIR, load_cmapss

import json
from pathlib import Path

REFERENCE_PATH = RAW_DIR / "train_FD001.txt"
CURRENT_PATH = RAW_DIR.parent / "drift" / "train_FD002.txt"
REPORT_PATH = Path("reports") / "drift_report.html"
SUMMARY_PATH = Path("drift_summary.json")

def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns=FEATURE_SENSORS):
    """compare current data against references data, column by column."""
    report = Report([DataDriftPreset()])
    return report.run(current_data=current[columns], reference_data=reference[columns])

def summarize(result):
    """Reduce a drift result to a small dictionary of plain numbers."""
    metrics = result.dict()["metrics"]
    overall = metrics[0]["value"]

    scores = {}
    for metric in metrics[1:]:
        column = metric["config"]["column"]
        scores[column] = round(float(metric["value"]), 3)

    return {
        "drifted_columns": int(overall["count"]),
        "share": overall["share"],
        "scores": scores,
    }

def main():
    referece = load_cmapss(REFERENCE_PATH)
    current = load_cmapss(CURRENT_PATH)

    result = run_drift_report(referece, current)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    result.save_html(str(REPORT_PATH))

    summary = summarize(result)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(f"{summary['drifted_columns']} of {len(summary['scores'])} columns drifted")


if __name__ == "__main__":
    main()