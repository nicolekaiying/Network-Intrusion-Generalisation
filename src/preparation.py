from pathlib import Path

import numpy as np
import pandas as pd

# An intrusion detection pipeline that detects malicious network traffic
# and evaluates how well it generalises across different days/attack types.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMN = "Label"
TARGET_COLUMN = "Is_attack"


def processed_name_from_raw(raw_path):
    # Remove common suffixes so we're left with just the dataset base name.
    base_name = (
        raw_path.name.replace(".pcap_ISCX.csv", "")
        .replace(".pcap_iscx.csv", "")
        .replace(".csv", "")
    )
    return f"{base_name}-Clean.csv"


def clean_raw_dataframe(dataframe):
    dataframe = dataframe.copy()
    dataframe.columns = dataframe.columns.str.strip()
    # Clean column names and normalize labels for consistency.
    dataframe[LABEL_COLUMN] = dataframe[LABEL_COLUMN].astype(str).str.strip().str.upper()
    dataframe[TARGET_COLUMN] = (dataframe[LABEL_COLUMN] != "BENIGN").astype(int)

    # Replace infinities and NaNs to keep models stable.
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).fillna(0)
    return dataframe


def main():
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    # Find all files within data/raw that end with .csv and store them in a list.

    if not raw_files:
        # If no raw CSVs are found, stop early with a clear message.
        print(f"[ERROR] no CSV files found in {RAW_DIR}")
        return

    for raw_path in raw_files:
        dataframe = pd.read_csv(raw_path)

        if LABEL_COLUMN not in dataframe.columns:
            print(f"[SKIPPED] missing Label column: {raw_path.name}")
            continue

        cleaned_dataframe = clean_raw_dataframe(dataframe)
        output_path = PROCESSED_DIR / processed_name_from_raw(raw_path)
        cleaned_dataframe.to_csv(output_path, index=False)
        print(f"[SUCCESS] {raw_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
