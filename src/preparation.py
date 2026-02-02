from pathlib import Path

import numpy as np
import pandas as pd

# An intrusion detection pipeline that detects malicious network traffic
# and evaluates how well it generalises across different days/attack types.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Find all files within data/raw that ends with .csv, and stores it in a list.
raw_files = sorted(RAW_DIR.glob("*.csv"))

# If raw_files happens to be empty, return an error.
if not raw_files:
    print(f"[error] no CSV files found in {RAW_DIR}")
else:
    for raw_path in raw_files:
        df = pd.read_csv(raw_path)

        # Clean column names and normalize labels for consistency.
        df.columns = df.columns.str.strip()
        if "Label" not in df.columns:
            print("[skip] missing Label column: {raw_path.name}")
            continue

        df["Label"] = df["Label"].astype(str).str.strip().str.upper()
        df["Is_attack"] = (df["Label"] != "BENIGN").astype(int)

        # Replace infinities and NaNs to keep models stable.
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Remove common suffixes so we're left with just the base name.
        base = (
            raw_path.name.replace(".pcap_ISCX.csv", "")
            .replace(".pcap_iscx.csv", "")
            .replace(".csv", "")
        )
        out_path = PROCESSED_DIR / "{base}-Clean.csv"
        df.to_csv(out_path, index=False)
        print(f"[ok] {raw_path.name} -> {out_path.name}")
