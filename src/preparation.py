import pandas as pd
from pathlib import Path

# Handling files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
in_path = PROJECT_ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
out_path = PROJECT_ROOT / "data" / "processed" / "Friday_WorkingHours-Afternoon-DDos-Clean.csv"
df = pd.read_csv(in_path)

# Clean column names by getting rid of white spaces and converting it into uppercase
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip().str.upper()

# Creates a new column which classifies "BENIGN" and "DDOS" into 0 and 1 respectively
df["Is_attack"] = (df["Label"] != "BENIGN").astype(int)

# Saves the sanitized dataframe to out_path
df.to_csv(out_path, index=False)

# Debugging purposes
# print(df["Label"].value_counts())
