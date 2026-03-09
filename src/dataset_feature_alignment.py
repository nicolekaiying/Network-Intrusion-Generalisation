from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_processed_csvs(csv):
#Loads all processed CSVs from "data" / "processed"
    datasets = {}
    #Creates a dataset dictionary.
    csv_files = sorted(csv.glob("*.csv"))
    #Load and sort all files ending with .csv in alphabetical order.
    if len(csv_files) == 0:
    #If no files, raise an error.
        raise FileNotFoundError(f"[ERROR] No processed CSV files found in: {csv}")

    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        if "Is_attack" not in dataframe.columns:
        #IF column "Is_attack" does not exist within df then skip to next file.
            continue

        dataset_key = csv_path.stem
        datasets[dataset_key] = dataframe
        #Creates a key-value pair with the key being the file name & the value being the dataframe.

    return datasets


def common_feature_columns(datasets):
#Group common features together.
    dataset_keys = list(datasets.keys())
    if len(dataset_keys) == 0:
        return []

    first_dataset_key = dataset_keys[0]
    first_df = datasets[first_dataset_key]

    common_features = []
    for col in first_df.columns:
        if col != "Label" and col != "Is_attack":
            common_features.append(col)
            #Add all columns of first dataframe to common_features except for "Label" and "Is_Attack" columns.

    for dataset_key in dataset_keys[1:]:
    #Starting from the second dataset key.
        dataframe = datasets[dataset_key]

        kept_features = []
        for feature in common_features:
            if feature in dataframe.columns:
                kept_features.append(feature)
                #If the same column already exists in the next dataframe, then add that to kept_features.

        common_features = kept_features

    common_features = sorted(common_features)
    return common_features


def reduced_feature_columns(all_feature_columns):
#Function to remove all constant feature columns.
    columns_to_remove = [
        "Fwd Header Length.1",
        "Bwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk",
        "Bwd Avg Packets/Bulk",
        "Bwd PSH Flags",
        "Bwd URG Flags",
        "Fwd Avg Bulk Rate",
        "Fwd Avg Bytes/Bulk",
        "Fwd Avg Packets/Bulk",
    ]

    reduced_columns = []

    for column_name in all_feature_columns:
        if column_name not in columns_to_remove:
            reduced_columns.append(column_name)

    return reduced_columns


def dataset_summary(datasets):
    rows = []

    for dataset_key, df in datasets.items():
        y = df["Is_attack"].to_numpy()

        attack_rows = int(np.sum(y))
        total_rows = int(len(y))
        benign_rows = total_rows - attack_rows

        if total_rows > 0:
            attack_rate = float(attack_rows) / float(total_rows)
        else:
            attack_rate = 0.0

        row = {
            "dataset": dataset_key,
            "rows": len(df),
            "Attack rate": attack_rate,
            "Attack rows": attack_rows,
            "Benign Rows": benign_rows,
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("dataset")
    summary_df = summary_df.reset_index(drop=True)

    print("[CREATED]: 'dataset_summary.csv'")

    return summary_df


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_datasets = load_processed_csvs(PROCESSED_DIR)
    summary = dataset_summary(all_datasets)
    features = common_feature_columns(all_datasets)

    out_csv = RESULTS_DIR / "dataset_summary.csv"
    summary.to_csv(out_csv, index=False)
