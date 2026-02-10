from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


#Removes -Clean.csv from path name for the purpose of
def dataset_name_from_file(csv_path):
    name = csv_path.name

    if name.endswith("-Clean.csv"):
        name = name[: len(name) - len("-Clean.csv")]

    if name.endswith(".csv"):
        name = name[: len(name) - len(".csv")]

    return name


#Loads all processed CSVs from data / processed.
def load_processed_csvs(processed_dir):

    #Creates a dictionary to store dataframes.
    datasets = {}

    #Load and sort all CSV files in Alphabetical order.
    csv_files = sorted(processed_dir.glob("*.csv"))

    #If there are no CSV files return an error.
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No processed CSV files found in: {processed_dir}")

    #Checks if a CSV file is empty (bytes = 0)
    for csv_path in csv_files:
        if csv_path.stat().st_size == 0:
            print(f"[SKIP] empty processed file: {csv_path.name}")
            continue

        df = pd.read_csv(csv_path)

        #If file does not contain Is_attack column, skip it.
        if "Is_attack" not in df.columns:
            print(f"[SKIP] missing Is_attack (run preparation.py?): {csv_path.name}")
            continue

        #Label is required for counting different attack types (sanity check).
        if "Label" not in df.columns:
            print(f"[SKIP] missing Label column: {csv_path.name}")
            continue

        #Store csv_path as key and the CSV's dataframe as its value in a dictionary, do it for each CSV files.
        dataset_key = dataset_name_from_file(csv_path)
        datasets[dataset_key] = df

    return datasets


#Group common columns together.
def common_feature_columns(datasets):
    dataset_keys = list(datasets.keys())
    if len(dataset_keys) == 0:
        return []

    first_dataset_key = dataset_keys[0]
    first_df = datasets[first_dataset_key]

    #Use the first dataframe to create an initial columns list
    common_features = []
    for col in first_df.columns:
        if col != "Label" and col != "Is_attack":
            common_features.append(col)

    #Starting from the second dataset, if there are common columns, add it to kept_features[].
    for dataset_key in dataset_keys[1:]:
        df = datasets[dataset_key]

        kept_features = []
        for feature in common_features:
            if feature in df.columns:
                kept_features.append(feature)

        common_features = kept_features

    common_features = sorted(common_features)
    return common_features


#Purely for research sanity checks.
def dataset_summary(datasets):
    rows = []

    for dataset_key, df in datasets.items():

        #Numpy array of 0 and 1s.
        y = df["Is_attack"].to_numpy()

        attack_rows = int(np.sum(y))
        total_rows = int(len(y))
        benign_rows = total_rows - attack_rows

        if total_rows > 0:
            attack_rate = float(attack_rows) / float(total_rows)
        else:
            attack_rate = 0.0

        # Count how many different labels exist, and how many different attack types exist.
        # We treat anything that is not BENIGN as an "attack type".
        label_series = df["Label"].astype(str)
        label_series = label_series.str.strip().str.upper()
        unique_labels = sorted(label_series.unique().tolist())
        unique_attack_labels = []
        for label in unique_labels:
            if label != "BENIGN":
                unique_attack_labels.append(label)

        row = {
            "dataset": dataset_key,
            "rows": len(df),
            "Attack rate": attack_rate,
            "Attack rows": attack_rows,
            "Benign Rows": benign_rows,
            "Unique Labels": len(unique_labels),
            "Unique Attack Types": len(unique_attack_labels),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("dataset")
    summary_df = summary_df.reset_index(drop=True)
    return summary_df


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_processed_csvs(PROCESSED_DIR)
    summary = dataset_summary(datasets)
    features = common_feature_columns(datasets)

    out_csv = RESULTS_DIR / "dataset_summary.csv"
    summary.to_csv(out_csv, index=False)

    print(f"[SUCCESS] loaded datasets: {len(datasets)}")
    print(f"[SUCCESS] common feature columns across all datasets: {len(features)}")
    print(f"[SUCCESS] wrote: {out_csv}")
