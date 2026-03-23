from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

NON_FEATURE_COLUMNS = {"Label", "Is_attack"}


def load_processed_csvs(processed_dir):
    # Load all processed CSVs from data/processed.
    datasets = {}
    csv_files = sorted(processed_dir.glob("*.csv"))
    # Load and sort all files ending with .csv in alphabetical order.

    if len(csv_files) == 0:
        # If no files are found, raise an error.
        raise FileNotFoundError(f"[ERROR] No processed CSV files found in: {processed_dir}")

    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        if "Is_attack" not in dataframe.columns:
            # Skip files that are not part of the processed experiment pipeline.
            continue

        dataset_key = csv_path.stem
        datasets[dataset_key] = dataframe
        # Create a key-value pair with the filename stem as the dataset key.

    return datasets


def common_feature_columns(datasets):
    # Find the feature columns shared across every processed dataset.
    dataset_keys = list(datasets.keys())
    if len(dataset_keys) == 0:
        return []

    first_dataset_key = dataset_keys[0]
    first_df = datasets[first_dataset_key]

    common_features = []
    for column_name in first_df.columns:
        if column_name not in NON_FEATURE_COLUMNS:
            common_features.append(column_name)
            # Start with all candidate feature columns from the first dataset.

    for dataset_key in dataset_keys[1:]:
        # Starting from the second dataset, keep only columns seen everywhere.
        dataframe = datasets[dataset_key]

        kept_features = []
        for feature in common_features:
            if feature in dataframe.columns:
                kept_features.append(feature)
                # If the feature also exists in this dataset, keep it.

        common_features = kept_features

    return sorted(common_features)


def constant_feature_columns(datasets, all_feature_columns):
    # Find shared features that never change value within any processed dataset.
    if len(datasets) == 0 or len(all_feature_columns) == 0:
        return []

    constant_columns = None

    for dataframe in datasets.values():
        feature_frame = dataframe[all_feature_columns]
        unique_counts = feature_frame.nunique(dropna=False)
        dataset_constant_columns = set(unique_counts[unique_counts <= 1].index)

        if constant_columns is None:
            constant_columns = dataset_constant_columns
        else:
            constant_columns = constant_columns.intersection(dataset_constant_columns)

        if len(constant_columns) == 0:
            return []

    return sorted(constant_columns)


def reduced_feature_columns(all_feature_columns, datasets=None):
    # Remove shared feature columns that are constant across every dataset.
    if datasets is None:
        raise ValueError(
            "datasets is required so constant feature columns can be detected "
            "from the processed data."
        )

    columns_to_remove = set(constant_feature_columns(datasets, all_feature_columns))

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
