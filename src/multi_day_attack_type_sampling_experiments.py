import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset_feature_alignment import (
    PROCESSED_DIR,
    RESULTS_DIR,
    common_feature_columns,
    load_processed_csvs,
    reduced_feature_columns,
)
from experiment_helpers import evaluate_predictions, to_percentage
from multi_day_experiments import split_train_test_by_day

# THIS EXPERIMENT KEEPS THE SAME MULTI-DAY HELD-OUT-DAY SETUP
# BUT SAMPLES THE TRAINING DATA BY EXACT ATTACK TYPE LABEL
# SO RARE ATTACK TYPES ARE LESS LIKELY TO DISAPPEAR FROM TRAINING

MAX_BENIGN_ROWS = 50000
MAX_ATTACK_TYPE_ROWS = 20000


def sample_training_by_attack_type(training_dataframe):
    sampled_dataframes = []

    benign_rows = training_dataframe[training_dataframe["Label"] == "BENIGN"]
    if len(benign_rows) > MAX_BENIGN_ROWS:
        benign_rows = benign_rows.sample(
            n=MAX_BENIGN_ROWS,
            random_state=42,
        )
    if len(benign_rows) > 0:
        sampled_dataframes.append(benign_rows)

    label_names = sorted(training_dataframe["Label"].dropna().unique())

    for label_name in label_names:
        if label_name == "BENIGN":
            continue

        attack_type_rows = training_dataframe[training_dataframe["Label"] == label_name]
        if len(attack_type_rows) > MAX_ATTACK_TYPE_ROWS:
            attack_type_rows = attack_type_rows.sample(
                n=MAX_ATTACK_TYPE_ROWS,
                random_state=42,
            )

        if len(attack_type_rows) > 0:
            sampled_dataframes.append(attack_type_rows)

    if len(sampled_dataframes) == 0:
        return training_dataframe

    sampled_training_dataframe = pd.concat(
        sampled_dataframes,
        axis=0,
        ignore_index=True,
    )

    sampled_training_dataframe = sampled_training_dataframe.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    return sampled_training_dataframe


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_processed_csvs(PROCESSED_DIR)
    all_feature_columns = common_feature_columns(datasets)
    reduced_columns = reduced_feature_columns(all_feature_columns, datasets=datasets)

    feature_sets = [
        ("all_features", all_feature_columns),
        ("reduced_features", reduced_columns),
    ]

    dataframes_by_day = {}
    dataset_keys_by_day = {}

    for dataset_key, dataframe in datasets.items():
        dataset_name_parts = dataset_key.split("-")
        day_name = dataset_name_parts[0]

        if day_name not in dataframes_by_day:
            dataframes_by_day[day_name] = []
            dataset_keys_by_day[day_name] = []

        dataframes_by_day[day_name].append(dataframe)
        dataset_keys_by_day[day_name].append(dataset_key)

    day_names = sorted(dataframes_by_day.keys())
    results_rows = []

    for held_out_day in day_names:
        print(f"[HELD OUT DAY]: {held_out_day}")

        split_result = split_train_test_by_day(
            dataframes_by_day,
            dataset_keys_by_day,
            day_names,
            held_out_day,
        )
        if split_result is None:
            print(f"[SKIP] missing train/test data when holding out {held_out_day}.")
            continue

        (
            combined_training_dataframe,
            combined_held_out_test_dataframe,
            training_dataset_keys,
            held_out_test_dataset_keys,
        ) = split_result

        full_pooled_training_row_count = len(combined_training_dataframe)
        sampled_training_dataframe = sample_training_by_attack_type(
            combined_training_dataframe
        )

        print(
            f"[ATTACK-TYPE SAMPLING] Held-out day {held_out_day}: "
            f"reduced training rows from {full_pooled_training_row_count} "
            f"to {len(sampled_training_dataframe)}"
        )

        training_labels = sampled_training_dataframe["Is_attack"]
        unique_training_classes = training_labels.unique()
        if len(unique_training_classes) < 2:
            print(
                f"[SKIP] pooled training data for held-out day {held_out_day} only has one class."
            )
            continue

        training_days = []
        for day_name in day_names:
            if day_name != held_out_day:
                training_days.append(day_name)

        training_days_text = ", ".join(training_days)
        training_dataset_files_text = ", ".join(sorted(training_dataset_keys))
        held_out_test_dataset_files_text = ", ".join(sorted(held_out_test_dataset_keys))
        held_out_test_labels = combined_held_out_test_dataframe["Is_attack"]

        for feature_set_name, chosen_feature_columns in feature_sets:
            training_features = sampled_training_dataframe[chosen_feature_columns]
            held_out_test_features = combined_held_out_test_dataframe[chosen_feature_columns]

            print(
                f"[FEATURE SET] {feature_set_name} ({len(chosen_feature_columns)} cols) "
                f"for {held_out_day}"
            )

            logistic_regression_scaler = StandardScaler()
            logistic_regression_scaler.fit(training_features)
            training_features_scaled = logistic_regression_scaler.transform(
                training_features
            )
            held_out_test_features_scaled = logistic_regression_scaler.transform(
                held_out_test_features
            )

            logistic_regression_model = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                solver="saga",
                verbose=0,
            )
            logistic_regression_model.fit(training_features_scaled, training_labels)

            logistic_regression_predictions = logistic_regression_model.predict(
                held_out_test_features_scaled
            )
            logistic_regression_metrics = evaluate_predictions(
                held_out_test_labels,
                logistic_regression_predictions,
            )
            to_percentage(logistic_regression_metrics)

            row = {}
            row["experiment"] = "multi_day_attack_type_sampling"
            row["sampling_method"] = "attack_type_aware"
            row["feature_set"] = feature_set_name
            row["feature_count"] = len(chosen_feature_columns)
            row["model"] = "logistic_regression"
            row["held_out_day"] = held_out_day
            row["training_days"] = training_days_text
            row["training_dataset_files"] = training_dataset_files_text
            row["test_dataset_files"] = held_out_test_dataset_files_text
            row["full_train_rows"] = full_pooled_training_row_count
            row["train_rows"] = len(training_features)
            row["test_rows"] = len(held_out_test_features)
            row["benign_cap"] = MAX_BENIGN_ROWS
            row["attack_type_cap"] = MAX_ATTACK_TYPE_ROWS
            row["accuracy"] = logistic_regression_metrics["accuracy"]
            row["precision"] = logistic_regression_metrics["precision"]
            row["recall"] = logistic_regression_metrics["recall"]
            row["f1"] = logistic_regression_metrics["f1"]
            row["tn"] = logistic_regression_metrics["tn"]
            row["fp"] = logistic_regression_metrics["fp"]
            row["fn"] = logistic_regression_metrics["fn"]
            row["tp"] = logistic_regression_metrics["tp"]
            results_rows.append(row)

            random_forest_model = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            random_forest_model.fit(training_features, training_labels)

            random_forest_predictions = random_forest_model.predict(
                held_out_test_features
            )
            random_forest_metrics = evaluate_predictions(
                held_out_test_labels,
                random_forest_predictions,
            )
            to_percentage(random_forest_metrics)

            row = {}
            row["experiment"] = "multi_day_attack_type_sampling"
            row["sampling_method"] = "attack_type_aware"
            row["feature_set"] = feature_set_name
            row["feature_count"] = len(chosen_feature_columns)
            row["model"] = "random_forest"
            row["held_out_day"] = held_out_day
            row["training_days"] = training_days_text
            row["training_dataset_files"] = training_dataset_files_text
            row["test_dataset_files"] = held_out_test_dataset_files_text
            row["full_train_rows"] = full_pooled_training_row_count
            row["train_rows"] = len(training_features)
            row["test_rows"] = len(held_out_test_features)
            row["benign_cap"] = MAX_BENIGN_ROWS
            row["attack_type_cap"] = MAX_ATTACK_TYPE_ROWS
            row["accuracy"] = random_forest_metrics["accuracy"]
            row["precision"] = random_forest_metrics["precision"]
            row["recall"] = random_forest_metrics["recall"]
            row["f1"] = random_forest_metrics["f1"]
            row["tn"] = random_forest_metrics["tn"]
            row["fp"] = random_forest_metrics["fp"]
            row["fn"] = random_forest_metrics["fn"]
            row["tp"] = random_forest_metrics["tp"]
            results_rows.append(row)

    results_dataframe = pd.DataFrame(results_rows)
    output_csv_path = RESULTS_DIR / "multi_day_attack_type_sampling_metrics.csv"
    results_dataframe.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] wrote: {output_csv_path}")


if __name__ == "__main__":
    main()
