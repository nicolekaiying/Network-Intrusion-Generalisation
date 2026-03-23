import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_feature_alignment import (
    PROCESSED_DIR,
    RESULTS_DIR,
    common_feature_columns,
    load_processed_csvs,
    reduced_feature_columns,
)
from experiment_helpers import (
    choose_best_threshold,
    evaluate_predictions,
    predictions_from_threshold,
    to_percentage,
)
from multi_day_experiments import (
    MAX_POOLED_TRAINING_ROWS,
    split_train_test_by_day,
    undersample_majority_class,
)

# THIS EXPERIMENT USES THE SAME MULTI-DAY HELD-OUT-DAY SETUP
# BUT TUNES THE LOGISTIC REGRESSION THRESHOLD ON A VALIDATION SPLIT
# INSTEAD OF ALWAYS USING THE DEFAULT 0.5 CUTOFF

VALIDATION_SIZE = 0.2
THRESHOLD_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
MINIMUM_PRECISION = 0.70


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

        if full_pooled_training_row_count > MAX_POOLED_TRAINING_ROWS:
            print(
                f"[SAMPLING] Reducing pooled training rows for held-out day {held_out_day} "
                f"from {full_pooled_training_row_count} to {MAX_POOLED_TRAINING_ROWS}"
            )

            sampled_split_result = train_test_split(
                combined_training_dataframe,
                train_size=MAX_POOLED_TRAINING_ROWS,
                random_state=42,
                stratify=combined_training_dataframe["Is_attack"],
            )

            sampled_training_dataframe = sampled_split_result[0]
        else:
            sampled_training_dataframe = combined_training_dataframe

        training_dataframe = undersample_majority_class(sampled_training_dataframe)
        if len(training_dataframe) != len(sampled_training_dataframe):
            print(
                f"[UNDERSAMPLING] Held-out day {held_out_day}: "
                f"reduced training rows from {len(sampled_training_dataframe)} "
                f"to {len(training_dataframe)}"
            )
        else:
            print(
                f"[UNDERSAMPLING] Held-out day {held_out_day}: "
                f"training rows stayed at {len(training_dataframe)}"
            )

        training_labels = training_dataframe["Is_attack"]
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
            training_features = training_dataframe[chosen_feature_columns]
            held_out_test_features = combined_held_out_test_dataframe[
                chosen_feature_columns
            ]

            print(
                f"[FEATURE SET] {feature_set_name} ({len(chosen_feature_columns)} cols) "
                f"for {held_out_day}"
            )

            try:
                validation_split_result = train_test_split(
                    training_features,
                    training_labels,
                    test_size=VALIDATION_SIZE,
                    random_state=42,
                    stratify=training_labels,
                )
            except ValueError as exc:
                print(
                    f"[SKIP] could not create validation split for {held_out_day}, "
                    f"{feature_set_name}: {exc}"
                )
                continue

            threshold_training_features = validation_split_result[0]
            validation_features = validation_split_result[1]
            threshold_training_labels = validation_split_result[2]
            validation_labels = validation_split_result[3]

            threshold_scaler = StandardScaler()
            threshold_scaler.fit(threshold_training_features)

            threshold_training_features_scaled = threshold_scaler.transform(
                threshold_training_features
            )
            validation_features_scaled = threshold_scaler.transform(
                validation_features
            )

            threshold_model = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                solver="saga",
                verbose=0,
            )
            threshold_model.fit(
                threshold_training_features_scaled,
                threshold_training_labels,
            )

            validation_probabilities = threshold_model.predict_proba(
                validation_features_scaled
            )[:, 1]

            chosen_threshold, validation_metrics, threshold_selection_mode = (
                choose_best_threshold(
                    validation_labels,
                    validation_probabilities,
                    THRESHOLD_VALUES,
                    MINIMUM_PRECISION,
                )
            )

            full_training_scaler = StandardScaler()
            full_training_scaler.fit(training_features)

            full_training_features_scaled = full_training_scaler.transform(
                training_features
            )
            held_out_test_features_scaled = full_training_scaler.transform(
                held_out_test_features
            )

            final_model = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                solver="saga",
                verbose=0,
            )
            final_model.fit(full_training_features_scaled, training_labels)

            held_out_attack_probabilities = final_model.predict_proba(
                held_out_test_features_scaled
            )[:, 1]
            held_out_predictions = predictions_from_threshold(
                held_out_attack_probabilities,
                chosen_threshold,
            )
            held_out_metrics = evaluate_predictions(
                held_out_test_labels,
                held_out_predictions,
            )

            validation_metrics_percent = dict(validation_metrics)
            held_out_metrics_percent = dict(held_out_metrics)
            to_percentage(validation_metrics_percent)
            to_percentage(held_out_metrics_percent)

            row = {}
            row["experiment"] = "multi_day_threshold_tuning"
            row["model"] = "logistic_regression"
            row["feature_set"] = feature_set_name
            row["feature_count"] = len(chosen_feature_columns)
            row["held_out_day"] = held_out_day
            row["training_days"] = training_days_text
            row["training_dataset_files"] = training_dataset_files_text
            row["test_dataset_files"] = held_out_test_dataset_files_text
            row["full_train_rows"] = full_pooled_training_row_count
            row["sampled_train_rows"] = len(sampled_training_dataframe)
            row["train_rows"] = len(training_features)
            row["validation_rows"] = len(validation_features)
            row["test_rows"] = len(held_out_test_features)
            row["default_threshold"] = 0.5
            row["chosen_threshold"] = chosen_threshold
            row["minimum_precision_target"] = MINIMUM_PRECISION * 100
            row["threshold_values"] = ", ".join(str(value) for value in THRESHOLD_VALUES)
            row["threshold_selection_mode"] = threshold_selection_mode
            row["validation_accuracy"] = validation_metrics_percent["accuracy"]
            row["validation_precision"] = validation_metrics_percent["precision"]
            row["validation_recall"] = validation_metrics_percent["recall"]
            row["validation_f1"] = validation_metrics_percent["f1"]
            row["accuracy"] = held_out_metrics_percent["accuracy"]
            row["precision"] = held_out_metrics_percent["precision"]
            row["recall"] = held_out_metrics_percent["recall"]
            row["f1"] = held_out_metrics_percent["f1"]
            row["tn"] = held_out_metrics_percent["tn"]
            row["fp"] = held_out_metrics_percent["fp"]
            row["fn"] = held_out_metrics_percent["fn"]
            row["tp"] = held_out_metrics_percent["tp"]
            results_rows.append(row)

    results_dataframe = pd.DataFrame(results_rows)
    output_csv_path = RESULTS_DIR / "multi_day_threshold_tuning_metrics.csv"
    results_dataframe.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] wrote: {output_csv_path}")


if __name__ == "__main__":
    main()
