import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    MAX_POOLED_TRAINING_ROWS,
    RANDOM_STATE,
    choose_best_threshold,
    evaluate_predictions,
    group_datasets_by_day,
    predictions_from_threshold,
    sample_training_dataframe,
    split_train_test_by_day,
    to_percentage,
    undersample_majority_class,
)

# TRAINS ON THE POOLED NON-TEST DAYS, THEN USES A VALIDATION SPLIT TO CHOOSE A BETTER DECISION THRESHOLD.
# REPORTS VALIDATION AND TEST RESULTS TO SHOW WHETHER THRESHOLD TUNING HELPS ON THE HELD-OUT DAY.

VALIDATION_SIZE = 0.2
THRESHOLD_VALUES = [round(value / 100, 2) for value in range(1, 100)]
MINIMUM_PRECISION = 0.70
RANDOM_FOREST_MAX_FEATURES_OPTIONS = ["sqrt", "log2", 0.2]
RANDOM_FOREST_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 12
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 6
RANDOM_FOREST_MIN_SAMPLES_LEAF = 4


def build_threshold_result_row(
    model_name,
    feature_set_name,
    held_out_day,
    train_rows,
    validation_rows,
    test_rows,
    chosen_threshold,
    validation_metrics,
    held_out_metrics,
    tuned_max_features=None,
):
    row = {}
    row["Experiment"] = "Multi-Day-Threshold-Tuning"
    row["Model"] = model_name
    row["Feature-Set"] = feature_set_name
    row["Test-Day"] = held_out_day
    row["Train-Rows"] = train_rows
    row["Validation-Rows"] = validation_rows
    row["Test-Rows"] = test_rows
    row["Chosen-Threshold"] = chosen_threshold
    row["Tuned-Max-Features"] = tuned_max_features
    row["Validation-Precision"] = validation_metrics["precision"]
    row["Validation-Recall"] = validation_metrics["recall"]
    row["Validation-F1"] = validation_metrics["f1"]
    row["Accuracy"] = held_out_metrics["accuracy"]
    row["Precision"] = held_out_metrics["precision"]
    row["Recall"] = held_out_metrics["recall"]
    row["F1"] = held_out_metrics["f1"]
    return row


def choose_best_random_forest_setup(
    training_features,
    training_labels,
    validation_features,
    validation_labels,
):
    best_result = None

    for max_features in RANDOM_FOREST_MAX_FEATURES_OPTIONS:
        candidate_model = RandomForestClassifier(
            n_estimators=RANDOM_FOREST_ESTIMATORS,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
            min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            max_features=max_features,
        )
        candidate_model.fit(training_features, training_labels)

        validation_probabilities = candidate_model.predict_proba(validation_features)[:, 1]
        chosen_threshold, validation_metrics, _ = choose_best_threshold(
            validation_labels,
            validation_probabilities,
            THRESHOLD_VALUES,
            MINIMUM_PRECISION,
        )

        candidate_score = (
            validation_metrics["precision"] >= MINIMUM_PRECISION,
            validation_metrics["recall"],
            validation_metrics["f1"],
        )

        if best_result is None or candidate_score > best_result["score"]:
            best_result = {
                "max_features": max_features,
                "chosen_threshold": chosen_threshold,
                "validation_metrics": validation_metrics,
                "score": candidate_score,
            }

    return best_result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_processed_csvs(PROCESSED_DIR)
    all_feature_columns = common_feature_columns(datasets)
    reduced_columns = reduced_feature_columns(all_feature_columns, datasets=datasets)

    feature_sets = [
        ("All-Features", all_feature_columns),
        ("Reduced-Features", reduced_columns),
    ]

    dataframes_by_day, dataset_keys_by_day = group_datasets_by_day(datasets)
    day_names = sorted(dataframes_by_day.keys())
    results_rows = []

    for held_out_day in day_names:
        print(f"[HELD OUT DAY]: {held_out_day}")

        training_days = [day_name for day_name in day_names if day_name != held_out_day]
        split_result = split_train_test_by_day(
            dataframes_by_day,
            dataset_keys_by_day,
            train_days=training_days,
            test_days=[held_out_day],
        )
        if split_result is None:
            print(f"[SKIP] missing train/test data when holding out {held_out_day}.")
            continue

        (
            combined_training_dataframe,
            combined_held_out_test_dataframe,
            _training_dataset_keys,
            _held_out_test_dataset_keys,
        ) = split_result

        sampled_training_dataframe, _ = sample_training_dataframe(
            combined_training_dataframe,
            max_rows=MAX_POOLED_TRAINING_ROWS,
            sampling_name=f"pooled training rows for held-out day {held_out_day}",
        )

        try:
            split_training_dataframe, validation_dataframe = train_test_split(
                sampled_training_dataframe,
                test_size=VALIDATION_SIZE,
                random_state=RANDOM_STATE,
                stratify=sampled_training_dataframe["Is_attack"],
            )
        except ValueError as exc:
            print(
                f"[SKIP] could not create validation split for {held_out_day}: {exc}"
            )
            continue

        training_dataframe = undersample_majority_class(split_training_dataframe)
        if len(training_dataframe) != len(split_training_dataframe):
            print(
                f"[UNDERSAMPLING] Held-out day {held_out_day}: "
                f"reduced threshold-training rows from {len(split_training_dataframe)} "
                f"to {len(training_dataframe)}"
            )

        training_labels = training_dataframe["Is_attack"]
        unique_training_classes = training_labels.unique()
        if len(unique_training_classes) < 2:
            print(
                f"[SKIP] pooled training data for held-out day {held_out_day} only has one class."
            )
            continue

        validation_labels = validation_dataframe["Is_attack"]
        held_out_test_labels = combined_held_out_test_dataframe["Is_attack"]

        for feature_set_name, chosen_feature_columns in feature_sets:
            training_features = training_dataframe[chosen_feature_columns]
            validation_features = validation_dataframe[chosen_feature_columns]
            held_out_test_features = combined_held_out_test_dataframe[
                chosen_feature_columns
            ]

            threshold_scaler = StandardScaler()
            threshold_scaler.fit(training_features)

            training_features_scaled = threshold_scaler.transform(training_features)
            validation_features_scaled = threshold_scaler.transform(validation_features)
            held_out_test_features_scaled = threshold_scaler.transform(
                held_out_test_features
            )

            threshold_model = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                solver="lbfgs",
            )
            threshold_model.fit(training_features_scaled, training_labels)

            validation_probabilities = threshold_model.predict_proba(
                validation_features_scaled
            )[:, 1]

            chosen_threshold, validation_metrics, _ = choose_best_threshold(
                validation_labels,
                validation_probabilities,
                THRESHOLD_VALUES,
                MINIMUM_PRECISION,
            )

            held_out_attack_probabilities = threshold_model.predict_proba(
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

            results_rows.append(
                build_threshold_result_row(
                    model_name="Logistic-Regression",
                    feature_set_name=feature_set_name,
                    held_out_day=held_out_day,
                    train_rows=len(training_features),
                    validation_rows=len(validation_features),
                    test_rows=len(held_out_test_features),
                    chosen_threshold=chosen_threshold,
                    validation_metrics=validation_metrics_percent,
                    held_out_metrics=held_out_metrics_percent,
                )
            )

            random_forest_selection = choose_best_random_forest_setup(
                training_features,
                training_labels,
                validation_features,
                validation_labels,
            )

            random_forest_model = RandomForestClassifier(
                n_estimators=RANDOM_FOREST_ESTIMATORS,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=RANDOM_FOREST_MAX_DEPTH,
                min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT,
                min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
                max_features=random_forest_selection["max_features"],
            )
            random_forest_model.fit(training_features, training_labels)

            held_out_random_forest_probabilities = random_forest_model.predict_proba(
                held_out_test_features
            )[:, 1]
            held_out_random_forest_predictions = predictions_from_threshold(
                held_out_random_forest_probabilities,
                random_forest_selection["chosen_threshold"],
            )
            held_out_random_forest_metrics = evaluate_predictions(
                held_out_test_labels,
                held_out_random_forest_predictions,
            )

            random_forest_validation_metrics_percent = dict(
                random_forest_selection["validation_metrics"]
            )
            held_out_random_forest_metrics_percent = dict(
                held_out_random_forest_metrics
            )
            to_percentage(random_forest_validation_metrics_percent)
            to_percentage(held_out_random_forest_metrics_percent)

            results_rows.append(
                build_threshold_result_row(
                    model_name="Random-Forest",
                    feature_set_name=feature_set_name,
                    held_out_day=held_out_day,
                    train_rows=len(training_features),
                    validation_rows=len(validation_features),
                    test_rows=len(held_out_test_features),
                    chosen_threshold=random_forest_selection["chosen_threshold"],
                    validation_metrics=random_forest_validation_metrics_percent,
                    held_out_metrics=held_out_random_forest_metrics_percent,
                    tuned_max_features=random_forest_selection["max_features"],
                )
            )

    results_dataframe = pd.DataFrame(results_rows)
    output_csv_path = RESULTS_DIR / "multi-day-threshold-tuning_metrics.csv"
    results_dataframe.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] wrote: {output_csv_path}")


if __name__ == "__main__":
    main()
