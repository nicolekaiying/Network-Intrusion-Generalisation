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
from experiment_helpers import (
    MAX_POOLED_TRAINING_ROWS,
    RANDOM_STATE,
    append_held_out_day_per_attack_type_rows,
    build_held_out_day_result_row,
    evaluate_predictions,
    group_datasets_by_day,
    sample_training_dataframe,
    split_train_test_by_day,
    to_percentage,
    undersample_majority_class,
)

# TRAINS EACH MODEL ON THE POOLED NON-TEST DAYS AND TESTS IT ON ONE HELD-OUT DAY.
# REPORTS OVERALL RESULTS AND PER-ATTACK-TYPE RESULTS TO SHOW HOW WELL PERFORMANCE GENERALISES ACROSS DAYS.


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_processed_csvs(PROCESSED_DIR)
    all_feature_columns = common_feature_columns(datasets)
    reduced_feature_list = reduced_feature_columns(all_feature_columns, datasets=datasets)
    named_feature_sets = [
        ("All-Features", all_feature_columns),
        ("Reduced-Features", reduced_feature_list),
    ]

    dataframes_by_day, dataset_keys_by_day = group_datasets_by_day(datasets)
    day_names = sorted(dataframes_by_day)

    results_rows = []
    per_attack_type_results_rows = []

    for held_out_day in day_names:
        print(f"[HELD OUT DAY] {held_out_day}")

        training_days = [day_name for day_name in day_names if day_name != held_out_day]
        day_split_data = split_train_test_by_day(
            dataframes_by_day,
            dataset_keys_by_day,
            train_days=training_days,
            test_days=[held_out_day],
        )

        if day_split_data is None:
            print(f"[SKIP] Missing train/test data when holding out {held_out_day}.")
            continue

        combined_training_dataframe, held_out_test_dataframe = day_split_data[:2]

        sampled_training_dataframe, _ = sample_training_dataframe(
            # _ is solely for logging purposes.
            combined_training_dataframe,
            max_rows=MAX_POOLED_TRAINING_ROWS,
            sampling_name=f"pooled training rows for held-out day {held_out_day}",
        )

        training_dataframe = undersample_majority_class(sampled_training_dataframe)

        training_labels = training_dataframe["Is_attack"]

        if len(training_labels.unique()) < 2:
            print(
                f"[SKIP] pooled training data for held-out day {held_out_day} only has one class."
            )
            continue

        held_out_test_labels = held_out_test_dataframe["Is_attack"]

        for feature_set_name, selected_feature_columns in named_feature_sets:
            training_features = training_dataframe[selected_feature_columns]
            held_out_test_features = held_out_test_dataframe[selected_feature_columns]

            logistic_regression_scaler = StandardScaler()
            logistic_regression_scaler.fit(training_features)
            training_features_scaled = logistic_regression_scaler.transform(training_features)
            held_out_test_features_scaled = logistic_regression_scaler.transform(held_out_test_features)

            logistic_regression_model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )
            logistic_regression_model.fit(training_features_scaled, training_labels)

            logistic_regression_predictions = logistic_regression_model.predict(held_out_test_features_scaled)
            logistic_regression_metrics = evaluate_predictions(held_out_test_labels, logistic_regression_predictions)
            to_percentage(logistic_regression_metrics)

            results_rows.append(
                build_held_out_day_result_row(
                    experiment_name="Multi-Day-Held-Out-Day",
                    feature_set_name=feature_set_name,
                    model_name="Logistic-Regression",
                    held_out_day=held_out_day,
                    train_rows=len(training_features),
                    test_rows=len(held_out_test_features),
                    metrics=logistic_regression_metrics,
                )
            )

            append_held_out_day_per_attack_type_rows(
                results_rows=per_attack_type_results_rows,
                experiment_name="Multi-Day-Held-Out-Day",
                feature_set_name=feature_set_name,
                model_name="Logistic-Regression",
                held_out_day=held_out_day,
                train_rows=len(training_features),
                test_dataframe=held_out_test_dataframe,
                predictions=logistic_regression_predictions,
            )

            rf_training_features = training_dataframe[selected_feature_columns]
            rf_training_labels = training_dataframe["Is_attack"]

            random_forest_model = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=4,
            )
            random_forest_model.fit(rf_training_features, rf_training_labels)

            random_forest_predictions = random_forest_model.predict(held_out_test_features)
            random_forest_metrics = evaluate_predictions(held_out_test_labels,random_forest_predictions)
            to_percentage(random_forest_metrics)

            results_rows.append(
                build_held_out_day_result_row(
                    experiment_name="Multi-Day-Held-Out-Day",
                    feature_set_name=feature_set_name,
                    model_name="Random-Forest",
                    held_out_day=held_out_day,
                    train_rows=len(training_features),
                    test_rows=len(held_out_test_features),
                    metrics=random_forest_metrics,
                )
            )

            append_held_out_day_per_attack_type_rows(
                results_rows=per_attack_type_results_rows,
                experiment_name="Multi-Day-Held-Out-Day",
                feature_set_name=feature_set_name,
                model_name="Random-Forest",
                held_out_day=held_out_day,
                train_rows=len(training_features),
                test_dataframe=held_out_test_dataframe,
                predictions=random_forest_predictions,
            )

    results_dataframe = pd.DataFrame(results_rows)
    output_csv_path = RESULTS_DIR / "multi-day-experiment_metrics.csv"
    results_dataframe.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] wrote: {output_csv_path}")

    per_attack_type_results_dataframe = pd.DataFrame(per_attack_type_results_rows)
    per_attack_type_output_csv_path = RESULTS_DIR / "multi-day-per-attack-type_metrics.csv"
    per_attack_type_results_dataframe.to_csv(per_attack_type_output_csv_path, index=False)
    print(f"[SUCCESS] wrote: {per_attack_type_output_csv_path}")


if __name__ == "__main__":
    main()
