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
    append_train_test_day_per_attack_type_rows,
    build_train_test_day_result_row,
    combine_days,
    evaluate_predictions,
    group_datasets_by_day,
    sample_training_dataframe,
    to_percentage,
    undersample_majority_class,
)

# TRAINS EACH MODEL ON ONE FULL DAY AND TESTS IT ON EVERY OTHER DAY.
# REPORTS OVERALL RESULTS AND PER-ATTACK-TYPE RESULTS TO SHOW HOW WELL PERFORMANCE TRANSFERS ACROSS DAYS.


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
    per_attack_type_results_rows = []

    for train_day in day_names:
        train_result = combine_days(
            dataframes_by_day,
            dataset_keys_by_day,
            [train_day],
        )
        if train_result is None:
            print(f"[SKIP] Missing training data for {train_day}.")
            continue

        combined_training_dataframe, _training_dataset_keys = train_result
        sampled_training_dataframe, _ = sample_training_dataframe(
            combined_training_dataframe,
            max_rows=MAX_POOLED_TRAINING_ROWS,
            sampling_name=f"training rows for train day {train_day}",
        )
        training_dataframe = undersample_majority_class(sampled_training_dataframe)

        if len(training_dataframe) != len(sampled_training_dataframe):
            print(
                f"[UNDERSAMPLING] Train day {train_day}: "
                f"Reduced training rows from {len(sampled_training_dataframe)} "
                f"to {len(training_dataframe)}"
            )

        training_labels = training_dataframe["Is_attack"]
        if len(training_labels.unique()) < 2:
            print(f"[SKIP] {train_day} only has one class after sampling.")
            continue

        test_results_by_day = {}
        for test_day in day_names:
            if test_day == train_day:
                continue

            test_result = combine_days(
                dataframes_by_day,
                dataset_keys_by_day,
                [test_day],
            )
            if test_result is None:
                print(f"[SKIP] Missing test data for day {test_day}.")
                continue

            test_results_by_day[test_day] = test_result

        for feature_set_name, chosen_feature_columns in feature_sets:
            training_features = training_dataframe[chosen_feature_columns]

            logistic_regression_scaler = StandardScaler()
            logistic_regression_scaler.fit(training_features)
            training_features_scaled = logistic_regression_scaler.transform(training_features)

            logistic_regression_model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )
            logistic_regression_model.fit(
                training_features_scaled,
                training_labels,
            )

            rf_training_features = training_dataframe[chosen_feature_columns]
            rf_training_labels = training_dataframe["Is_attack"]

            random_forest_model = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=4,
            )
            random_forest_model.fit(rf_training_features, rf_training_labels)

            for test_day, test_result in test_results_by_day.items():
                combined_test_dataframe, _test_dataset_keys = test_result
                held_out_test_labels = combined_test_dataframe["Is_attack"]
                held_out_test_features = combined_test_dataframe[chosen_feature_columns]

                held_out_test_features_scaled = logistic_regression_scaler.transform(held_out_test_features)
                logistic_regression_predictions = logistic_regression_model.predict(held_out_test_features_scaled)
                logistic_regression_metrics = evaluate_predictions(held_out_test_labels,logistic_regression_predictions)
                to_percentage(logistic_regression_metrics)

                results_rows.append(
                    build_train_test_day_result_row(
                        experiment_name="Single-Day-Transfer",
                        feature_set_name=feature_set_name,
                        model_name="Logistic-Regression",
                        train_day=train_day,
                        test_day=test_day,
                        train_rows=len(training_features),
                        test_rows=len(held_out_test_features),
                        metrics=logistic_regression_metrics,
                    )
                )

                append_train_test_day_per_attack_type_rows(
                    results_rows=per_attack_type_results_rows,
                    experiment_name="Single-Day-Transfer",
                    feature_set_name=feature_set_name,
                    model_name="Logistic-Regression",
                    train_day=train_day,
                    test_day=test_day,
                    train_rows=len(training_features),
                    test_dataframe=combined_test_dataframe,
                    predictions=logistic_regression_predictions,
                )

                random_forest_predictions = random_forest_model.predict(held_out_test_features)
                random_forest_metrics = evaluate_predictions(held_out_test_labels,random_forest_predictions)
                to_percentage(random_forest_metrics)

                results_rows.append(
                    build_train_test_day_result_row(
                        experiment_name="Single-Day-Transfer",
                        feature_set_name=feature_set_name,
                        model_name="Random-Forest",
                        train_day=train_day,
                        test_day=test_day,
                        train_rows=len(training_features),
                        test_rows=len(held_out_test_features),
                        metrics=random_forest_metrics,
                    )
                )

                append_train_test_day_per_attack_type_rows(
                    results_rows=per_attack_type_results_rows,
                    experiment_name="Single-Day-Transfer",
                    feature_set_name=feature_set_name,
                    model_name="Random-Forest",
                    train_day=train_day,
                    test_day=test_day,
                    train_rows=len(training_features),
                    test_dataframe=combined_test_dataframe,
                    predictions=random_forest_predictions,
                )

    results_df = pd.DataFrame(results_rows)
    out_csv = RESULTS_DIR / "single-day-transfer-experiment_metrics.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] wrote: {out_csv}")

    per_attack_type_results_df = pd.DataFrame(per_attack_type_results_rows)
    per_attack_type_out_csv = RESULTS_DIR / "single-day-transfer-per-attack-type_metrics.csv"
    per_attack_type_results_df.to_csv(per_attack_type_out_csv, index=False)
    print(f"[SUCCESS] wrote: {per_attack_type_out_csv}")


if __name__ == "__main__":
    main()
