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
    RANDOM_STATE,
    build_held_out_day_result_row,
    evaluate_predictions,
    group_datasets_by_day,
    split_train_test_by_day,
    to_percentage,
)

# TRAINS ON THE POOLED NON-TEST DAYS, BUT REBALANCES THE TRAINING DATA BY ATTACK TYPE BEFORE FITTING.
# REPORTS OVERALL RESULTS TO SHOW WHETHER ATTACK-TYPE-AWARE SAMPLING HELPS ON THE HELD-OUT DAY.

MAX_BENIGN_ROWS = 50000
MAX_ATTACK_TYPE_ROWS = 20000


def sample_training_by_attack_type(training_dataframe):
    sampled_dataframes = []

    benign_rows = training_dataframe[training_dataframe["Label"] == "BENIGN"]
    if len(benign_rows) > MAX_BENIGN_ROWS:
        benign_rows = benign_rows.sample(
            n=MAX_BENIGN_ROWS,
            random_state=RANDOM_STATE,
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
                random_state=RANDOM_STATE,
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
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return sampled_training_dataframe
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
            print(f"[SKIP] Missing train/test data when holding out {held_out_day}.")
            continue

        (
            combined_training_dataframe,
            combined_held_out_test_dataframe,
            _training_dataset_keys,
            _held_out_test_dataset_keys,
        ) = split_result

        full_pooled_training_row_count = len(combined_training_dataframe)
        sampled_training_dataframe = sample_training_by_attack_type(
            combined_training_dataframe
        )

        print(
            f"[ATTACK-TYPE SAMPLING] Held-out day {held_out_day}: "
            f"Reduced training rows from {full_pooled_training_row_count} "
            f"to {len(sampled_training_dataframe)}"
        )

        training_labels = sampled_training_dataframe["Is_attack"]
        unique_training_classes = training_labels.unique()
        if len(unique_training_classes) < 2:
            print(
                f"[SKIP] pooled training data for held-out day {held_out_day} only has one class."
            )
            continue

        held_out_test_labels = combined_held_out_test_dataframe["Is_attack"]

        for feature_set_name, chosen_feature_columns in feature_sets:
            training_features = sampled_training_dataframe[chosen_feature_columns]
            held_out_test_features = combined_held_out_test_dataframe[chosen_feature_columns]

            logistic_regression_scaler = StandardScaler()
            logistic_regression_scaler.fit(training_features)
            training_features_scaled = logistic_regression_scaler.transform(training_features)
            held_out_test_features_scaled = logistic_regression_scaler.transform(held_out_test_features)

            logistic_regression_model = LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="lbfgs",
            )
            logistic_regression_model.fit(training_features_scaled, training_labels)

            logistic_regression_predictions = logistic_regression_model.predict(held_out_test_features_scaled)
            logistic_regression_metrics = evaluate_predictions(held_out_test_labels,logistic_regression_predictions)
            to_percentage(logistic_regression_metrics)

            results_rows.append(
                build_held_out_day_result_row(
                    experiment_name="Multi-Day-Attack-Type-Sampling",
                    model_name="Logistic-Regression",
                    feature_set_name=feature_set_name,
                    held_out_day=held_out_day,
                    train_rows=len(training_features),
                    test_rows=len(held_out_test_features),
                    metrics=logistic_regression_metrics,
                )
            )

            random_forest_model = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=4,
            )
            random_forest_model.fit(training_features, training_labels)

            random_forest_predictions = random_forest_model.predict(held_out_test_features)
            random_forest_metrics = evaluate_predictions(held_out_test_labels,random_forest_predictions)
            to_percentage(random_forest_metrics)

            results_rows.append(
                build_held_out_day_result_row(
                    experiment_name="Multi-Day-Attack-Type-Sampling",
                    model_name="Random-Forest",
                    feature_set_name=feature_set_name,
                    held_out_day=held_out_day,
                    train_rows=len(training_features),
                    test_rows=len(held_out_test_features),
                    metrics=random_forest_metrics,
                )
            )

    results_dataframe = pd.DataFrame(results_rows)
    output_csv_path = RESULTS_DIR / "multi-day-attack-type-sampling_metrics.csv"
    results_dataframe.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] wrote: {output_csv_path}")


if __name__ == "__main__":
    main()
