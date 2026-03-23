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
from experiment_helpers import evaluate_predictions, to_percentage

# This experiment trains the model on multiple days, then checks how well it
# generalises to one unseen held-out day.

MAX_POOLED_TRAINING_ROWS = 150000
RANDOM_STATE = 42


def undersample_majority_class(training_dataframe):
    benign_rows = training_dataframe[training_dataframe["Is_attack"] == 0]
    attack_rows = training_dataframe[training_dataframe["Is_attack"] == 1]

    benign_row_count = len(benign_rows)
    attack_row_count = len(attack_rows)

    if benign_row_count == 0 or attack_row_count == 0:
        # Do nothing if one class is missing.
        return training_dataframe

    if benign_row_count > attack_row_count:
        larger_class_rows = benign_rows
        smaller_class_rows = attack_rows
    else:
        larger_class_rows = attack_rows
        smaller_class_rows = benign_rows

    sampled_larger_class_rows = larger_class_rows.sample(
        # Take only as many rows as there are minority rows.
        n=len(smaller_class_rows),
        random_state=RANDOM_STATE,
    )

    balanced_training_dataframe = pd.concat(
        # Creates a balanced dataframe by stacking sampled_larger_class_rows and smaller_class_rows.
        [sampled_larger_class_rows, smaller_class_rows],
        axis=0,
        ignore_index=True,
    )

    balanced_training_dataframe = balanced_training_dataframe.sample(
        # Shuffles the rows.
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return balanced_training_dataframe


def group_datasets_by_day(datasets):
    dataframes_by_day = {}
    dataset_keys_by_day = {}

    for dataset_key, dataframe in datasets.items():
        day_name = dataset_key.split("-")[0]

        if day_name not in dataframes_by_day:
            dataframes_by_day[day_name] = []
            dataset_keys_by_day[day_name] = []

        dataframes_by_day[day_name].append(dataframe)
        dataset_keys_by_day[day_name].append(dataset_key)

    return dataframes_by_day, dataset_keys_by_day


def split_train_test_by_day(dataframes_by_day, dataset_keys_by_day, day_names, held_out_day):
    training_dataframes = []
    training_keys = []
    test_dataframes = []
    test_keys = []

    for day_name in day_names:
        if day_name == held_out_day:
            test_dataframes.extend(dataframes_by_day[day_name])
            test_keys.extend(dataset_keys_by_day[day_name])
        else:
            training_dataframes.extend(dataframes_by_day[day_name])
            training_keys.extend(dataset_keys_by_day[day_name])

    if not training_dataframes or not test_dataframes:
        return None

    combined_training_dataframe = pd.concat(
        training_dataframes,
        axis=0,
        ignore_index=True,
    )
    combined_test_dataframe = pd.concat(
        test_dataframes,
        axis=0,
        ignore_index=True,
    )

    return combined_training_dataframe, combined_test_dataframe, training_keys, test_keys


def sample_pooled_training_dataframe(combined_training_dataframe, held_out_day):
    full_pooled_training_row_count = len(combined_training_dataframe)

    if full_pooled_training_row_count > MAX_POOLED_TRAINING_ROWS:
        print(
            f"[SAMPLING] Reducing pooled training rows for held-out day {held_out_day} "
            f"from {full_pooled_training_row_count} to {MAX_POOLED_TRAINING_ROWS}"
        )

        sampled_split_result = train_test_split(
            combined_training_dataframe,
            train_size=MAX_POOLED_TRAINING_ROWS,
            random_state=RANDOM_STATE,
            stratify=combined_training_dataframe["Is_attack"],
        )
        sampled_training_dataframe = sampled_split_result[0]
    else:
        sampled_training_dataframe = combined_training_dataframe

    return sampled_training_dataframe, full_pooled_training_row_count


def build_metrics_row(
    experiment_name,
    feature_set_name,
    chosen_feature_columns,
    model_name,
    held_out_day,
    training_days_text,
    training_dataset_files_text,
    held_out_test_dataset_files_text,
    full_train_rows,
    sampled_train_rows,
    train_rows,
    test_rows,
    metrics,
):
    row = {}
    row["experiment"] = experiment_name
    row["feature_set"] = feature_set_name
    row["feature_count"] = len(chosen_feature_columns)
    row["model"] = model_name
    row["held_out_day"] = held_out_day
    row["training_days"] = training_days_text
    row["training_dataset_files"] = training_dataset_files_text
    row["test_dataset_files"] = held_out_test_dataset_files_text
    row["full_train_rows"] = full_train_rows
    row["sampled_train_rows"] = sampled_train_rows
    row["train_rows"] = train_rows
    row["test_rows"] = test_rows
    row["accuracy"] = metrics["accuracy"]
    row["precision"] = metrics["precision"]
    row["recall"] = metrics["recall"]
    row["f1"] = metrics["f1"]
    row["tn"] = metrics["tn"]
    row["fp"] = metrics["fp"]
    row["fn"] = metrics["fn"]
    row["tp"] = metrics["tp"]
    return row


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_processed_csvs(PROCESSED_DIR)
    all_feature_columns = common_feature_columns(datasets)
    reduced_columns = reduced_feature_columns(all_feature_columns, datasets=datasets)
    feature_sets = [
        ("all_features", all_feature_columns),
        ("reduced_features", reduced_columns),
    ]

    dataframes_by_day, dataset_keys_by_day = group_datasets_by_day(datasets)
    day_names = sorted(dataframes_by_day.keys())

    results_rows = []

    for held_out_day in day_names:
        print(f"[HELD OUT DAY] {held_out_day}")

        split_result = split_train_test_by_day(dataframes_by_day, dataset_keys_by_day, day_names, held_out_day)

        if split_result is None:
            print(f"[SKIP] missing train/test data when holding out {held_out_day}.")
            continue

        (
            combined_training_dataframe,
            combined_held_out_test_dataframe,
            training_dataset_keys,
            held_out_test_dataset_keys,
        ) = split_result

        sampled_training_dataframe, full_pooled_training_row_count = sample_pooled_training_dataframe(
            combined_training_dataframe,
            held_out_day,
        )

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
        if len(training_labels.unique()) < 2:
            print(
                f"[SKIP] pooled training data for held-out day {held_out_day} only has one class."
            )
            continue

        training_days = [day_name for day_name in day_names if day_name != held_out_day]
        training_days_text = ", ".join(training_days)
        training_dataset_files_text = ", ".join(sorted(training_dataset_keys))
        held_out_test_dataset_files_text = ", ".join(sorted(held_out_test_dataset_keys))

        print(
            f"[TRAINING ROWS] Held-out day {held_out_day}: "
            f"using {len(training_dataframe)} training rows"
        )

        held_out_test_labels = combined_held_out_test_dataframe["Is_attack"]

        for feature_set_name, chosen_feature_columns in feature_sets:
            training_features = training_dataframe[chosen_feature_columns]
            held_out_test_features = combined_held_out_test_dataframe[chosen_feature_columns]

            print(
                f"[FEATURE SET] {feature_set_name} "
                f"({len(chosen_feature_columns)} cols) for {held_out_day}"
            )

            logistic_regression_scaler = StandardScaler()
            logistic_regression_scaler.fit(training_features)
            training_features_scaled = logistic_regression_scaler.transform(training_features)
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

            results_rows.append(
                build_metrics_row(
                    experiment_name="multi_day_held_out_day",
                    feature_set_name=feature_set_name,
                    chosen_feature_columns=chosen_feature_columns,
                    model_name="logistic_regression",
                    held_out_day=held_out_day,
                    training_days_text=training_days_text,
                    training_dataset_files_text=training_dataset_files_text,
                    held_out_test_dataset_files_text=held_out_test_dataset_files_text,
                    full_train_rows=full_pooled_training_row_count,
                    sampled_train_rows=len(sampled_training_dataframe),
                    train_rows=len(training_features),
                    test_rows=len(held_out_test_features),
                    metrics=logistic_regression_metrics,
                )
            )

            random_forest_model = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            random_forest_model.fit(training_features, training_labels)

            random_forest_predictions = random_forest_model.predict(held_out_test_features)
            random_forest_metrics = evaluate_predictions(
                held_out_test_labels,
                random_forest_predictions,
            )
            to_percentage(random_forest_metrics)

            results_rows.append(
                build_metrics_row(
                    experiment_name="multi_day_held_out_day",
                    feature_set_name=feature_set_name,
                    chosen_feature_columns=chosen_feature_columns,
                    model_name="random_forest",
                    held_out_day=held_out_day,
                    training_days_text=training_days_text,
                    training_dataset_files_text=training_dataset_files_text,
                    held_out_test_dataset_files_text=held_out_test_dataset_files_text,
                    full_train_rows=full_pooled_training_row_count,
                    sampled_train_rows=len(sampled_training_dataframe),
                    train_rows=len(training_features),
                    test_rows=len(held_out_test_features),
                    metrics=random_forest_metrics,
                )
            )

    results_dataframe = pd.DataFrame(results_rows)
    output_csv_path = RESULTS_DIR / "multi_day_experiment_metrics.csv"
    results_dataframe.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] wrote: {output_csv_path}")


if __name__ == "__main__":
    main()
