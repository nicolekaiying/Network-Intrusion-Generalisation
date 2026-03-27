import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
MAX_POOLED_TRAINING_ROWS = 500000
UNDERSAMPLING_RATIO = 3


def evaluate_predictions(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    metrics = {}
    metrics["accuracy"] = accuracy_score(true_labels, predicted_labels)
    metrics["precision"] = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )
    metrics["recall"] = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )
    metrics["f1"] = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp
    return metrics


def to_percentage(metrics_dict):
    for key in ("accuracy", "precision", "recall", "f1"):
        if key in metrics_dict:
            metrics_dict[key] = round(metrics_dict[key] * 100, 4)


def build_held_out_day_result_row(
    experiment_name,
    feature_set_name,
    model_name,
    held_out_day,
    train_rows,
    test_rows,
    metrics,
):
    row = {}
    row["Experiment"] = experiment_name
    row["Feature-Set"] = feature_set_name
    row["Model"] = model_name
    row["Test-Day"] = held_out_day
    row["Train-Rows"] = train_rows
    row["Test-Rows"] = test_rows
    row["Accuracy"] = metrics["accuracy"]
    row["Precision"] = metrics["precision"]
    row["Recall"] = metrics["recall"]
    row["F1"] = metrics["f1"]
    return row


def append_held_out_day_per_attack_type_rows(
    results_rows,
    experiment_name,
    feature_set_name,
    model_name,
    held_out_day,
    train_rows,
    test_dataframe,
    predictions,
):
    attack_labels_in_test_dataset = sorted(test_dataframe["Label"].dropna().unique())

    for attack_label in attack_labels_in_test_dataset:
        if attack_label == "BENIGN":
            continue

        per_attack_type_subset_dataframe = test_dataframe[
            (test_dataframe["Label"] == "BENIGN") |
            (test_dataframe["Label"] == attack_label)
        ].copy()

        per_attack_type_true_labels = per_attack_type_subset_dataframe["Is_attack"]
        per_attack_type_predictions = predictions[
            per_attack_type_subset_dataframe.index
        ]
        per_attack_type_metrics = evaluate_predictions(
            per_attack_type_true_labels,
            per_attack_type_predictions,
        )
        to_percentage(per_attack_type_metrics)

        attack_row_count = len(
            per_attack_type_subset_dataframe[
                per_attack_type_subset_dataframe["Label"] == attack_label
            ]
        )

        row = {}
        row["Experiment"] = experiment_name
        row["Feature-Set"] = feature_set_name
        row["Model"] = model_name
        row["Test-Day"] = held_out_day
        row["Attack-Type"] = attack_label
        row["Attack-Rows"] = attack_row_count
        row["Train-Rows"] = train_rows
        row["Test-Rows"] = len(per_attack_type_subset_dataframe)
        row["Precision"] = per_attack_type_metrics["precision"]
        row["Recall"] = per_attack_type_metrics["recall"]
        row["F1"] = per_attack_type_metrics["f1"]
        results_rows.append(row)


def build_train_test_day_result_row(
    experiment_name,
    feature_set_name,
    model_name,
    train_day,
    test_day,
    train_rows,
    test_rows,
    metrics,
):
    row = {}
    row["Experiment"] = experiment_name
    row["Feature-Set"] = feature_set_name
    row["Model"] = model_name
    row["Train-Day"] = train_day
    row["Test-Day"] = test_day
    row["Train-Rows"] = train_rows
    row["Test-Rows"] = test_rows
    row["Accuracy"] = metrics["accuracy"]
    row["Precision"] = metrics["precision"]
    row["Recall"] = metrics["recall"]
    row["F1"] = metrics["f1"]
    return row


def append_train_test_day_per_attack_type_rows(
    results_rows,
    experiment_name,
    feature_set_name,
    model_name,
    train_day,
    test_day,
    train_rows,
    test_dataframe,
    predictions,
):
    attack_labels_in_test_dataset = sorted(test_dataframe["Label"].dropna().unique())

    for attack_label in attack_labels_in_test_dataset:
        if attack_label == "BENIGN":
            continue

        per_attack_type_subset_dataframe = test_dataframe[
            (test_dataframe["Label"] == "BENIGN") |
            (test_dataframe["Label"] == attack_label)
        ].copy()

        per_attack_type_true_labels = per_attack_type_subset_dataframe["Is_attack"]
        per_attack_type_predictions = predictions[
            per_attack_type_subset_dataframe.index
        ]
        per_attack_type_metrics = evaluate_predictions(
            per_attack_type_true_labels,
            per_attack_type_predictions,
        )
        to_percentage(per_attack_type_metrics)

        attack_row_count = len(
            per_attack_type_subset_dataframe[
                per_attack_type_subset_dataframe["Label"] == attack_label
            ]
        )

        row = {}
        row["Experiment"] = experiment_name
        row["Feature-Set"] = feature_set_name
        row["Model"] = model_name
        row["Train-Day"] = train_day
        row["Test-Day"] = test_day
        row["Attack-Type"] = attack_label
        row["Attack-Rows"] = attack_row_count
        row["Train-Rows"] = train_rows
        row["Test-Rows"] = len(per_attack_type_subset_dataframe)
        row["Precision"] = per_attack_type_metrics["precision"]
        row["Recall"] = per_attack_type_metrics["recall"]
        row["F1"] = per_attack_type_metrics["f1"]
        results_rows.append(row)


def undersample_majority_class(training_dataframe):
    benign_rows = training_dataframe[training_dataframe["Is_attack"] == 0]
    attack_rows = training_dataframe[training_dataframe["Is_attack"] == 1]

    benign_row_count = len(benign_rows)
    attack_row_count = len(attack_rows)

    if benign_row_count == 0 or attack_row_count == 0:
        return training_dataframe

    if benign_row_count > attack_row_count:
        larger_class_rows = benign_rows
        smaller_class_rows = attack_rows
    else:
        larger_class_rows = attack_rows
        smaller_class_rows = benign_rows

    target_larger_class_size = min(
        len(larger_class_rows),
        len(smaller_class_rows) * UNDERSAMPLING_RATIO,
    )

    sampled_larger_class_rows = larger_class_rows.sample(
        n=target_larger_class_size,
        random_state=RANDOM_STATE,
    )

    balanced_training_dataframe = pd.concat(
        [sampled_larger_class_rows, smaller_class_rows],
        axis=0,
        ignore_index=True,
    )

    balanced_training_dataframe = balanced_training_dataframe.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return balanced_training_dataframe


def group_datasets_by_day(datasets):
    # Bucket each dataset under its day name so later code can split by whole days.
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


def combine_days(dataframes_by_day, dataset_keys_by_day, selected_days):
    # Merge all datasets from the requested days into one dataframe plus one key list.
    combined_dataframes = []
    combined_keys = []

    for day_name in selected_days:
        if day_name not in dataframes_by_day:
            continue

        combined_dataframes.extend(dataframes_by_day[day_name])
        combined_keys.extend(dataset_keys_by_day[day_name])

    if not combined_dataframes:
        return None

    combined_dataframe = pd.concat(
        combined_dataframes,
        axis=0,
        ignore_index=True,
    )
    return combined_dataframe, combined_keys


def split_train_test_by_day(
    dataframes_by_day,
    dataset_keys_by_day,
    train_days,
    test_days,
):
    # Build one combined dataframe for the training days and another for the test days.
    combined_train_data = combine_days(
        dataframes_by_day,
        dataset_keys_by_day,
        train_days,
    )
    combined_test_data = combine_days(
        dataframes_by_day,
        dataset_keys_by_day,
        test_days,
    )

    if combined_train_data is None or combined_test_data is None:
        return None

    combined_training_dataframe, training_keys = combined_train_data
    combined_test_dataframe, test_keys = combined_test_data
    return combined_training_dataframe, combined_test_dataframe, training_keys, test_keys


def sample_training_dataframe(training_dataframe, max_rows, sampling_name):
    full_training_row_count = len(training_dataframe)

    if full_training_row_count > max_rows:
        print(
            f"[SAMPLING] Reducing {sampling_name} rows "
            f"from {full_training_row_count} to {max_rows}"
        )

        sampled_split_result = train_test_split(
            training_dataframe,
            train_size=max_rows,
            random_state=RANDOM_STATE,
            stratify=training_dataframe["Is_attack"],
        )
        sampled_training_dataframe = sampled_split_result[0]
    else:
        sampled_training_dataframe = training_dataframe

    return sampled_training_dataframe, full_training_row_count


def predictions_from_threshold(attack_probabilities, threshold):
    predicted_labels = (attack_probabilities >= threshold).astype(int)
    return predicted_labels


def choose_best_threshold(
    true_labels,
    attack_probabilities,
    threshold_values,
    minimum_precision,
):
    best_threshold = None
    best_metrics = None
    selection_mode = "minimum_precision_then_best_recall"

    for threshold in threshold_values:
        predicted_labels = predictions_from_threshold(
            attack_probabilities,
            threshold,
        )
        metrics = evaluate_predictions(true_labels, predicted_labels)

        if metrics["precision"] < minimum_precision:
            continue

        if best_metrics is None or metrics["recall"] > best_metrics["recall"]:
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None:
        selection_mode = "best_f1_fallback"

        for threshold in threshold_values:
            predicted_labels = predictions_from_threshold(
                attack_probabilities,
                threshold,
            )
            metrics = evaluate_predictions(true_labels, predicted_labels)

            if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                best_threshold = threshold
                best_metrics = metrics

    return best_threshold, best_metrics, selection_mode
