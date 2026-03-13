import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_feature_alignment import PROCESSED_DIR, RESULTS_DIR, common_feature_columns, load_processed_csvs


def evaluate_predictions(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    tn = int(cm[0, 0]) #Benign correctly predicted as Benign.
    fp = int(cm[0, 1]) #Benign incorrectly predicted as Attack.
    fn = int(cm[1, 0]) #Attack incorrectly predicted as Benign.
    tp = int(cm[1, 1]) #Attack correctly predicted as Attack.

    metrics = {}
    # Overall correctness.
    metrics["accuracy"] = accuracy_score(true_labels, predicted_labels)
    #When I predicted Attack, how often is that True?
    metrics["precision"] = precision_score(true_labels, predicted_labels, zero_division=0)
    #Of all Attacks, how many did I catch?
    metrics["recall"] = recall_score(true_labels, predicted_labels, zero_division=0)
    #Balance between precision and recall.
    metrics["f1"] = f1_score(true_labels, predicted_labels, zero_division=0)
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp
    return metrics


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_processed_csvs(PROCESSED_DIR)

    feature_columns = common_feature_columns(datasets)

    results_rows = []

    training_dataset_keys = sorted(datasets.keys())
    testing_dataset_keys = sorted(datasets.keys())


    for train_key in training_dataset_keys:
        train_df = datasets[train_key]

        training_features_full_dataset = train_df[feature_columns]
        training_labels_full_dataset = train_df["Is_attack"]

        #Some datasets (e.g., all-BENIGN) cannot be used to train a classifier.
        unique_classes = training_labels_full_dataset.unique()
        if len(unique_classes) < 2:
            print(f"[SKIPPED]: {train_key}, Training dataset only has one class.")
            continue

        training_features_split = None
        in_domain_test_features = None
        training_labels_split = None
        in_domain_test_labels = None
        can_do_in_domain = True

        try:
            split_result = train_test_split(
                training_features_full_dataset,
                training_labels_full_dataset,
                test_size=0.2,
                random_state=42,
                stratify=training_labels_full_dataset,
            )
            training_features_split = split_result[0]
            #Split training_features_full_dataset into 80/20. 80% for training.
            in_domain_test_features = split_result[1]
            #Split 20% of training_features_full_dataset for testing.
            training_labels_split = split_result[2]
            #Split training_labels_full_dataset into 80/20. 80% for training.
            in_domain_test_labels = split_result[3]
            #Split 20% of training_labels_full_dataset for testing.

        except ValueError as exc:
        #Catch exceptions.
            print(f"[WARNING] Could not create in-domain split for {train_key}: {exc}")
            print("[WARNING] Training on the full dataset and skipping in-domain evaluation.")
            training_features_split = training_features_full_dataset
            training_labels_split = training_labels_full_dataset
            can_do_in_domain = False

        # print(f"[TRAINING]: {train_key} (Train Rows = {len(training_features_split)})")


# ----------------------------------------------------------------------------------------------------------------------


        # LOGISTIC REGRESSION


        logistic_regression_scaler = StandardScaler()
        #Standardize each feature columns by computing its mean and standard deviation.
        logistic_regression_scaler.fit(training_features_split)
        #Learn about each feature column's mean and standard deviation from training_features_split.

        training_features_split_scaled = logistic_regression_scaler.transform(training_features_split)
        #Take the data learned from lr_scaler.fit and convert each value into its scaled version.

        logistic_regression_model = LogisticRegression(
        #Creates logistic regression model object.
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            verbose=1,
        )
        logistic_regression_model.fit(training_features_split_scaled, training_labels_split)
        #The model learns the correlation between the scaled features and its label.

        # IN DOMAIN: TRAIN ON ONE DAY, THEN TRAIN IT ON A HELD OUT SPLIT FROM THE SAME DAY.

        if can_do_in_domain:
            in_domain_test_features_scaled = logistic_regression_scaler.transform(in_domain_test_features)
            #Used the same scaler from line 91 to prevent test-set information leak.
            logistic_regression_in_prediction = logistic_regression_model.predict(in_domain_test_features_scaled)
            #Asks trained logistic regression model to make predictions.
            logistic_regression_in_metrics = evaluate_predictions(in_domain_test_labels, logistic_regression_in_prediction)
            #Compare real labels: in_domain_test_labels with predicted labels: logistic_regression_in_prediction.

            row = {}
            row["model"] = "Logistic Regression"
            row["train_dataset"] = train_key
            row["test_dataset"] = train_key
            row["evaluation"] = "in_domain"
            row["train_rows"] = len(training_features_split)
            row["test_rows"] = len(in_domain_test_features)
            row["accuracy"] = logistic_regression_in_metrics["accuracy"]
            row["precision"] = logistic_regression_in_metrics["precision"]
            row["recall"] = logistic_regression_in_metrics["recall"]
            row["f1"] = logistic_regression_in_metrics["f1"]
            row["tn"] = logistic_regression_in_metrics["tn"]
            row["fp"] = logistic_regression_in_metrics["fp"]
            row["fn"] = logistic_regression_in_metrics["fn"]
            row["tp"] = logistic_regression_in_metrics["tp"]
            results_rows.append(row)

            print(f"[LOGISTIC REGRESSION TRAINED]: {train_key}")


        # CROSS-DOMAIN: TRAIN ON ONE DAY AND TEST IT ON OTHER DAYS.

        for test_key in testing_dataset_keys:
            if test_key == train_key:
            #If test dataset is the same as train dataset, skip it because it's already handled during in_domain.
                continue

            test_df = datasets[test_key]
            cross_domain_test_features = test_df[feature_columns]
            cross_domain_test_labels = test_df["Is_attack"]

            cross_domain_test_features_scaled = logistic_regression_scaler.transform(cross_domain_test_features)
            logistic_regression_test_predict = logistic_regression_model.predict(cross_domain_test_features_scaled)
            #Make predictions on the new test dataset.
            lr_test_metrics = evaluate_predictions(cross_domain_test_labels, logistic_regression_test_predict)
            #Compare real labels: cross_domain_test_labels with predicted labels: logistic_regression_test_predict.

            row = {}
            row["model"] = "Logistic_regression"
            row["train_dataset"] = train_key
            row["test_dataset"] = test_key
            row["evaluation"] = "cross_domain"
            row["train_rows"] = len(training_features_split)
            row["test_rows"] = len(cross_domain_test_features)
            row["accuracy"] = lr_test_metrics["accuracy"]
            row["precision"] = lr_test_metrics["precision"]
            row["recall"] = lr_test_metrics["recall"]
            row["f1"] = lr_test_metrics["f1"]
            row["tn"] = lr_test_metrics["tn"]
            row["fp"] = lr_test_metrics["fp"]
            row["fn"] = lr_test_metrics["fn"]
            row["tp"] = lr_test_metrics["tp"]
            results_rows.append(row)

            print(f"[TESTED]: {test_key}")


# ----------------------------------------------------------------------------------------------------------------------


        # RANDOM FOREST


        random_forest_model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        random_forest_model.fit(training_features_split, training_labels_split)

        # IN-DOMAIN

        if can_do_in_domain:
            random_forest_in_prediction = random_forest_model.predict(in_domain_test_features)
            random_forest_in_metrics = evaluate_predictions(in_domain_test_labels, random_forest_in_prediction)
            #Compares true labels: in_domain_test_labels with prediction labels: random_forest_in_prediction.

            row = {}
            row["model"] = "Random_forest"
            row["train_dataset"] = train_key
            row["test_dataset"] = train_key
            row["evaluation"] = "in_domain"
            row["train_rows"] = len(training_features_split)
            row["test_rows"] = len(in_domain_test_features)
            row["accuracy"] = random_forest_in_metrics["accuracy"]
            row["precision"] = random_forest_in_metrics["precision"]
            row["recall"] = random_forest_in_metrics["recall"]
            row["f1"] = random_forest_in_metrics["f1"]
            row["tn"] = random_forest_in_metrics["tn"]
            row["fp"] = random_forest_in_metrics["fp"]
            row["fn"] = random_forest_in_metrics["fn"]
            row["tp"] = random_forest_in_metrics["tp"]
            results_rows.append(row)

            print(f"[RANDOM FOREST TRAINED]: {train_key}")

        # CROSS-DOMAIN

        for test_key in testing_dataset_keys:
            if test_key == train_key:
                continue

            test_df = datasets[test_key]
            cross_domain_test_features = test_df[feature_columns]
            cross_domain_test_labels = test_df["Is_attack"]

            rf_test_pred = random_forest_model.predict(cross_domain_test_features)
            rf_test_metrics = evaluate_predictions(
                cross_domain_test_labels, rf_test_pred
            )

            row = {}
            row["model"] = "Random_forest"
            row["train_dataset"] = train_key
            row["test_dataset"] = test_key
            row["evaluation"] = "cross_domain"
            row["train_rows"] = len(training_features_split)
            row["test_rows"] = len(cross_domain_test_features)
            row["accuracy"] = rf_test_metrics["accuracy"]
            row["precision"] = rf_test_metrics["precision"]
            row["recall"] = rf_test_metrics["recall"]
            row["f1"] = rf_test_metrics["f1"]
            row["tn"] = rf_test_metrics["tn"]
            row["fp"] = rf_test_metrics["fp"]
            row["fn"] = rf_test_metrics["fn"]
            row["tp"] = rf_test_metrics["tp"]
            results_rows.append(row)

            print(f"[TRAINED]: {test_key}")


# ----------------------------------------------------------------------------------------------------------------------

    results_df = pd.DataFrame(results_rows)
    out_csv = RESULTS_DIR / "experiment_metrics.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] wrote: {out_csv}")


if __name__ == "__main__":
    main()
