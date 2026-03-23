import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_feature_alignment import PROCESSED_DIR, RESULTS_DIR, common_feature_columns, load_processed_csvs
from experiment_helpers import (
    append_per_attack_type_rows,
    build_metrics_row,
    evaluate_predictions,
    to_percentage,
)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_processed_csvs(PROCESSED_DIR)

    feature_columns = common_feature_columns(datasets)

    results_rows = []
    per_attack_type_results_rows = []

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
            max_iter=300,
            class_weight="balanced",
            solver="liblinear",
            verbose=0,
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
            to_percentage(logistic_regression_in_metrics)

            row = build_metrics_row(
                model_name="logistic_regression",
                train_dataset=train_key,
                test_dataset=train_key,
                evaluation_name="in_domain",
                train_rows=len(training_features_split),
                test_rows=len(in_domain_test_features),
                metrics=logistic_regression_in_metrics,
            )
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
            to_percentage(lr_test_metrics)

            row = build_metrics_row(
                model_name="logistic_regression",
                train_dataset=train_key,
                test_dataset=test_key,
                evaluation_name="cross_domain",
                train_rows=len(training_features_split),
                test_rows=len(cross_domain_test_features),
                metrics=lr_test_metrics,
            )
            results_rows.append(row)

            append_per_attack_type_rows(
                results_rows=per_attack_type_results_rows,
                model_name="logistic_regression",
                train_dataset=train_key,
                test_dataset=test_key,
                train_rows=len(training_features_split),
                test_dataframe=test_df,
                predictions=logistic_regression_test_predict,
            )

            print(f"[TESTED]: {test_key}")


# ----------------------------------------------------------------------------------------------------------------------


        # RANDOM FOREST


        random_forest_model = RandomForestClassifier(
            n_estimators=100,
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
            to_percentage(random_forest_in_metrics)

            row = build_metrics_row(
                model_name="random_forest",
                train_dataset=train_key,
                test_dataset=train_key,
                evaluation_name="in_domain",
                train_rows=len(training_features_split),
                test_rows=len(in_domain_test_features),
                metrics=random_forest_in_metrics,
            )
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
            to_percentage(rf_test_metrics)

            row = build_metrics_row(
                model_name="random_forest",
                train_dataset=train_key,
                test_dataset=test_key,
                evaluation_name="cross_domain",
                train_rows=len(training_features_split),
                test_rows=len(cross_domain_test_features),
                metrics=rf_test_metrics,
            )
            results_rows.append(row)

            append_per_attack_type_rows(
                results_rows=per_attack_type_results_rows,
                model_name="random_forest",
                train_dataset=train_key,
                test_dataset=test_key,
                train_rows=len(training_features_split),
                test_dataframe=test_df,
                predictions=rf_test_pred,
            )

            print(f"[TRAINED]: {test_key}")


# ----------------------------------------------------------------------------------------------------------------------

    results_df = pd.DataFrame(results_rows)
    out_csv = RESULTS_DIR / "single-day-cross-day-experiment_metrics.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] wrote: {out_csv}")

    per_attack_type_results_df = pd.DataFrame(per_attack_type_results_rows)
    per_attack_type_out_csv = RESULTS_DIR / "single-day-cross-day-per-attack-type_metrics.csv"
    per_attack_type_results_df.to_csv(per_attack_type_out_csv, index=False)
    print(f"[SUCCESS] wrote: {per_attack_type_out_csv}")


if __name__ == "__main__":
    main()
