# Testing model on test sets

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]

processed_csv = PROJECT_ROOT / "data" / "processed" / "Friday_WorkingHours-Afternoon-DDos-Clean.csv"

df = pd.read_csv(processed_csv)

# Drops Label and Is_Attack columns
features = df.drop(columns=["Label", "Is_attack"])

target = df["Is_attack"]

features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

rf_model = PROJECT_ROOT / "models" / "RF-Friday-WorkingHours-Afternoon-DDos.pkl"

model_test = joblib.load(rf_model)

# Based on each test row feature, predict 0 or 1
predictions = model_test.predict(features_test)
print(predictions)

# Accuracy of the model
accuracy = accuracy_score(target_test, predictions)
print(accuracy)

# Returns 2x2 table
cm = confusion_matrix(target_test, predictions) # Returns [TN, FP]
print(cm)                                               # [FN, TP]





