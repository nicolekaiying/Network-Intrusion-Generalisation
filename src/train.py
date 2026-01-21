# Trains the model to learn patterns which separates 0 and 1

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Machine learning model chosen
import joblib # Tools to save/load models
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_path = PROJECT_ROOT / "data" / "processed" / "Friday_WorkingHours-Afternoon-DDos-Clean.csv"
df = pd.read_csv(data_path)

# What I want the model to learn to predict
target = df["Is_attack"]

# Drop columns that interfere with the training
features = df.drop(columns=["Label", "Is_attack"])

# This splits training and testing set into 80% and 20% respectively
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size = 0.2,
    random_state = 42,
    stratify=target
)

# Create the random forest model
model = RandomForestClassifier(
    n_estimators=200, # 200 decision trees
    random_state=42, # Default seed number
    n_jobs=-1 # (-1) Speedup training by using all available CPU cores, (1) Uses one core, (2) Uses two cores
)

# Essentially what this tells the model to do is to learn the patterns that separates 0 and 1 from the many example network traffics
model.fit(features_train, target_train)

# Saves it to a file to be reused later
model_path = PROJECT_ROOT / "models" / "RF-Friday-WorkingHours-Afternoon-DDos.pkl"

joblib.dump(model, model_path)


