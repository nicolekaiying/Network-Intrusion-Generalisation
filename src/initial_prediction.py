import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_path = PROJECT_ROOT / "data" / "processed" / "Tuesday-WorkingHours-Clean.csv"
model_path = PROJECT_ROOT / 'models' / "RF-Friday-WorkingHours-Afternoon-DDos.pkl"

df = pd.read_csv(data_path)

features = df.drop(columns=["Label", "Is_attack"])

model = joblib.load(model_path)

# Select sample rows so that output is short and concise
sample = features.sample(1, random_state=42) # Select the first five rows

predictions = model.predict(sample)

for x in range(len(sample)):
    predicted_class = predictions[x]
    label = "ATTACK" if predicted_class == 1 else "BENIGN"

    print(f"ROW {x}: Prediction = {label}")