# Cross-Day Intrusion Detection Experiments

This project studies how well machine learning models generalise across different days of network traffic in the CICIDS2017 dataset.

The main research questions are:

1. How well does a model generalise when it is trained on one whole day and tested on another whole day?
2. Does pooling several training days improve cross-day performance and stability?
3. Do threshold tuning and attack-type-aware sampling improve the pooled multi-day setup?

## REPOSITORY STRUCTURE

### MAIN EXPERIMENT SCRIPTS

- `src/preparation.py`
  Cleans the raw CICIDS2017 CSV files and writes processed files to `data/processed/`.

- `src/single_day_transfer_experiments.py`
  Trains each model on one full day and tests it on every other day.

- `src/multi_day_experiments.py`
  Holds out one full day for testing and pools the remaining days for training.

- `src/multi_day_threshold_tuning_experiments.py`
  Uses a validation split on the pooled training data to choose a decision threshold before testing on the held-out day.

- `src/multi_day_attack_type_sampling_experiments.py`
  Rebalances the pooled training data by attack type before training and testing on the held-out day.

- `src/experiment_helpers.py`
  Shared preprocessing, sampling, evaluation, and row-building helpers used by the experiment scripts.

### INITIAL BASIC PIPELINE SCRIPTS

These scripts were kept to show the earlier end-to-end testing pipeline before the final experiment scripts were developed:

- `src/initial_train.py`
- `src/initial_evaluation.py`
- `src/initial_prediction.py`

## DATASET NOTE

The `data/` folder is not included in this repository because the raw and processed CICIDS2017 CSV files are too large for normal Git/GitHub storage.

To run the project locally:

1. Place the original dataset CSV files in `data/raw/`
2. Run `src/preparation.py`
3. This will create cleaned files in `data/processed/`

The project expects day-level CSV files such as:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-WorkingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

## REQUIREMENTS

This project was run with Python 3.11.

Install the main dependencies with:

```bash
pip install pandas numpy scikit-learn joblib
```

If you are using a virtual environment, activate it before installing the packages.

## HOW-TO RUN

Prepare the cleaned files:

```bash
python3.11 src/preparation.py
```

Run the single-day transfer baseline:

```bash
python3.11 src/single_day_transfer_experiments.py
```

Run the pooled multi-day baseline:

```bash
python3.11 src/multi_day_experiments.py
```

Run the threshold-tuning extension:

```bash
python3.11 src/multi_day_threshold_tuning_experiments.py
```

Run the attack-type-aware sampling extension:

```bash
python3.11 src/multi_day_attack_type_sampling_experiments.py
```

## RESULTS

Final experiment outputs are stored in `results/`.

Important files include:

- `results/single-day-transfer-experiment_metrics.csv`
- `results/single-day-transfer-per-attack-type_metrics.csv`
- `results/multi-day-experiment_metrics.csv`
- `results/multi-day-per-attack-type_metrics.csv`
- `results/multi-day-threshold-tuning_metrics.csv`
- `results/multi-day-attack-type-sampling_metrics.csv`

Robustness outputs are stored in:

- `results/robustness_analysis_multiday/`
- `results/robustness_analysis_singleday/`

## NOTES

- The main robust finding in this project is the pooled multi-day baseline.
- The single-day transfer experiment was also re-run across multiple seeds, but it was more variable than the pooled setup.
- The threshold-tuning and attack-type-aware sampling scripts are extension experiments and should be interpreted more cautiously than the main pooled baseline.
