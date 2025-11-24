import os
import yaml
import itertools
import pandas as pd
from train import train
from copy import deepcopy


# List available config files
base_yaml = "config/base.yaml"

with open(base_yaml, 'r') as f:
    base_config = yaml.safe_load(f)

print('******************** Dataset Selection ********************')
dataset_code = {'1': 'ml-1m', '2': 'ml-100k', '3': 'video_games', '4': 'cds_and_vinyl', '5': 'industrial_and_scientific'}

print("Available datasets:", dataset_code)
data_code = input("Enter the dataset code: ").strip()
base_config['dataset_code'] = dataset_code[data_code]

hyper_yaml_dir = "config/hyper"

available_configs = [f.replace('.yaml', '') for f in os.listdir(hyper_yaml_dir) if f.endswith('.yaml')]
print("Available models:", ", ".join(available_configs))
model_code = input("Enter the model code: ").strip()
hyper_yaml = os.path.join(hyper_yaml_dir, f"{model_code}.yaml")

with open(hyper_yaml, 'r') as f:
    param_grid = yaml.safe_load(f)

if "-" in base_config['dataset_code']:
    dataset_initials = base_config['dataset_code']   # keep as is
else:
    dataset_initials = "_".join([part[0] for part in base_config['dataset_code'].split("_")])

if not os.path.exists(f"experiments/hyperparameter_tuning/{dataset_initials}"):
    os.makedirs(f"experiments/hyperparameter_tuning/{dataset_initials}")

RESULT_FILE = f'experiments/hyperparameter_tuning/{dataset_initials}/{param_grid["model_code"][0]}_hypertune_results.xlsx'

# Load existing results if they exist
if os.path.exists(RESULT_FILE):
    existing_val_df = pd.read_excel(RESULT_FILE, sheet_name='val')
    existing_test_df = pd.read_excel(RESULT_FILE, sheet_name='test')
else:
    existing_val_df = pd.DataFrame()
    existing_test_df = pd.DataFrame()

# Add missing keys to existing DataFrames and initialize with 0
param_keys = list(param_grid.keys())
if not existing_val_df.empty:
    for key in param_keys:
        if key not in existing_val_df.columns:
            existing_val_df[key] = 0
            existing_test_df[key] = 0

# Keep a set of existing combinations to avoid duplicates
existing_keys = set()
if not existing_val_df.empty:
    key_cols = list(param_grid.keys())
    existing_keys = set(tuple(row[k] for k in key_cols) for _, row in existing_val_df.iterrows())

# Prepare to collect new results
val_records = []
test_records = []

# Generate all hyperparameter combinations
keys, values = zip(*param_grid.items())
all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

base_config['dataset_code'] = dataset_code[data_code]

for combo in all_combinations:
    combo_key = tuple(combo[k] for k in keys)
    if combo_key in existing_keys:
        print(f"Skipping already completed combination: {combo}")
        continue

    print(f"Running combo: {combo}")
    config = deepcopy(base_config)
    config.update(combo)
    parts = [f"{k.split('_')[-1][0]}_{v}" for k, v in combo.items()]
    export_path = (os.path.join(f"experiments/hyperparameter_tuning/{dataset_initials}/{config['model_code']}_tuned/", "_".join(parts)))

    # try:
    val_metrics, test_metrics = train(config, export_path)

    # Add hyperparams to metrics
    val_entry = {**combo, **val_metrics}
    test_entry = {**combo, **test_metrics}

    val_records.append(val_entry)
    test_records.append(test_entry)

    # Append to existing DataFrames
    existing_val_df = pd.concat([existing_val_df, pd.DataFrame([val_entry])], ignore_index=True)
    existing_test_df = pd.concat([existing_test_df, pd.DataFrame([test_entry])], ignore_index=True)

    # Save after each run
    with pd.ExcelWriter(RESULT_FILE, engine='openpyxl', mode='w') as writer:
        existing_val_df.to_excel(writer, sheet_name='val', index=False)
        existing_test_df.to_excel(writer, sheet_name='test', index=False)
    #
    # except Exception as e:
    #     print(f"Failed for combo {combo}: {e}")
