import os

# base_dir = "experiments/hyperparameter_tuning/hetrec2011-lastfm-2k/grit_tuned"  # Change this to your directory path
# base_dir = "experiments/hyperparameter_tuning/industrial_and_scientific/grit_tuned"  # Change this to your directory path
base_dir = "experiments/hyperparameter_tuning/ml-100k/grit_tuned"  # Change this to your directory path

for root, dirs, _ in os.walk(base_dir):
    for dir_name in dirs:
        if "hypertune" in dir_name:
            old_path = os.path.join(root, dir_name)
            new_name = dir_name.replace("_c_hypertune", "")
            new_path = os.path.join(root, new_name)

            print(f"Renaming:\n{old_path}\n→ {new_path}")
            os.rename(old_path, new_path)