import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import seaborn as sns

# --- Dataset files ---
files = [
    r'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\ml-1m\grit_hypertune_results.xlsx',
    r'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\ml-100k\grit_hypertune_results.xlsx',
    r'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\i_a_s\grit_hypertune_results.xlsx',
    r'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\v_g\grit_hypertune_results.xlsx',
    r'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\c_a_v\grit_hypertune_results.xlsx'
]

# Map short names to display names
dataset_map = {
    'ml_1m': 'MovieLens 1M',
    'ml_100k': 'MovieLens 100K',
    'v_g': 'Video Games',
    'i_a_s': 'Industrial & Scientific',
    'c_a_v': 'CDs & Vinyl'
}
dataset_keys = ['ml_1m','ml_100k','v_g','i_a_s','c_a_v']
dataset_names = [dataset_map[k] for k in dataset_keys]
all_groups = set()
all_betas = set()
tables = []

for f in files:
    T = pd.read_excel(f, sheet_name='val')
    T['avgScore'] = (T['Recall@10'] + T['MRR@10']) / 2
    all_groups.update(T['num_groups'].unique())
    all_betas.update(T['beta'].unique())
    tables.append(T)

all_groups = sorted(list(all_groups))
all_betas = sorted(list(all_betas))

# --- Build matrices with max scores ---
group_mat = np.full((len(files), len(all_groups)), np.nan)
beta_mat  = np.full((len(files), len(all_betas)), np.nan)

for i, T in enumerate(tables):
    for j, g in enumerate(all_groups):
        mask = T['num_groups'] == g
        if mask.any():
            # Take the max score instead of mean
            group_mat[i, j] = T.loc[mask, 'avgScore'].max()
    for j, b in enumerate(all_betas):
        mask = T['beta'] == b
        if mask.any():
            beta_mat[i, j] = T.loc[mask, 'avgScore'].max()

# --- Color palette (normalized RGB) ---
palette = [
    "#FFF44F",  # Lemon Yellow
    "#FF6F91",  # Blush Pink
    "#87CEEB",  # Sky Blue
    "#8EEA8E",  # Light Green (0.56, 0.93, 0.56)
    "#FACCA0",  # Peach       (0.98, 0.80, 0.69)
    "#AAD5E6",  # Light Cyan  (0.67, 0.84, 0.90)
]

# --- Function to plot grouped bars without gaps ---
def plot_grouped_bars(mat, dataset_names, param_values, xlabel, title, palette, save_path=None):
    n_datasets, n_params = mat.shape

    # Convert matrix to long-format DataFrame for seaborn
    data = []
    for i in range(n_datasets):
        for j in range(n_params):
            data.append({
                "dataset": dataset_names[i],
                "param": str(param_values[j]),
                "value": mat[i, j]
            })
    df_long = pd.DataFrame(data)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    ax = sns.barplot(
        data=df_long,
        x="dataset",
        y="value",
        hue="param",
        palette=palette,
        edgecolor="black",
        linewidth=0.5,
        dodge=True,
        ci=None,
        width=0.5
    )

    # X-axis formatting
    ax.set_xticklabels(
        [lbl.get_text() for lbl in ax.get_xticklabels()],
        rotation=20, ha="center",
        fontsize=14, fontweight="bold", family="Times New Roman"
    )
    plt.xlabel(xlabel, fontsize=14, fontweight="bold", family="Times New Roman")
    plt.ylabel("Score", fontsize=14, fontweight="bold", family="Times New Roman")

    # Y-axis scaling: round up to nearest 0.02
    max_val = np.nanmax(mat)
    y_max = np.ceil((max_val + 0.02) / 0.02) * 0.02
    plt.yticks(
        np.arange(0, y_max + 0.001, 0.02),
        fontsize=14, fontweight="bold", family="Times New Roman"
    )
    plt.ylim(0, y_max)

    # Grid
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Legend above
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=min(8, n_params),
        frameon=False,
        prop=font_manager.FontProperties(family="Times New Roman", size=14, weight="bold")
    )

    # Title (optional)
    # plt.title(title, fontsize=14, fontweight="bold", family="Times New Roman")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# --- Plot ---
plot_grouped_bars(group_mat, dataset_names, all_groups, 'Dataset', 'Effect of κ across datasets', palette, save_path='experiments/ablation/combined/hyper_k.png')
plot_grouped_bars(beta_mat, dataset_names, all_betas, 'Dataset', 'Effect of β across datasets', palette, save_path='experiments/ablation/combined/hyper_beta.png')
