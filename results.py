import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def process_hypertune_results(hypertune_folder_path, output_file_path):
    """
    Process hyperparameter tuning results from multiple datasets and models.

    Args:
        hypertune_folder_path: Path to the hypertune folder containing dataset subfolders
        output_file_path: Path for the output Excel file
    """

    best_val_results = []
    best_test_results = []

    dataset_folders = [d for d in os.listdir(hypertune_folder_path)
                       if os.path.isdir(os.path.join(hypertune_folder_path, d))]

    print(f"Found {len(dataset_folders)} dataset folders: {dataset_folders}")

    for dataset_name in dataset_folders:
        dataset_path = os.path.join(hypertune_folder_path, dataset_name)
        xlsx_files = glob.glob(os.path.join(dataset_path, "*.xlsx"))

        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Found {len(xlsx_files)} Excel files")

        for xlsx_file in xlsx_files:
            model_name = Path(xlsx_file).stem

            try:
                val_df = pd.read_excel(xlsx_file, sheet_name='val')
                test_df = pd.read_excel(xlsx_file, sheet_name='test')

                print(f"  Processing model: {model_name}")
                print(f"    Val rows: {len(val_df)}, Test rows: {len(test_df)}")

                # Identify metric columns (start with Recall@ or MRR@ or NDCG@ etc.)
                metric_start_idx = min([val_df.columns.get_loc(c) for c in val_df.columns if "@" in c])
                param_cols = val_df.columns[:metric_start_idx].tolist()

                required_cols = ['Recall@10', 'MRR@10']  # metrics to optimize on

                if not all(col in val_df.columns for col in required_cols):
                    print(f"    Warning: Missing required columns in {model_name}")
                    continue

                filtered_val_df = val_df.copy()

                # Add a unique key for parameters
                filtered_val_df['param_key'] = filtered_val_df[param_cols].astype(str).agg('|'.join, axis=1)
                test_df['param_key'] = test_df[param_cols].astype(str).agg('|'.join, axis=1)

                # Average performance
                filtered_val_df['avg_performance'] = filtered_val_df[required_cols].mean(axis=1)

                # Pick best row
                best_idx = filtered_val_df['avg_performance'].idxmax()
                best_val_row = filtered_val_df.loc[best_idx].copy()

                best_val_row['dataset_name'] = dataset_name
                best_val_row['model_name'] = model_name

                # Match with test_df via param_key
                key = best_val_row['param_key']
                if key in test_df['param_key'].values:
                    best_test_row = test_df[test_df['param_key'] == key].iloc[0].copy()
                    best_test_row['dataset_name'] = dataset_name
                    best_test_row['model_name'] = model_name
                else:
                    print(f"    Warning: No matching test row found for {model_name}, key={key}")
                    best_test_row = best_val_row.copy()
                    perf_cols = [col for col in best_test_row.index if '@' in str(col)]
                    best_test_row[perf_cols] = np.nan

                best_val_results.append(best_val_row.drop('param_key'))
                best_test_results.append(best_test_row.drop('param_key'))

                print(f"    Best performance: {best_val_row['avg_performance']:.4f}")

            except Exception as e:
                print(f"    Error processing {model_name}: {str(e)}")
                continue

    if not best_val_results:
        print("No valid results found!")
        return

    final_val_df = pd.DataFrame(best_val_results).drop(columns=['avg_performance'], errors='ignore')
    final_test_df = pd.DataFrame(best_test_results).drop(columns=['avg_performance'], errors='ignore')

    cols = final_val_df.columns.tolist()
    if 'dataset_name' in cols and 'model_name' in cols:
        cols.remove('dataset_name')
        cols.remove('model_name')
        cols = ['dataset_name', 'model_name'] + cols
        final_val_df = final_val_df[cols]
        final_test_df = final_test_df[cols]

    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        final_val_df.to_excel(writer, sheet_name='val', index=False)
        final_test_df.to_excel(writer, sheet_name='test', index=False)

    print(f"\n=== SUMMARY ===")
    print(f"Total models processed: {len(final_val_df)}")
    print(f"Output file created: {output_file_path}")
    print(f"Val sheet rows: {len(final_val_df)}")
    print(f"Test sheet rows: {len(final_test_df)}")


# Example usage
if __name__ == "__main__":
    # Set your paths here
    hypertune_folder = "experiments/hyperparameter_tuning/"  # Path to your hypertune folder
    output_file = "experiments/hyperparameter_tuning/Optimal Results.xlsx"  # Output file name

    # Check if hypertune folder exists
    if not os.path.exists(hypertune_folder):
        print(f"Error: Folder '{hypertune_folder}' not found!")
        print("Please make sure you're running this script from the correct directory")
        print("or update the 'hypertune_folder' variable with the correct path.")
    else:
        # Process the results
        process_hypertune_results(hypertune_folder, output_file)
        print(f"\nDone! Check '{output_file}' for the consolidated results.")

    # Load Excel
    df = pd.read_excel(output_file, sheet_name="test")
    df = df[df["model_code"] != "grit_a"]

    # Model name mapping
    model_name_map = {
        "grit": "GrIT",
        "bsarec": "BSARec",
        "lrurec": "LRURec",
        "duorec": "DuoRec",
        "fmlprec": "FMLPRec",
        "linrec": "LinRec"
    }
    
    # Model name mapping
    dataset_name_map = {
        "ml-1m": "MovieLens 1M",
        "ml-100k": "MovieLens 100K",
        "c_a_v": "CDs & Vinyl",
        "v_g": "Video Games",
        "i_a_s": "Industrial & Scientific"
    }

    # Ks to consider
    k_values = [5, 10, 20]
    k_labels = [str(k) for k in k_values]  # string labels
    x_pos = range(len(k_values))  # equally spaced positions

    # Unique models
    models = df["model_code"].unique()

    # Assign unique colors & markers to each model
    colors = plt.cm.Dark2.colors
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "+", "x", "d", "|", "_"]
    style_map = {model: (colors[i % len(colors)], markers[i % len(markers)])
                 for i, model in enumerate(models)}

    # Apply global font style
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10

    save_dir = "experiments/plots"
    os.makedirs(save_dir, exist_ok=True)

    # Loop through each dataset
    for dataset in df['dataset_name'].unique():
        subset = df[df['dataset_name'] == dataset]

        plt.figure(figsize=(7.5, 4.5))  # slightly compact for publications

        # Plot each model’s line (all black, but different markers)
        for _, row in subset.iterrows():
            mrrs = [row[f"MRR@{k}"] for k in k_values]
            color, marker = style_map[row["model_code"]]
            model_label = model_name_map.get(row["model_code"], row["model_code"])
            plt.plot(
                x_pos, mrrs,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.5,
                label=model_label
            )

        plt.xlabel("k", fontsize=10)
        plt.ylabel("MRR@k", fontsize=10)
        plt.xticks(x_pos, k_labels)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Legend on top, multi-column
        ncol = len(models)
        plt.legend(
            fontsize=9,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=ncol
        )

        # Tight layout
        plt.tight_layout()
        # Save figure
        save_path = os.path.join(save_dir, f"{dataset}_mrr_line.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Number of variables (axes)
        num_vars = len(k_values)

        # Angles for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # close loop

        # Setup figure
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        all_mrrs = []
        # Plot each model
        for i, model in enumerate(models):
            row = subset[subset["model_code"] == model]
            if row.empty:
                continue

            mrrs = [row.iloc[0][f"MRR@{k}"] for k in k_values]
            all_mrrs.extend(mrrs)

            mrrs += mrrs[:1]  # close loop

            label = model_name_map.get(model, model)
            ax.plot(angles, mrrs, color=colors[i % len(colors)], linewidth=1.25, label=label)
            ax.fill(angles, mrrs, color=colors[i % len(colors)], alpha=0.1)

        if all_mrrs:
            min_val, max_val = min(all_mrrs), max(all_mrrs)
            margin = 0.05 * (max_val - min_val)  # 5% padding
            ax.set_ylim(min_val - margin, max_val + margin)

        # Add axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(k_labels, fontsize=10)
        ax.tick_params(axis="x", pad=-0)  # smaller pad = closer, default is ~10

        # Y-axis (radial)
        ax.set_rlabel_position(30)
        # Get the auto-generated yticks
        yticks = ax.get_yticks()

        # Keep only alternate labels
        yticklabels = [f"{t:.3f}" if i % 2 == 0 else "" for i, t in enumerate(yticks)]

        ax.set_yticks(yticks)  # keep default ticks
        ax.set_yticklabels(yticklabels, fontsize=8)

        # Style
        ax.tick_params(colors="black", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Legend
        ax.legend(
            bbox_to_anchor=(-0.05, 1.05),
            loc="upper left",
            ncol=1,
            fontsize=9,
            frameon=False
        )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset}_mrr.png"), dpi=300)
        plt.close()

    # Desired dataset plotting order
    ordered_datasets = [
        "ml-100k", "ml-1m", "v_g", "i_a_s", "c_a_v"
    ]

    # Ensure only datasets present in df are included
    datasets = [d for d in ordered_datasets if d in df['dataset_name'].unique()]
    n_datasets = len(datasets)

    # Single row of subplots
    fig, axes = plt.subplots(
        nrows=1, ncols=n_datasets,
        figsize=(4.5 * n_datasets, 5.5),
        subplot_kw=dict(polar=True),
        squeeze=False
    )

    num_vars = len(k_values)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    for idx, dataset in enumerate(datasets):
        subset = df[df['dataset_name'] == dataset]
        ax = axes[0, idx]

        all_mrrs = []

        for i, model in enumerate(models):
            row = subset[subset["model_code"] == model]
            if row.empty:
                continue

            mrrs = [row.iloc[0][f"MRR@{k}"] for k in k_values]
            all_mrrs.extend(mrrs)

            mrrs += mrrs[:1]  # close loop
            label = model_name_map.get(model, model)

            ax.plot(angles, mrrs, color=colors[i % len(colors)], linewidth=1.25, label=label)
            ax.fill(angles, mrrs, color=colors[i % len(colors)], alpha=0.1)

        # scale y-axis with margin
        if all_mrrs:
            min_val, max_val = min(all_mrrs), max(all_mrrs)
            margin = 0.05 * (max_val - min_val)
            ax.set_ylim(min_val - margin, max_val + margin)

        # style axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(k_labels, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", pad=0)

        ax.set_rlabel_position(30)
        yticks = ax.get_yticks()
        yticklabels = [f"{t:.3f}" if i % 2 == 0 else "" for i, t in enumerate(yticks)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=12, fontweight="bold")

        # dataset label below plot
        dataset_label = dataset_name_map.get(dataset, dataset)
        ax.set_title("", fontsize=10, pad=15)  # clear top title
        ax.text(0.5, -0.05, dataset_label, transform=ax.transAxes,
                ha="center", va="center", fontsize=14, fontweight="bold")

    # Shared legend on top
    handles, labels = ax.get_legend_handles_labels()
    bold_font = FontProperties(weight="bold", size=14)

    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(models),
        prop=bold_font,  # <-- bold legend
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "mrr.png"), dpi=300)
    plt.close()