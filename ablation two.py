import os
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------
# Mapping of shorthand -> readable
# ---------------------------------
feat_mapping = {
    "none": "No Features",
    "gs": "Global Item Features",
    "gt": "Global Transition Features",
    "ls": "Local Item Features",
    "lt": "Local Transition Features",
    "gls": "Global & Local Item Features",
    "glt": "Global & Local Transition Features",
    "gst": "Global Item & Global Transition Features",
    "lst": "Local Item & Local Transition Features",
    "gslt": "Global Item & Local Transition Features",
    "gtls": "Global Transition & Local Item Features",
    "glsgt": "Global Item, Local Item & Global Transition Features",
    "glslt": "Global Item, Local Item & Local Transition Features",
    "gsglt": "Global Item, Global Transition & Local Transition Features",
    "lsglt": "Local Item, Global Transition & Local Transition Features",
    "all": "All Features"
}

# Dataset readable names
dataset_map = {
    "ml-100k": "MovieLens 100K",
    "ml-1m": "MovieLens 1M",
    "v_g": "Video Games",
    "i_a_s": "Industrial & Scientific",
    "c_a_v": "CDs & Vinyls"
}

# PE readable names
legend_map = {
    "le": "Learnable",
    "se": "Sinusoidal",
    "lse": "Learnable Sinusoidal"
}
legend_order = ["Sinusoidal", "Learnable Sinusoidal", "Learnable"]

dataset_order = ["ml-100k", "ml-1m", "i_a_s", "v_g", "c_a_v" ]
# ---------------------------------
# Progressive feature selection
# ---------------------------------
def get_progressive_features(df, metric):
    df = df.copy()
    df["fuse"] = df["fuse"].str.lower()
    best = {}

    # None (beta=0)
    best["none"] = df[(df["fuse"] == "all") & (df["beta"] == 0)].iloc[0]

    # Best single
    singles = ["gs", "gt", "ls", "lt"]
    best_single = df[df["fuse"].isin(singles)].sort_values(metric, ascending=False).iloc[0]
    best[best_single["fuse"]] = best_single
    single_code = best_single["fuse"]

    single_to_pairs = {
        "gs": ["gls", "gst", "gslt"],
        "gt": ["glt", "gst", "gtls"],
        "ls": ["gls", "lst", "gtls"],
        "lt": ["glt", "lst", "gslt"],
    }

    # Best pair containing single
    candidate_pairs = single_to_pairs[single_code]
    best_pair = df[df["fuse"].isin(candidate_pairs)].sort_values(metric, ascending=False).iloc[0]
    best[best_pair["fuse"]] = best_pair
    pair_code = best_pair["fuse"]

    pair_to_triplets = {
        "gls": ["glsgt", "glslt"],
        "glt": ["gsglt", "lsglt"],
        "gst": ["gsglt", "glsgt"],
        "lst": ["glslt", "lsglt"],
        "gslt": ["gsglt", "glslt"],
        "gtls": ["glsgt", "lsglt"],
    }

    # Best triplet containing pair
    candidate_triplets = pair_to_triplets[pair_code]
    best_triplet = df[df["fuse"].isin(candidate_triplets)].sort_values(metric, ascending=False).iloc[0]
    best[best_triplet["fuse"]] = best_triplet

    # All (beta>0)
    best["all"] = df[(df["fuse"] == "all") & (df["beta"] > 0)].sort_values(metric, ascending=False).iloc[0]
    return best


def best_dict_to_df(best_dict):
    # Explicit progression order
    order = ["none", "gs", "gt", "ls", "lt", "gls", "glt", "gst", "lst",
             "gslt", "gtls", "glsgt", "glslt", "gsglt", "lsglt", "all"]

    rows = []
    for fuse_code in order:
        if fuse_code in best_dict:
            items = best_dict[fuse_code]
            row_dict = dict(items)  # convert tuple pairs → dict
            row_dict["fuse_code"] = fuse_code
            rows.append(row_dict)

    # Convert list of dicts → DataFrame
    df = pd.DataFrame(rows)

    # Reorder so fuse_code is the first column
    cols = ["fuse_code"] + [c for c in df.columns if c != "fuse_code"]
    df = df[cols]

    return df

# ---------------------------------
# Spider plot
# ---------------------------------
def wrap_label(label, width=20):
    """Wrap long labels into multiple lines."""
    return "\n".join(textwrap.wrap(label, width=width))

def nice_step(max_val):
    """Choose a 'nice' radial tick step automatically."""
    if max_val <= 0.05:
        return 0.005
    elif max_val <= 0.1:
        return 0.01
    elif max_val <= 0.2:
        return 0.02
    elif max_val <= 0.5:
        return 0.05
    else:
        return 0.1

def plot_line(dataset, df, metrics, save_dir):
    best_feats = get_progressive_features(df, metrics[0])
    best_df = best_dict_to_df(best_feats)

    labels = list(best_feats.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#FF6F91",  # Golden Yellow
              "#1E90FF",  # Dodger Blue
              "#6A5ACD"]  # Coral/Blush Red

    markers = ["o", "s", "D", "^", "v", "P", "X"]

    # Plot metrics
    for idx, metric in enumerate(metrics):
        values = [best_feats[f][metric] for f in labels]
        ax.plot(range(len(labels)), values,
                marker=markers[idx % len(markers)],
                linewidth=4,  # Thick lines
                markersize=12,  # Larger markers
                markerfacecolor=colors[idx % len(colors)],
                markeredgecolor='white',  # White border around markers
                markeredgewidth=3,  # Thick white border
                label=metric,
                color=colors[idx % len(colors)])

    # X-axis labels (features)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(feat_mapping[l], width=20) for l in labels],
                       rotation=45, ha="right", fontsize=14, fontweight='bold', family="Times New Roman")

    # Y-axis ticks
    min_val = min(min(best_feats[f][m] for f in labels) for m in metrics)
    max_val = max(max(best_feats[f][m] for f in labels) for m in metrics)

    step = nice_step((max_val - min_val) / 5)  # finer step (≈ 5 ticks across the range)

    min_tick = np.floor(min_val / step) * step - step  # one step before min
    max_tick = np.ceil(max_val / step) * step + step  # one step after max

    ticks = np.arange(min_tick, max_tick + step, step)
    ax.set_yticks(ticks)

    tick_labels = [f"{t:.2f}" if i % 2 == 0 else "" for i, t in enumerate(ticks)]
    ax.set_yticklabels(tick_labels, fontsize=14, fontweight='bold', family="Times New Roman")

    ax.set_ylim(min_tick, max_tick)

    ax.set_xlabel('Fusion Strategy', fontsize=14, fontweight='bold', family="Times New Roman")
    ax.set_ylabel('Metric Score', fontsize=14, fontweight='bold', family="Times New Roman")

    # Title + Legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False,
              prop=font_manager.FontProperties(family="Times New Roman", size=14, weight='bold'))
    ax.grid(True, alpha=0.3, linestyle='--')

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset}/{dataset}_line.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return best_df

def plot_line_subplots(datasets, dfs, metrics, save_dir, ncols=2):
    """
    Fixed version with proper axes handling for any number of datasets.
    """
    n_datasets = len(datasets)
    nrows = (n_datasets + ncols - 1) // ncols  # Calculate rows needed

    # Create figure with subplots, leave extra space at top for legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))

    # Handle different subplot configurations - always flatten to 1D
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # This handles both 1D and 2D cases

    colors = ["#FF6F91",  # Golden Yellow
              "#1E90FF",  # Dodger Blue
              "#6A5ACD"]  # Coral/Blush Red
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    legend_handles = []
    legend_labels = []

    for i, dataset in enumerate(datasets):
        ax = axes[i]  # Simple indexing since axes is now always 1D

        df = dfs[dataset] if isinstance(dfs, dict) else dfs[i]
        best_feats = get_progressive_features(df, metrics[0])

        labels = list(best_feats.keys())

        for idx, metric in enumerate(metrics):
            values = [best_feats[f][metric] for f in labels]
            line = ax.plot(range(len(labels)), values,
                           marker=markers[idx % len(markers)],
                           linewidth=4,
                           markersize=12,
                           markerfacecolor=colors[idx % len(colors)],
                           markeredgecolor='white',
                           markeredgewidth=3,
                           label=metric,
                           color=colors[idx % len(colors)])

            # Collect legend handles from first subplot only
            if i == 0:
                legend_handles.extend(line)
                legend_labels.append(metric)

        # Rest of the formatting (same as original)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([wrap_label(feat_mapping[l], width=20) for l in labels],
                            ha="center", fontsize=12, fontweight='bold', family="Times New Roman")

        min_val = min(min(best_feats[f][m] for f in labels) for m in metrics)
        max_val = max(max(best_feats[f][m] for f in labels) for m in metrics)
        step = nice_step((max_val - min_val) / 5)
        min_tick = np.floor(min_val / step) * step - step
        max_tick = np.ceil(max_val / step) * step + step
        ticks = np.arange(min_tick, max_tick + step, step)
        ax.set_yticks(ticks)
        tick_labels = [f"{t:.2f}" if i % 2 == 0 else "" for i, t in enumerate(ticks)]
        ax.set_yticklabels(tick_labels, fontsize=12, fontweight='bold', family="Times New Roman")
        ax.set_ylim(min_tick, max_tick)

        ax.set_xlabel('Fusion Strategy', fontsize=12, fontweight='bold', family="Times New Roman")
        ax.set_ylabel('Metric Score', fontsize=12, fontweight='bold', family="Times New Roman")
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add dataset name below the subplot
        dataset_name = dataset_map.get(dataset, dataset)
        ax.text(0.5, -0.25, dataset_name, transform=ax.transAxes,
                ha='center', va='top', fontsize=14, fontweight='bold',
                family="Times New Roman")

    # Hide empty subplots
    for i in range(n_datasets, nrows * ncols):
        axes[i].axis('off')

    # Add shared legend at the top
    fig.legend(legend_handles, legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=len(metrics), frameon=False,
               prop=font_manager.FontProperties(family="Times New Roman", size=14, weight='bold'))

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for legend
    plt.savefig(os.path.join(save_dir, "combined/featurefuse.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------------
# PE Comparison Barplots
# ---------------------------------
def plot_pe_bars(combined_df, metrics, save_dir):
    sns.set_theme(style="whitegrid", font_scale=1.2)
    palette = [
        "#FFF44F",  # Lemon Yellow
        "#FF6F91",  # Blush Pink
        "#87CEEB",  # Sky Blue
    ]
    for m in metrics:
        subset = combined_df[combined_df["metric"] == m].copy()
        subset["pe"] = subset["pe"].map(legend_map)
        subset["pe"] = pd.Categorical(subset["pe"], categories=legend_order, ordered=True)
        subset["dataset"] = pd.Categorical(subset["dataset"], categories=dataset_order, ordered=True)

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            data=subset, x="dataset", y="value", hue="pe",
            palette=palette, edgecolor="black", linewidth=0.5,
            dodge=True, width=0.5, hue_order=legend_order
        )

        # Rename x-axis
        ax.set_xticklabels([dataset_map.get(lbl.get_text(), lbl.get_text())
                            for lbl in ax.get_xticklabels()],
                           rotation=20, ha="center", fontsize=14, family="Times New Roman", fontweight="bold")

        # Y-axis scaling
        max_val = subset["value"].max()
        y_max = np.ceil((max_val + 0.02) / 0.02) * 0.02
        ax.set_yticks(np.arange(0, y_max + 0.001, 0.02))
        ax.set_ylim(0, y_max)

        plt.yticks(fontsize=12, family="Times New Roman", fontweight="bold")
        plt.ylabel(m, fontsize=14, family="Times New Roman", fontweight="bold")
        plt.xlabel("Dataset", fontsize=14, family="Times New Roman", fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Legend above
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False,
            prop=font_manager.FontProperties(family="Times New Roman", size=14, weight="bold"),
        )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"combined/{m}_PE.png"), dpi=300, bbox_inches="tight")
        plt.close()


        # --- Combined Subplots ---
        n_metrics = len(metrics)
        ncols = int(n_metrics)
        nrows = 1

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(8 * ncols, 6 * nrows),
            squeeze=False
        )

        for idx, m in enumerate(metrics):
            subset = combined_df[combined_df["metric"] == m].copy()
            subset["pe"] = subset["pe"].map(legend_map)
            subset["pe"] = pd.Categorical(subset["pe"], categories=legend_order, ordered=True)
            subset["dataset"] = pd.Categorical(subset["dataset"], categories=dataset_order, ordered=True)

            ax = axes[idx // ncols, idx % ncols]
            sns.barplot(
                data=subset, x="dataset", y="value", hue="pe",
                palette=palette, edgecolor="black", linewidth=0.5,
                dodge=True, width=0.5, hue_order=legend_order, ax=ax
            )

            # Remove subplot legends
            if ax.get_legend() is not None:
                ax.legend_.remove()

            # Rename x-axis
            ax.set_xticklabels(
                [dataset_map.get(lbl.get_text(), lbl.get_text())
                 for lbl in ax.get_xticklabels()],
                rotation=20, ha="center",
                fontsize=14, fontweight="bold", family="Times New Roman"
            )

            # Y-axis scaling
            max_val = subset["value"].max()
            y_max = np.ceil((max_val + 0.02) / 0.02) * 0.02
            ax.set_yticks(np.arange(0, y_max + 0.001, 0.02))
            ax.set_ylim(0, y_max)
            ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()],
                               fontsize=14, family="Times New Roman")

            # Labels & ticks (bold, size 14)
            ax.set_ylabel(m, fontsize=14, fontweight="bold", family="Times New Roman")
            ax.set_xlabel("Dataset", fontsize=14, fontweight="bold", family="Times New Roman")
            ax.tick_params(axis="y", labelsize=14, labelrotation=0)
            ax.tick_params(axis="x", labelsize=14)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Remove empty subplots
        for j in range(idx + 1, nrows * ncols):
            fig.delaxes(axes[j // ncols, j % ncols])

        # Shared legend (bold, size 14)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
            prop=font_manager.FontProperties(family="Times New Roman", size=14, weight="bold"),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend
        plt.savefig(os.path.join(save_dir, "combined/hyper_pe.png"), dpi=300, bbox_inches="tight")
        plt.close()

# ---------------------------------
# Main Execution
# ---------------------------------
def main():
    base_dir = "experiments/hyperparameter_tuning"
    save_dir = "experiments/ablation"
    metrics = ["Recall@5", "NDCG@5", "MRR@5"]

    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    combined_results = []

    for dataset in datasets:
        if dataset in ["archive"]:
            continue

        file_path = os.path.join(base_dir, dataset, "grit_a_hypertune_results.xlsx")
        if not os.path.exists(file_path):
            continue

        df = pd.read_excel(file_path, sheet_name="test")
        print(f"Plotting line for {dataset} ...")
        best_df = plot_line(dataset, df, metrics, save_dir)
        best_df.to_csv(os.path.join(save_dir, dataset, "grit_bestfeats.csv"))

        # Collect PE results (all + beta>0)
        subset = df[(df["fuse"] == "all") & (df["beta"] != 0)]
        if not subset.empty:
            for m in metrics:
                for pe, group in subset.groupby("pe"):
                    combined_results.append([dataset, pe, m, group[m].mean()])

    dfs = {}
    subplot_data = ['ml-1m', 'v_g']
    for dataset in subplot_data:
        file_path = os.path.join(base_dir, dataset, "grit_a_hypertune_results.xlsx")
        if not os.path.exists(file_path):
            continue
        df = pd.read_excel(file_path, sheet_name="test")
        dfs[dataset] = df

    plot_line_subplots(subplot_data, dfs, metrics, save_dir, ncols=2)

    if combined_results:
        combined_df = pd.DataFrame(combined_results, columns=["dataset", "pe", "metric", "value"])
        plot_pe_bars(combined_df, metrics, save_dir)


if __name__ == "__main__":
    main()
