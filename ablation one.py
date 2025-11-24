import os
import yaml
import torch
import random
import numpy as np
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from datasets.ml_1m import ML1MDataset
from datasets.ml_100k import ML100KDataset
from datasets.amazon import AmazonIndustrialDataset, AmazonVideoGamesDataset, AmazonCDDataset
from matplotlib.colors import LinearSegmentedColormap

# Import from your existing modules
DATASETS = {
    'ml-1m': ML1MDataset,
    'ml-100k': ML100KDataset,
    'cds_and_vinyl': AmazonCDDataset,
    'industrial_and_scientific': AmazonIndustrialDataset,
    'video_games': AmazonVideoGamesDataset,
}

# Dataset readable names
dataset_map = {
    "ml-100k": "ml-100k",
    "ml-1m": "ml-1m",
    "video_games": "v_g",
    "industrial_and_scientific": "i_a_s",
    "cds_and_vinyl": "c_a_v"
}

def dataloader_factory(args):
    """Get dataloader based on dataset code"""
    dataset = DATASETS[args['dataset_code']](args)
    from dataloader.dataloader import SequentialDataloader
    dataloader = SequentialDataloader(args, dataset)
    return dataloader.get_pytorch_dataloaders()


def load_model_and_config(model_code, dataset_code):
    """Load model configuration and initialize model"""
    # Load base config
    base_yaml = "config/base.yaml"
    with open(base_yaml, 'r') as f:
        base_config = yaml.safe_load(f)

    base_config['dataset_code'] = dataset_code

    # Load model-specific hyperparameter config
    model_yaml = f"config/optimal/{model_code}.yaml"
    with open(model_yaml, 'r') as f:
        param_grid = yaml.safe_load(f)

    # Use the first (presumably optimal) configuration from hyperparameter tuning
    optimal_params = {key: values[0] for key, values in param_grid.items()}
    config = deepcopy(base_config)
    config.update(optimal_params)

    _, _, _ = dataloader_factory(config)

    # Initialize model
    if config['model_code'] == 'grit_a':
        from model.gritrec_a import GRITRecAblation
        model = GRITRecAblation(config)
    else:
        raise ValueError(f"This script is designed for GRIT Ablation model, got: {config['model_code']}")

    return model, config


def find_model_checkpoint(config, param_combination, model_code):
    """Find the trained model checkpoint using hyperparameter tuning folder structure"""
    # Construct dataset initials
    if "-" in config['dataset_code']:
        dataset_initials = config['dataset_code']
    else:
        dataset_initials = "_".join([part[0] for part in config['dataset_code'].split("_")])

    # Read the YAML file again to get the original parameter order
    model_yaml = f"config/optimal/{model_code}.yaml"
    with open(model_yaml, 'r') as f:
        param_grid = yaml.safe_load(f)

    # Create parameter combination string in the same order as YAML file
    parts = []
    for key in param_grid.keys():  # This preserves the YAML order
        if key in param_combination:
            value = param_combination[key]
            parts.append(f"{key.split('_')[-1][0]}_{value}")

    param_string = "_".join(parts)

    # Construct the exact path used in hyperparameter tuning
    model_path = f"experiments/hyperparameter_tuning/{dataset_initials}/{config['model_code']}_tuned/{param_string}"
    models_path = os.path.join(model_path, "models")

    print(f"Looking for checkpoint in: {models_path}")

    if os.path.exists(models_path):
        # Look for .pth files
        pth_files = [f for f in os.listdir(models_path) if f.endswith('.pth')]
        if pth_files:
            # Return the best model (usually contains 'best' in filename)
            best_files = [f for f in pth_files if 'best' in f.lower()]
            if best_files:
                checkpoint_path = os.path.join(models_path, best_files[0])
                print(f"Found best checkpoint: {checkpoint_path}")
                return checkpoint_path
            else:
                checkpoint_path = os.path.join(models_path, pth_files[0])
                print(f"Found checkpoint: {checkpoint_path}")
                return checkpoint_path

    print("\nAvailable experiment directories:")
    print(f"  Primary: {models_path}: {'EXISTS' if os.path.exists(models_path) else 'NOT FOUND'}")

    raise FileNotFoundError("Could not find model checkpoint. Please check the paths above.")


def load_trained_model(model_code, dataset_code):
    """Load trained model with weights using hyperparameter tuning structure"""
    model, config = load_model_and_config(model_code, dataset_code)

    # Get the parameter combination that was used (first/optimal from yaml)
    model_yaml = f"config/optimal/{model_code}.yaml"
    with open(model_yaml, 'r') as f:
        param_grid = yaml.safe_load(f)

    # Extract the optimal parameters (first values from each parameter)
    optimal_params = {key: values[0] for key, values in param_grid.items()}
    config.update(optimal_params)

    checkpoint_path = find_model_checkpoint(config, optimal_params, model_code)

    print(f"Loading model from: {checkpoint_path}")
    print(f"Using parameters: {optimal_params}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully!")
    print(f"Final config: {config}")

    return model, config


def analyze_group_weights(model, config):
    """Ablation 1: Analyze group similarity via W W^T"""
    print("\n" + "=" * 50)
    print("ABLATION 1: GROUP WEIGHTS ANALYSIS")
    print("=" * 50)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.weight"] = "bold"

    # Create ablation folder
    ablation_dir = f"experiments/ablation/{dataset_map[config['dataset_code']]}/"
    os.makedirs(ablation_dir, exist_ok=True)

    # Extract comb_lin linear layer
    stats_layer = model.item_encoder.blocks[0].layer.grit_stats_layer
    comb_lin = stats_layer.comb_lin

    linear_layer = None
    for module in comb_lin:
        if isinstance(module, torch.nn.Linear):
            linear_layer = module
            break

    if linear_layer is None:
        print("Could not find linear layer in comb_lin module")
        return None, None, None

    # Get weights
    weights = linear_layer.weight.data  # [num_groups, 2*num_groups]

    # Compute W W^T (group similarity)
    weights_norm = torch.nn.functional.normalize(weights, p=2, dim=1)  # L2 normalize each row
    weight_similarity = torch.mm(weights_norm, weights_norm.t())  # [num_groups, num_groups]

    # Save matrices
    weights_file = os.path.join(ablation_dir, f"{dataset_map[config['dataset_code']]}_group_weights.npy")
    similarity_file = os.path.join(ablation_dir, f"{dataset_map[config['dataset_code']]}_similarity_weights.npy")
    np.save(weights_file, weights.numpy())
    np.save(similarity_file, weight_similarity.numpy())

    print(f"Saved weights → {weights_file}")
    print(f"Saved W W^T similarity → {similarity_file}")

    # Heatmap: full similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(weight_similarity.numpy(), cmap="coolwarm",
                xticklabels=False, yticklabels=False, cbar=True,
                linewidths=0.1, linecolor="white")
    plt.xlabel("Groups", fontsize=14, fontname="Times New Roman", fontweight="bold")
    plt.ylabel("Groups", fontsize=14, fontname="Times New Roman", fontweight="bold")
    heatmap_file = os.path.join(ablation_dir, f"{dataset_map[config['dataset_code']]}_group_sim.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved full similarity heatmap → {heatmap_file}")

    # Identify random pairs
    num_groups = weight_similarity.size(0)
    rand_idx = torch.randperm(num_groups)[:50]
    submatrix = weight_similarity[rand_idx][:, rand_idx]

    # Heatmap: top-10 distinct groups
    plt.figure(figsize=(8, 6))
    sns.heatmap(submatrix.numpy(), cmap="coolwarm",
                xticklabels=False, yticklabels=False,
                cbar=True, linewidths=0.1, linecolor="white")
    plt.xlabel("Groups", fontsize=14, fontname="Times New Roman", fontweight="bold")
    plt.ylabel("Groups", fontsize=14, fontname="Times New Roman", fontweight="bold")
    distinct_file = os.path.join(ablation_dir, f"{dataset_map[config['dataset_code']]}_subset_sim.png")
    plt.savefig(distinct_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dissimilarity heatmap → {distinct_file}")

    return None


def generate_distinct_colors(groups, colormaps, seed=30):
    random.seed(seed)
    np.random.seed(seed)

    n_groups = len(groups)
    n_cmaps = len(colormaps)

    # oversample within each colormap
    oversample = max(n_groups, 20)   # ensure enough diversity
    sampled_colors = []

    for cmap_name in colormaps:
        cmap = plt.cm.get_cmap(cmap_name, oversample)
        # pick evenly spaced colors from the whole range
        sampled_colors.extend([cmap(i / oversample) for i in range(oversample)])

    # Deduplicate by rounding RGBA values
    sampled_colors = list({tuple(np.round(c, 3)) for c in sampled_colors})

    # --- Interleave colors to maximize separation ---
    step = max(1, len(sampled_colors) // n_groups)
    color_pool = [sampled_colors[(i * step) % len(sampled_colors)] for i in range(n_groups)]

    # Shuffle groups but keep colors well separated
    random.shuffle(groups)
    return {g: c for g, c in zip(groups, color_pool)}


def plot_cell_state_heatmap(cell_states, ablation_dir, config):
    """
    Plot heatmap of all groups (y-axis) over timesteps (x-axis) in grayscale
    with higher values shown in black. Dynamically sets vmin and vmax from data.
    """
    data = cell_states.numpy().T   # [G, L]
    # Create custom colormap: white → yellow → red → violet
    custom_cmap = LinearSegmentedColormap.from_list(
        "white_yellow_red_violet",
        ["white", "brown", "black"]
    )
    vmin = data.min()
    vmax = data.max()

    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(
        data,
        cmap=custom_cmap,    # high=black, low=white
        cbar=False,
        vmin=vmin,
        vmax=vmax
    )

    plt.xlabel("Timestep", fontsize=16, fontname="Times New Roman", fontweight="bold")
    plt.ylabel("Group ID", fontsize=16, fontname="Times New Roman", fontweight="bold")

    # Custom xticks: show ticks at all positions but label only 1,5,10,...
    timesteps = np.arange(1, data.shape[1] + 1)  # 1..L
    labels = [str(t) if (t == 1 or t % 5 == 0) and t <= 50 else "" for t in timesteps]

    plt.xticks(
        ticks=np.arange(data.shape[1]) + 0.5,  # center of heatmap cells
        labels=labels,
        rotation=0,
        fontsize=12, fontname="Times New Roman"
    )

    plt.yticks([])  # remove y-axis ticks

    # --- Add a box (all spines visible) ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    outpath = os.path.join(ablation_dir, f"{dataset_map[config['dataset_code']]}_group_timeline_map.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved heatmap plot at {outpath}")


def analyze_user_cell_states(model, config, topk=20):
    """Extract cell states for a user and visualize top-k group membership at steps {1} ∪ {5,10,15,...}"""
    print("\n" + "=" * 50)
    print("ABLATION 2: USER CELL STATES TIMELINE")
    print("=" * 50)

    # Ablation folder
    ablation_dir = f"experiments/ablation/{dataset_map[config['dataset_code']]}/"
    os.makedirs(ablation_dir, exist_ok=True)

    # Dataset (not dataloader)
    _, _, tst = dataloader_factory(config)
    dataset = tst.dataset


    # --------------------------
    # Find users with full-length sequences (no padding)
    # --------------------------
    lengths = []
    for i in range(len(dataset)):
        seq, _, _, _ = dataset[i]
        non_padded_len = (seq != 0).sum().item()
        lengths.append(non_padded_len)

    max_len = max(lengths)
    full_len_indices = [i for i, l in enumerate(lengths) if l == max_len]

    if not full_len_indices:
        print("No full-length sequences found")
        return

    # Pick one random user with max length
    user_index = random.choice(full_len_indices)
    seq, answer, times, user = tst.dataset[user_index]   # one user
    input_ids = seq.unsqueeze(0)  # [1, L]

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)

    # Cell states last layer: [1, L, G] -> [L, G]
    cell_states = outputs['cell_states'][-1][0].cpu()
    cell_states_file = os.path.join(ablation_dir, f"{dataset_map[config['dataset_code']]}_cell_states.npy")
    np.save(cell_states_file, cell_states.numpy())

    plot_cell_state_heatmap(cell_states, ablation_dir, config)

    non_padded_positions = (seq != 0).nonzero(as_tuple=True)[0]
    if len(non_padded_positions) == 0:
        print("No valid (non-padded) items found in sequence")
        return

    actual_cell_states = cell_states[non_padded_positions]  # [actual_L, G]

    # --------------------------
    # Convert cell states -> active group IDs
    # --------------------------
    top_groups_each_step = torch.topk(actual_cell_states, k=topk, dim=1).indices.tolist()
    time_points = list(range(1, len(top_groups_each_step) + 1))
    max_t = len(time_points)

    # Collect unique groups
    all_groups = sorted(list(set(sum(top_groups_each_step, []))))
    group_id_to_y = {g: i for i, g in enumerate(all_groups)}

    colormaps = ["tab10", "Set1", "inferno", "Dark2", "Accent", "viridis", "plasma", "inferno", "magma", "tab20"]
    group_id_to_color = generate_distinct_colors(all_groups, colormaps)

    # --------------------------
    # Plot timeline (points at 1 and multiples of 5)
    # --------------------------
    plt.figure(figsize=(14, 8))

    selected_steps = {1}
    selected_steps.update(range(5, max_t + 1, 5))  # 5,10,15,...

    last_selected_idx = None
    for t_idx, groups_at_t in enumerate(top_groups_each_step):
        current_time = time_points[t_idx]
        if current_time not in selected_steps:
            continue

        # Scatter points
        for group_id in groups_at_t:
            y_pos = group_id_to_y[group_id]
            plt.scatter(current_time, y_pos,
                        color=group_id_to_color[group_id], marker="o", # square markers
                        s=40, zorder=5)

            # Draw persistence line if also present at previous selected step
            if last_selected_idx is not None:
                prev_time = time_points[last_selected_idx]
                prev_groups = top_groups_each_step[last_selected_idx]
                if group_id in prev_groups:
                    plt.plot([prev_time, current_time], [y_pos, y_pos],
                             color=group_id_to_color[group_id],
                             linestyle='-', linewidth=2.2, zorder=1)

        last_selected_idx = t_idx

    # --------------------------
    # Aesthetics
    # --------------------------
    plt.xlabel('Timestep', fontsize=16, fontname="Times New Roman", fontweight="bold")
    plt.ylabel('Group ID', fontsize=16, fontname="Times New Roman", fontweight="bold")
    plt.yticks([])  # remove y-axis ticks
    plt.xticks(fontsize=16, fontname="Times New Roman", fontweight="bold")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Keep axis from 0 to max_t, with little margin
    plt.xlim(0, max_t + 1)

    # Ticks at exactly: 1,5,10,...,50 (clamped to max_t)
    desired_ticks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    visible_ticks = [t for t in desired_ticks if t <= max_t]
    plt.xticks(visible_ticks, fontsize=10, fontname="Times New Roman")

    outpath = os.path.join(ablation_dir, f"{dataset_map[config['dataset_code']]}_group_timeline.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"Saved timeline plot at {outpath}")

def main():
    """Main function to perform both ablations"""
    print("GRIT Model Analysis Script")
    print("=" * 50)

    # Get user input for model and dataset
    print('\n******************** Dataset Selection ********************')
    dataset_code = {'1': 'ml-1m', '2': 'ml-100k', '3': 'video_games',
                    '4': 'cds_and_vinyl', '5': 'industrial_and_scientific'}
    print("Available datasets:", dataset_code)
    data_code = input("Enter the dataset code: ").strip()
    selected_dataset = dataset_code[data_code]

    print('\n******************** Model Selection ********************')
    available_models = ['grit', 'grit_a']  # Only GRIT models supported
    print("Available GRIT models:", ", ".join(available_models))
    model_code = input("Enter the model code: ").strip()

    if model_code not in available_models:
        raise ValueError(f"Model {model_code} not supported. Use: {available_models}")

    # Load trained model
    print(f"\nLoading trained {model_code} model for {selected_dataset} dataset...")

    model, config = load_trained_model(model_code, selected_dataset)


    # Perform ablations
    # Ablation 1: Group weights analysis
    analyze_group_weights(model, config)

    # Ablation 2: User cell states analysis
    analyze_user_cell_states(model, config, topk=20)

    print(f"\n{'=' * 50}")
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Generated files:")
    print("  - group_weights_analysis.png: Visualization of group combination weights")
    print("  - user_cell_states_analysis.png: User cell states evolution analysis")

if __name__ == "__main__":
    main()