import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from pathlib import Path
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, move_batch_to_device
from utils.config_utils import load_and_merge_config
from models import get_model_class
from utils.file_utils import find_model_path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from pathlib import Path

def normalize(img):
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min) if img_min != img_max else img

def prepare_image(img):
    img = np.squeeze(img)
    if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 1):
        return img
    elif img.ndim == 3 and (img.shape[0] == 3 or img.shape[0] == 1):
        return np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        return img
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def plot_image_rows(cfg, obs_history, future_obs, predictions, dones_np, sample_idx):
    t_obs = min(5, obs_history.shape[0])
    t_pred = min(5, future_obs.shape[0])

    fig, axes = plt.subplots(3, max(t_obs, t_pred), figsize=(8, 6))

    # Plot observation history
    for i in range(t_obs):
        obs = normalize(prepare_image(obs_history[i]))
        axes[0, i].imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
        axes[0, i].set_title(f'Input ($t-{t_obs-1-i}$)')  # Remove fontsize argument
        axes[0, i].axis('off')

    # Plot ground truth and predictions
    for i in range(t_pred):
        gt_np = normalize(prepare_image(future_obs[i]))
        pred_np = normalize(prepare_image(predictions[i]))

        axes[1, i].imshow(gt_np, cmap='viridis' if gt_np.ndim == 2 else None)
        axes[1, i].set_title(f'Ground-truth ($t+{i+1}$)')  # Remove fontsize argument
        axes[1, i].axis('off')

        axes[2, i].imshow(pred_np, cmap='viridis' if pred_np.ndim == 2 else None)
        axes[2, i].set_title(f'Prediction ($t+{i+1}$)')  # Remove fontsize argument
        axes[2, i].axis('off')

        if dones_np[i]:
            add_termination_marker(axes[1, i])
            add_termination_marker(axes[2, i])

    plt.tight_layout()
    save_path = Path(cfg.project_dir) / f'image_rows_plot_{sample_idx}.pdf'
    plt.savefig(save_path, format='pdf', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Image rows plot saved to {save_path}")

def plot_hazard_survival(cfg, hazard_softplus, cumulative_hazard, survival_prob, t_pred, dones_np, sample_idx):
    fig, (ax_hazard, ax_survival) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    time_steps = np.arange(1, t_pred + 1)

    # Hazard and Cumulative Hazard subplot
    ax_hazard.plot(time_steps, hazard_softplus, label='Hazard', color='#0072B2', marker='o', markersize=4)
    ax_hazard.plot(time_steps, cumulative_hazard, label='Cumulative Hazard', color='#D55E00', marker='s', markersize=4)
    ax_hazard.set_ylabel('Hazard')
    ax_hazard.legend(loc='best')

    # Survival Probability subplot
    ax_survival.plot(time_steps, survival_prob, label='Predicted Survival', color='#009E73', marker='^', markersize=4)
    ax_survival.set_ylabel('Survival Probability')
    ax_survival.set_ylim(0, 1)
    ax_survival.legend(loc='best')

    if dones_np.any():
        term_step = np.argmax(dones_np) + 1
        for ax in [ax_hazard, ax_survival]:
            ax.axvline(x=term_step, color='red', linestyle='--', linewidth=2, label='Termination')
            ax.text(term_step, ax.get_ylim()[1], 'Termination', rotation=0, verticalalignment='bottom', fontweight='bold', color='red')

    ax_survival.set_xlabel('Time Step')
    
    # Adjust x-axis limits and ticks
    max_steps_to_show = min(5, t_pred)
    ax_survival.set_xlim(1, max_steps_to_show)
    ax_survival.set_xticks(range(1, max_steps_to_show + 1))
    ax_survival.set_xticklabels([f't+{i}' for i in range(1, max_steps_to_show + 1)])

    plt.tight_layout()
    save_path = Path(cfg.project_dir) / f'hazard_survival_plot_{sample_idx}.pdf'
    plt.savefig(save_path, format='pdf', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Hazard and survival plot saved to {save_path}")

def plot_validation_sample(cfg, model, dataset, device, sample_idx):
    setup_plot_style()

    sample = move_batch_to_device(dataset[sample_idx], device)
    obs_history = sample['observations'][0].cpu().numpy()
    future_obs = sample['next_observations'][0].cpu().numpy()
    dones = sample['dones'][0].cpu().numpy()

    with torch.no_grad():
        model_output = model(sample)

    predictions = model_output['predictions'][0].cpu().numpy()
    hazard = model_output['hazard'][0].cpu().numpy()
    hazard_softplus = F.softplus(torch.tensor(hazard)).numpy()
    dones_np = dones[:hazard_softplus.shape[0]]

    # Ignoring if the sample is not terminated
    if not dones_np.any():
        print(f"Sample {sample_idx} is not terminated. Skipping...")
        return
    
    # Ignoring if it terminates at step 1 or 2
    if dones_np[0] or dones_np[1]:
        print(f"Sample {sample_idx} terminates at step 1 or 2. Skipping...")
        return

    cumulative_hazard = np.cumsum(hazard_softplus, axis=0)
    survival_prob = np.exp(-cumulative_hazard)

    plot_image_rows(cfg, obs_history, future_obs, predictions, dones_np, sample_idx)
    plot_hazard_survival(cfg, hazard_softplus, cumulative_hazard, survival_prob, hazard_softplus.shape[0], dones_np, sample_idx)

FONT_SIZE = 12

def setup_plot_style():
    plt.style.use('default')  # Reset to default style
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#CCCCCC",
    })

def plot_observation_history(axes, obs_history, t_obs):
    for i, ax in enumerate(axes):
        obs = normalize(prepare_image(obs_history[i][:t_obs]))
        ax.imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
        ax.set_title(f'Input ($t-{t_obs-1-i}$)', fontsize=FONT_SIZE+2)
        ax.axis('off')

def plot_ground_truth_and_predictions(gt_axes, pred_axes, future_obs, predictions, dones_np, t_pred):
    for i in range(t_pred):
        gt_np = normalize(prepare_image(future_obs[i]))
        pred_np = normalize(prepare_image(predictions[i]))

        title_gt = f'Ground-truth ($t+{i+1}$)'
        title_pred = f'Prediction ($t+{i+1}$)'

        gt_axes[i].imshow(gt_np, cmap='viridis' if gt_np.ndim == 2 else None)
        gt_axes[i].set_title(title_gt, fontsize=FONT_SIZE+2)
        gt_axes[i].axis('off')

        pred_axes[i].imshow(pred_np, cmap='viridis' if pred_np.ndim == 2 else None)
        pred_axes[i].set_title(title_pred, fontsize=FONT_SIZE+2)
        pred_axes[i].axis('off')

        if dones_np[i]:
            add_termination_marker(gt_axes[i])
            add_termination_marker(pred_axes[i])

def add_termination_marker(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    line_kwargs = dict(color='red', alpha=0.8, linewidth=3, linestyle='-')
    
    # Draw an 'X'
    ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], **line_kwargs)
    ax.plot([xlim[0], xlim[1]], [ylim[1], ylim[0]], **line_kwargs)
    
    # Add "DONE" text
    # ax.text(0.5 * (xlim[0] + xlim[1]), 0.2 * (ylim[0] + ylim[1]), 'DONE', 
    #         horizontalalignment='center', verticalalignment='center',
    #         color='red', fontweight='bold', fontsize=12)
    
def main():
    parser = argparse.ArgumentParser(description="Visualize a validation sample")
    parser.add_argument('--model-type', type=str, required=True, help='Class of trained model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    args = parser.parse_args()

    print("Loading configuration...")
    cfg = load_and_merge_config()
    
    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else cfg.device)
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = EnvironmentDataset(cfg)
    print(f"Dataset loaded with {len(dataset)} samples.")

    print("Getting data dimensions...")
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(dataset)
    print(f"Observation shape: {obs_shape}, Action dim: {action_dim}, Ego state dim: {ego_state_dim}")

    print(f"Initializing model of type {args.model_type}...")
    ModelClass = get_model_class(args.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    model = ModelClass(obs_shape=obs_shape, action_dim=action_dim, ego_state_dim=ego_state_dim, cfg=cfg).to(device)
    print("Model initialized.")

    print(f"Loading model weights from {args.model_path}...")
    model_path = find_model_path(cfg.project_dir, args.model_path)
    if model_path is None:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'], strict=False)
    print("Model weights loaded successfully.")

    for sample_idx in range(20, 120):
        print(f"Visualizing sample {sample_idx}")
        plot_validation_sample(cfg, model, dataset, device, sample_idx)

if __name__ == "__main__":
    main()