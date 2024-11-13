import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from pathlib import Path
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, move_batch_to_device
from utils.config_utils import load_and_merge_config
from utils.training_utils import load_model_state
from models import get_model_class
from utils.file_utils import find_model_path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from pathlib import Path

from plotting_setup import setup_plotting, FONT_SIZE
setup_plotting()

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

def get_global_latent_range(input_latents, predicted_latents=None):
    """Calculate global min/max values across all latent representations."""
    all_latents = []
    if input_latents is not None:
        all_latents.extend([lat.flatten() for lat in input_latents])
    if predicted_latents is not None:
        all_latents.extend([lat.flatten() for lat in predicted_latents])
    
    all_values = np.concatenate(all_latents)
    return np.min(all_values), np.max(all_values)

def plot_latent_grid(latent_vector, ax, title, vmin=None, vmax=None):
    """
    Plot a latent vector as a grid with consistent color mapping.
    
    Args:
        latent_vector: The latent vector to visualize
        ax: Matplotlib axis to plot on
        title: Title for the plot
        vmin: Minimum value for color mapping
        vmax: Maximum value for color mapping
    """
    # Reshape the latent vector into a square grid
    dim = int(np.sqrt(latent_vector.shape[0]))
    grid = latent_vector.reshape(dim, dim)
    
    # Plot the grid with consistent color mapping
    im = ax.imshow(grid, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    return im

def plot_image_rows(cfg, obs_history, future_obs, predictions, dones_np, sample_idx, show: bool = False, latent_info=None):
    t_obs = min(5, obs_history.shape[0])
    t_pred = min(5, future_obs.shape[0])

    if latent_info is not None:
        input_latents, predicted_latents = latent_info
        # Calculate global min/max for consistent color mapping
        vmin, vmax = get_global_latent_range(input_latents, predicted_latents)
        num_rows = 5  # Observations + latents + GT + predictions + predicted latents
        row_order = ['input_obs', 'input_latents', 'pred_latents', 'predictions', 'gt']
    else:
        num_rows = 3  # Just observations + GT + predictions
        row_order = ['input_obs', 'predictions', 'gt']
        vmin, vmax = None, None

    fig, axes = plt.subplots(num_rows, max(t_obs, t_pred), figsize=(8, 2 * num_rows))

    row_indices = {name: idx for idx, name in enumerate(row_order)}

    # Plot observation history
    for i in range(t_obs):
        obs = normalize(prepare_image(obs_history[i]))
        axes[row_indices['input_obs'], i].imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
        axes[row_indices['input_obs'], i].set_title(f'Input ($t-{t_obs-1-i}$)')
        axes[row_indices['input_obs'], i].axis('off')

        # Plot input latents if available
        if latent_info is not None:
            plot_latent_grid(input_latents[i], axes[row_indices['input_latents'], i], 
                           f'Latent $t-{t_obs-1-i}$', vmin=vmin, vmax=vmax)

    # Clear unused input columns
    for i in range(t_obs, max(t_obs, t_pred)):
        axes[row_indices['input_obs'], i].axis('off')
        if latent_info is not None:
            axes[row_indices['input_latents'], i].axis('off')

    # Plot ground truth and predictions
    for i in range(t_pred):
        gt_np = normalize(prepare_image(future_obs[i]))
        pred_np = normalize(prepare_image(predictions[i]))

        axes[row_indices['gt'], i].imshow(gt_np, cmap='viridis' if gt_np.ndim == 2 else None)
        axes[row_indices['gt'], i].set_title(f'GT ($t+{i+1}$)')
        axes[row_indices['gt'], i].axis('off')

        axes[row_indices['predictions'], i].imshow(pred_np, cmap='viridis' if pred_np.ndim == 2 else None)
        axes[row_indices['predictions'], i].set_title(f'Pred. ($t+{i+1}$)')
        axes[row_indices['predictions'], i].axis('off')

        if dones_np[i]:
            add_termination_marker(axes[row_indices['gt'], i])
            add_termination_marker(axes[row_indices['predictions'], i])
            if latent_info is not None:
                add_termination_marker(axes[row_indices['pred_latents'], i])

        # Plot predicted latents if available
        if latent_info is not None and predicted_latents is not None:
            plot_latent_grid(predicted_latents[i], axes[row_indices['pred_latents'], i], 
                           f'Latent $t+{i+1}$', vmin=vmin, vmax=vmax)

    # Clear unused prediction columns
    for i in range(t_pred, max(t_obs, t_pred)):
        axes[row_indices['gt'], i].axis('off')
        axes[row_indices['predictions'], i].axis('off')
        if latent_info is not None:
            axes[row_indices['pred_latents'], i].axis('off')

    plt.tight_layout()
    save_path = Path(cfg.project_dir) / f'image_rows_plot_{sample_idx}.pdf'
    plt.savefig(save_path, format='pdf', dpi=100, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Image rows plot saved to {save_path}")

def plot_hazard_survival(cfg, hazard_softplus, cumulative_hazard, survival_prob, t_pred, dones_np, sample_idx, show: bool = False):
    fig, ax_hazard = plt.subplots(figsize=(8, 2))
    
    # Create a second y-axis sharing the same x-axis
    ax_survival = ax_hazard.twinx()
    
    time_steps = np.arange(1, t_pred + 1)

    # Plot log hazard on left y-axis
    line1 = ax_hazard.plot(time_steps, np.log(hazard_softplus + 1e-7), 
                          label='Log. Hazard', color='#0072B2', 
                          marker='o', markersize=4)
    ax_hazard.set_ylabel('Log. Hazard', color='#0072B2')
    ax_hazard.tick_params(axis='y', labelcolor='#0072B2')

    # Plot survival probability on right y-axis
    line2 = ax_survival.plot(time_steps, survival_prob, 
                            label='Survival Probability', color='#009E73',
                            marker='^', markersize=4)
    ax_survival.set_ylabel('Survival Prob.', color='#009E73')
    ax_survival.tick_params(axis='y', labelcolor='#009E73')
    ax_survival.set_ylim(0, 1)

    # Add termination marker if applicable
    # if dones_np.any():
    #     term_step = np.argmax(dones_np) + 1
    #     ax_hazard.axvline(x=term_step, color='red', linestyle='--', 
    #                      linewidth=2, label='Termination')
    #     ax_hazard.text(term_step, ax_hazard.get_ylim()[1], 'Termination',
    #                   rotation=0, verticalalignment='bottom',
    #                   fontweight='bold', color='red')

    # Adjust x-axis limits and ticks
    max_steps_to_show = min(5, t_pred)
    ax_hazard.set_xlim(1, max_steps_to_show)
    ax_hazard.set_xticks(range(1, max_steps_to_show + 1))
    ax_hazard.set_xticklabels([f't+{i}' for i in range(1, max_steps_to_show + 1)])
    ax_hazard.set_xlabel('Time Step')

    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_hazard.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    save_path = Path(cfg.project_dir) / f'hazard_survival_plot_{sample_idx}.pdf'
    plt.savefig(save_path, format='pdf', dpi=100, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Hazard and survival plot saved to {save_path}")


def plot_validation_sample(cfg, model, sample, sample_idx, show: bool = False, plot_latent: bool = False):
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

    # Get latent representations if requested
    latent_info = None
    if plot_latent and hasattr(model, 'encode'):
        # Get input latents for each observation in the sequence
        input_latents = []
        for i in range(obs_history.shape[0]):
            obs_slice = sample['observations'][:, i:i+1]
            ego_state_slice = sample['ego_states'][:, i:i+1]
            input_batch = {
                'observations': obs_slice,
                'ego_states': ego_state_slice
            }
            latent = model.encode(input_batch)
            if isinstance(latent, tuple):
                latent = latent[0]
            input_latents.append(latent.detach().cpu().numpy()[0])
        
        predicted_latents = None
        if 'predicted_latents' in model_output:
            predicted_latents = model_output['predicted_latents'][0].detach().cpu().numpy()
        
        latent_info = (input_latents, predicted_latents)

    plot_image_rows(cfg, obs_history, future_obs, predictions, dones_np, sample_idx, show=show, latent_info=latent_info)
    plot_hazard_survival(cfg, hazard_softplus, cumulative_hazard, survival_prob, hazard_softplus.shape[0], dones_np, sample_idx, show=show)

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
    parser.add_argument('--show', action='store_true', help='Show the plots')
    parser.add_argument('--plot-latent', action='store_true', help='Plot latent representations')
    args = parser.parse_args()

    print("Loading configuration...")
    cfg = load_and_merge_config()
    
    print("Setting up device...")
    #device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else cfg.device)
    device = "cpu"
    cfg.device = device
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
    
    load_model_state(
        model_path=model_path,
        model=model,
        device=device,
        strict=False
    )
    # model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'], strict=False)
    print("Model weights loaded successfully.")

    batch = dataset[0]

    for sample_idx in range(0, 120):
        print(f"Visualizing sample {sample_idx}")

        minibatch = {k: v[sample_idx:sample_idx+1] for k, v in batch.items()}
        minibatch = move_batch_to_device(minibatch, device)
        plot_validation_sample(cfg, model, minibatch, sample_idx, args.show, args.plot_latent)

if __name__ == "__main__":
    main()