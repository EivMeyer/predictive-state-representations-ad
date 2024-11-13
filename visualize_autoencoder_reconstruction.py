import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, move_batch_to_device
from utils.config_utils import load_and_merge_config
from utils.training_utils import load_model_state
from models import get_model_class
from utils.file_utils import find_model_path
import argparse
from typing import Tuple, List
from matplotlib.gridspec import GridSpec
from plotting_setup import setup_plotting

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from typing import Tuple, List

# IEEE paper formatting settings
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "text.usetex": True,
#     "pgf.rcfonts": False,
#     "font.size": 9,                # General font size
#     "axes.titlesize": 9,           # Title font size
#     "axes.labelsize": 9,           # Axis label font size
#     "xtick.labelsize": 8,          # X-tick label font size
#     "ytick.labelsize": 8,          # Y-tick label font size
#     "legend.fontsize": 8,          # Legend font size
#     "lines.linewidth": 1.0,
#     "lines.markersize": 4,
#     "legend.frameon": False,
#     "axes.grid": True,
#     "grid.linewidth": 0.5
# })

setup_plotting(font_size=5)

def normalize(img: np.ndarray) -> np.ndarray:
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min) if img_min != img_max else img

def prepare_image(img: np.ndarray) -> np.ndarray:
    img = np.squeeze(img)
    if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 1):
        return img
    elif img.ndim == 3 and (img.shape[0] == 3 or img.shape[0] == 1):
        return np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        return img
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def get_autoencoder_output(model: torch.nn.Module, sample: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a sample through the autoencoder and return numpy arrays."""
    with torch.no_grad():
        model_output = model(sample)
        latent_mu, latent_logvar = model.encode(sample)
    
    # Get the original observation (last frame)
    observation = sample['observations'][0, -1].detach().cpu().numpy()
    observation = normalize(prepare_image(observation))
    
    # Get the latent representation
    latent_rep = latent_mu[0].detach().cpu().numpy()
    latent_dim = int(np.sqrt(latent_rep.shape[0]))
    latent_map = latent_rep.reshape(latent_dim, latent_dim)
    
    # Get the reconstruction
    reconstruction = model_output['predictions'][0, 0].detach().cpu().numpy()
    reconstruction = normalize(prepare_image(reconstruction))
    
    return observation, latent_map, reconstruction

def plot_multirow_samples(model: torch.nn.Module, samples: List[dict], 
                         output_dir: Path, n_rows: int, n_cols: int,
                         show: bool = False, fig_label: str = "") -> None:
    """Create a publication-ready multi-row figure with consistent colormap scaling."""
    
    # Calculate dimensions for IEEE single-column format
    fig_width = 3.5  # inches (IEEE single-column width)
    cell_aspect = 0.75  # height/width ratio for each cell
    cell_width = fig_width / 3  # divide by 3 for original, latent, reconstruction
    cell_height = cell_width * cell_aspect
    fig_height = cell_height * n_rows
    
    # First pass: get global min/max for latent space normalization
    all_latent_maps = []
    all_obs = []
    all_recons = []
    for sample in samples:
        observation, latent_map, reconstruction = get_autoencoder_output(model, sample)
        all_latent_maps.append(latent_map)
        all_obs.append(observation)
        all_recons.append(reconstruction)
    
    # Calculate global min/max for latent space
    global_min = min(map.min() for map in all_latent_maps)
    global_max = max(map.max() for map in all_latent_maps)

    # If one is negative and the other is positive, adjust them to have the same magnitude
    if global_min < 0 and global_max > 0:
        max_abs_value = max(abs(global_min), abs(global_max))
        global_min = -max_abs_value
        global_max = max_abs_value

    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Process each row
    for row in range(min(n_rows, len(samples))):
        observation = all_obs[row]
        latent_map = all_latent_maps[row]
        reconstruction = all_recons[row]
        
        # Original
        ax_orig = fig.add_subplot(gs[row, 0])
        ax_orig.imshow(observation)
        ax_orig.axis('off')
        if row == 0:
            ax_orig.set_title('Original')
        
        # Latent
        ax_latent = fig.add_subplot(gs[row, 1])
        im = ax_latent.imshow(latent_map, cmap='viridis', vmin=global_min, vmax=global_max)
        ax_latent.set_xticks([])
        ax_latent.set_yticks([])
        if row == 0:
            ax_latent.set_title('Latent')
        
        # Add colorbar with better spaced and formatted ticks
        cbar = plt.colorbar(im, ax=ax_latent)
        tick_locator = plt.LinearLocator(numticks=3)
        cbar.locator = tick_locator
        cbar.formatter = ticker.FormatStrFormatter('%.1f')
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=4.5)
        
        # Reconstruction
        ax_recon = fig.add_subplot(gs[row, 2])
        ax_recon.imshow(reconstruction)
        ax_recon.axis('off')
        if row == 0:
            ax_recon.set_title('Reconstruction')
            
        # Add row label
        ax_orig.text(-0.2, 0.5, f'Sample {row + 1}', 
                    transform=ax_orig.transAxes,
                    rotation=90, 
                    verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_path = output_dir / f'autoencoder_multirow{fig_label}.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved multi-row plot to {output_path}")
    
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize autoencoder reconstructions in publication format")
    parser.add_argument('--model-type', type=str, required=True, help='Class of trained model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output-dir', type=str, default='./output/autoencoder_vis', help='Output directory for plots')
    parser.add_argument('--n-rows', type=int, default=3, help='Number of rows in the combined plot')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--show', action='store_true', help='Show the plots')
    args = parser.parse_args()

    print("Loading configuration...")
    cfg = load_and_merge_config()
    
    print("Setting up device...")
    device = "cpu"  # Force CPU for visualization
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
    model.eval()
    print("Model weights loaded successfully.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process samples in batches
    samples_to_plot = []
    batch = dataset[1]
    
    for i in range(0, args.n_samples, args.batch_size):
        end_idx = min(i + args.batch_size, args.n_samples)
        print(f"Processing samples {i} to {end_idx}")
        
        for sample_idx in range(i, end_idx):
            minibatch = {k: v[sample_idx:sample_idx+1] for k, v in batch.items()}
            minibatch = move_batch_to_device(minibatch, device)
            samples_to_plot.append(minibatch)
    
    # Create plots with different numbers of rows
    n_plots = (len(samples_to_plot) + args.n_rows - 1) // args.n_rows
    for plot_idx in range(n_plots):
        start_idx = plot_idx * args.n_rows
        end_idx = min(start_idx + args.n_rows, len(samples_to_plot))
        plot_samples = samples_to_plot[start_idx:end_idx]
        
        plot_multirow_samples(
            model=model,
            samples=plot_samples,
            output_dir=output_dir,
            n_rows=args.n_rows,
            n_cols=3,  # Fixed for autoencoder (original, latent, reconstruction)
            show=args.show,
            fig_label=f"_plot{plot_idx+1}" if n_plots > 1 else ""
        )

if __name__ == "__main__":
    main()