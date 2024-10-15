import argparse
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.config_utils import config_wrapper
from environments import get_environment
from typing import List, Tuple, Dict
import pandas as pd
from scipy.stats import pearsonr

# Set figure dimensions (width = 3.5 inches for 2-column format)
plt.rcParams['figure.figsize'] = [3.5, 2.5]  # Width and height in inches

# Set font sizes for improved readability
plt.rcParams['font.size'] = 9                # General font size
plt.rcParams['axes.titlesize'] = 9           # Title font size
plt.rcParams['axes.labelsize'] = 9           # Axis label font size
plt.rcParams['xtick.labelsize'] = 8          # X-tick label font size
plt.rcParams['ytick.labelsize'] = 8          # Y-tick label font size
plt.rcParams['legend.fontsize'] = 8          # Legend font size

# Line widths and marker sizes
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 4

# Set legend frame
plt.rcParams['legend.frameon'] = False

# Set other properties
plt.rcParams['axes.grid'] = True             # Enable grid if preferred
plt.rcParams['grid.linewidth'] = 0.5

# DPI (dots per inch) for clarity in publications
plt.rcParams['figure.dpi'] = 300

def collect_latent_representations(model, env, num_episodes: int = 1000) -> Dict[str, np.ndarray]:
    latent_reps = []
    speeds = []
    steering_angles = []
    accelerations = []
    
    for _ in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            new_obs, _, done, info = env.step(action)
            
            latent_rep = obs[0]  # Assuming the latent representation is the first element of the observation
            latent_reps.append(latent_rep)
            
            # Collect driving metrics (you may need to adjust these based on your environment)
            speeds.append(info[0]['vehicle_current_state']['velocity'])
            steering_angles.append(info[0]['vehicle_current_state']['steering_angle'])
            accelerations.append(info[0]['vehicle_current_state']['acceleration'])
            
            obs = new_obs
    
    return {
        'latent_reps': np.array(latent_reps),
        'speeds': np.array(speeds),
        'steering_angles': np.array(steering_angles),
        'accelerations': np.array(accelerations),
    }

def plot_latent_projections(latent_reps: np.ndarray, metrics: Dict[str, np.ndarray], output_dir: Path):
    for metric_name, metric_values in metrics.items():
        # PCA projection
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(latent_reps)
        
        plt.figure(figsize=(3.5, 2.5))  # Updated figure size for IEEE 2-column format
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=metric_values, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label=metric_name.capitalize())
        plt.title(f'PCA of Latent Space Colored by {metric_name.capitalize()}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_pca_{metric_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # UMAP projection
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
        umap_result = umap_model.fit_transform(latent_reps)
        
        plt.figure(figsize=(3.5, 2.5))  # Updated figure size for IEEE 2-column format
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=metric_values, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label=metric_name.capitalize())
        plt.title(f'UMAP Projection of Latent Space Colored by {metric_name.capitalize()}')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_umap_{metric_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

def plot_latent_trajectory(latent_reps: np.ndarray, output_dir: Path):
    # Use t-SNE for this visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(latent_reps)
    
    plt.figure(figsize=(3.5, 2.5))  # Updated figure size for IEEE 2-column format
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.1, color='gray')
    
    # Plot trajectory for a single episode (adjust the slice as needed)
    episode_length = min(1000, len(tsne_result))  # Adjust based on your typical episode length
    colors = plt.cm.viridis(np.linspace(0, 1, episode_length))
    
    for i in range(episode_length - 1):
        plt.plot(tsne_result[i:i+2, 0], tsne_result[i:i+2, 1], c=colors[i])
    
    # Add markers for significant events (you'll need to identify these in your data)
    # Uncomment and adjust these lines when you have the data for significant events
    # turn_indices = [100, 300, 700]  # Example indices, replace with actual data
    # stop_indices = [200, 500]  # Example indices, replace with actual data
    # overtake_indices = [400, 800]  # Example indices, replace with actual data
    # plt.scatter(tsne_result[turn_indices, 0], tsne_result[turn_indices, 1], marker='o', s=100, c='red', label='Turns')
    # plt.scatter(tsne_result[stop_indices, 0], tsne_result[stop_indices, 1], marker='s', s=100, c='blue', label='Stops')
    # plt.scatter(tsne_result[overtake_indices, 0], tsne_result[overtake_indices, 1], marker='^', s=100, c='green', label='Overtaking')
    
    plt.title('Trajectory of a Single Episode in Latent Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # plt.legend()  # Uncomment when you add the markers for significant events
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_trajectory.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_correlations(latent_reps: np.ndarray, driving_metrics: Dict[str, np.ndarray], output_dir: Path):
    # Select top 10 latent dimensions with highest variance
    pca = PCA(n_components=10)
    top_latent_dims = pca.fit_transform(latent_reps)
    
    correlation_matrix = np.zeros((10, len(driving_metrics)))
    for i, metric in enumerate(driving_metrics.keys()):
        for j in range(10):
            correlation_matrix[j, i], _ = pearsonr(top_latent_dims[:, j], driving_metrics[metric])
    
    plt.figure(figsize=(3.5, 4.5))  # Taller figure for the correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=list(driving_metrics.keys()),
                yticklabels=[f'Dim {i+1}' for i in range(10)])
    plt.title('Correlation between Latent Dimensions and Driving Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_correlation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

@config_wrapper()
def main(cfg):
    parser = argparse.ArgumentParser(description="Analyze latent space of a trained RL agent")
    parser.add_argument('mode', choices=['collect', 'plot'], help="Run mode: collect data or plot results")
    parser.add_argument('--num_episodes', type=int, default=1000, help="Number of episodes to collect data from")
    parser.add_argument('--output_dir', type=str, default='./latent_analysis', help="Directory to save output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'collect':
        # Create the environment
        env_class = get_environment(cfg.environment)
        env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, rl_mode=True)

        # Load the trained model
        model_path = sorted(Path(cfg.project_dir).rglob('*.zip'), key=lambda x: x.stat().st_mtime, reverse=True)[0]
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=env)

        # Collect latent representations and driving metrics
        data = collect_latent_representations(model, env, args.num_episodes)

        # Save the collected data
        np.savez(output_dir / 'latent_data.npz', **data)
        print(f"Latent representations and driving metrics saved to {output_dir / 'latent_data.npz'}")

    elif args.mode == 'plot':
        # Load the collected data
        data = np.load(output_dir / 'latent_data.npz')
        latent_reps = data['latent_reps']
        

        # Generate plots for each metric
        metrics = {
            'speed': data['speeds'],
            'steering': data['steering_angles'],
            'acceleration': data['accelerations']
        }
        plot_latent_projections(latent_reps, metrics, output_dir)
        plot_latent_trajectory(latent_reps, output_dir)
        plot_latent_correlations(latent_reps, metrics, output_dir)

        print(f"Analysis complete. Plots saved in {output_dir}")

if __name__ == "__main__":
    main()