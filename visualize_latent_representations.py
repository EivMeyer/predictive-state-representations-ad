import argparse
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
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
plt.rcParams['figure.dpi'] = 100

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

setup_plot_style()

def random_sample_indices(data_length, sample_size):
    """Randomly sample indices."""
    return np.random.choice(data_length, size=sample_size, replace=False)

def collect_latent_representations(model, env, num_episodes: int = 1000) -> Dict[str, np.ndarray]:
    latent_reps = []
    speeds = []
    steering_angles = []
    accelerations = []
    episode_indices = []
    min_obstacle_distances = []
    num_close_obstacles = []
    
    for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            new_obs, _, done, info = env.step(action)

            current_time_step = env.get_attr('ego_vehicle_simulation')[0].current_time_step
            ego_position = env.get_attr('ego_vehicle_simulation')[0].ego_vehicle.state.position
            non_ego_obstacles = env.get_attr('ego_vehicle_simulation')[0].current_non_ego_obstacles
            distances_from_ego = [np.linalg.norm(ego_position - obstacle.state_at_time(current_time_step).position) for obstacle in non_ego_obstacles]
            min_distance = min(distances_from_ego)
            num_obstacles_within_25_meter_radius = sum([1 for distance in distances_from_ego if distance <= 25.0])
            
            latent_rep = obs[0]  # Assuming the latent representation is the first element of the observation
            latent_reps.append(latent_rep)
            
            # Collect driving metrics (you may need to adjust these based on your environment)
            min_obstacle_distances.append(min_distance)
            num_close_obstacles.append(num_obstacles_within_25_meter_radius)
            speeds.append(info[0]['vehicle_current_state']['velocity'])
            steering_angles.append(info[0]['vehicle_current_state']['steering_angle'])
            accelerations.append(info[0]['vehicle_current_state']['acceleration'])
            episode_indices.append(episode)
            
            obs = new_obs
            step += 1
    
    return {
        'latent_reps': np.array(latent_reps),
        'speeds': np.array(speeds),
        'steering_angles': np.array(steering_angles),
        'accelerations': np.array(accelerations),
        'episode_indices': np.array(episode_indices),
        'min_obstacle_distances': np.array(min_obstacle_distances),
        'num_close_obstacles': np.array(num_close_obstacles)
    }

import argparse
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm
from utils.config_utils import config_wrapper
from environments import get_environment
from typing import List, Tuple, Dict
import pandas as pd
from scipy.stats import pearsonr

# ... (keep all the existing imports and plot style configurations)

def plot_3d_latent_projections(latent_reps: np.ndarray, metrics: Dict[str, np.ndarray], output_dir: Path):
    sample_size = int(0.1 * len(latent_reps))  # 10% of the data
    sampled_indices = random_sample_indices(len(latent_reps), sample_size)
    
    sampled_latent_reps = latent_reps[sampled_indices]
    
    for metric_name, metric_values in metrics.items():
        sampled_metric_values = metric_values[sampled_indices]

        # 3D PCA projection
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(sampled_latent_reps)
        
        fig = plt.figure(figsize=(12, 10))  # Even larger figure for better visibility
        ax = fig.add_subplot(111, projection='3d')
        
        # Main scatter plot
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
                             c=sampled_metric_values, cmap='coolwarm', alpha=0.6)
        
        # Shadow projections and connecting lines
        min_x, min_y, min_z = np.min(pca_result, axis=0)
        
        for i in range(len(pca_result)):
            x, y, z = pca_result[i]
            # XY plane projection
            ax.scatter(x, y, min_z, c=sampled_metric_values[i], cmap='coolwarm', alpha=0.1)
            ax.plot([x, x], [y, y], [z, min_z], 'k:', lw=0.5, alpha=0.3)
            # XZ plane projection
            ax.scatter(x, min_y, z, c=sampled_metric_values[i], cmap='coolwarm', alpha=0.1)
            ax.plot([x, x], [y, min_y], [z, z], 'k:', lw=0.5, alpha=0.3)
            # YZ plane projection
            ax.scatter(min_x, y, z, c=sampled_metric_values[i], cmap='coolwarm', alpha=0.1)
            ax.plot([x, min_x], [y, y], [z, z], 'k:', lw=0.5, alpha=0.3)
        
        fig.colorbar(scatter, label=metric_name.capitalize())
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_title(f'3D PCA of Latent Space: {metric_name.capitalize()}')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_pca_3d_{metric_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # 3D t-SNE projection
        n_samples = sampled_latent_reps.shape[0]
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(sampled_latent_reps)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Main scatter plot
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], 
                             c=sampled_metric_values, cmap='coolwarm', alpha=0.6)
        
        # Shadow projections and connecting lines
        min_x, min_y, min_z = np.min(tsne_result, axis=0)
        
        for i in range(len(tsne_result)):
            x, y, z = tsne_result[i]
            # XY plane projection
            ax.scatter(x, y, min_z, c=sampled_metric_values[i], cmap='coolwarm', alpha=0.1)
            ax.plot([x, x], [y, y], [z, min_z], 'k:', lw=0.5, alpha=0.3)
            # XZ plane projection
            ax.scatter(x, min_y, z, c=sampled_metric_values[i], cmap='coolwarm', alpha=0.1)
            ax.plot([x, x], [y, min_y], [z, z], 'k:', lw=0.5, alpha=0.3)
            # YZ plane projection
            ax.scatter(min_x, y, z, c=sampled_metric_values[i], cmap='coolwarm', alpha=0.1)
            ax.plot([x, min_x], [y, y], [z, z], 'k:', lw=0.5, alpha=0.3)
        
        fig.colorbar(scatter, label=metric_name.capitalize())
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
        ax.set_title(f'3D t-SNE of Latent Space: {metric_name.capitalize()}')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_tsne_3d_{metric_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Enhanced 3D plots with connecting lines saved for {metric_name.capitalize()}")

def plot_latent_projections(latent_reps: np.ndarray, metrics: Dict[str, np.ndarray], output_dir: Path):
    sample_size = int(0.1 * len(latent_reps))  # 10% of the data
    sampled_indices = random_sample_indices(len(latent_reps), sample_size)
    
    sampled_latent_reps = latent_reps[sampled_indices]
    
    for metric_name, metric_values in metrics.items():
        sampled_metric_values = metric_values[sampled_indices]

        # PCA projection
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(sampled_latent_reps)
        
        plt.figure(figsize=(3.5, 2.5))  # Updated figure size for IEEE 2-column format
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=sampled_metric_values, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label=metric_name.capitalize())
        # plt.title(f'PCA of Latent Space Colored by {metric_name.capitalize()}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_pca_{metric_name}.pdf', dpi=100, bbox_inches='tight')
        plt.close()
        
        # UMAP projection
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
        umap_result = umap_model.fit_transform(sampled_latent_reps)
        plt.figure(figsize=(3.5, 2.5))  # Updated figure size for IEEE 2-column format
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=sampled_metric_values, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label=metric_name.capitalize())
        # plt.title(f'UMAP Projection of Latent Space Colored by {metric_name.capitalize()}')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_umap_{metric_name}.pdf', dpi=100, bbox_inches='tight')
        plt.close()


        n_samples = sampled_latent_reps.shape[0]  # Number of samples in your dataset
        perplexity = min(30, n_samples - 1)  # Adjust perplexity to be less than n_samples
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(sampled_latent_reps)
        plt.figure(figsize=(3.5, 2.5))  # Updated figure size for IEEE 2-column format
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=sampled_metric_values, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label=metric_name.capitalize())
        # plt.title(f'UMAP Projection of Latent Space Colored by {metric_name.capitalize()}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_tsne_{metric_name}.pdf', dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Plots saved for {metric_name.capitalize()}")

def plot_latent_trajectory(latent_reps: np.ndarray, episode_indices: np.ndarray, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    # Filter data for the first episode
    first_episode_mask = episode_indices == 0
    latent_reps = latent_reps[first_episode_mask]

    n_samples = latent_reps.shape[0]  # Number of samples in your dataset
    perplexity = min(30, n_samples - 1)  # Adjust perplexity to be less than n_samples

    # Use t-SNE for this visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(latent_reps)
    
    plt.figure(figsize=(3.5, 2.5))  # Updated figure size for IEEE 2-column format
    
    # Plot trajectory for the first episode
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(tsne_result)))
    
    for i in range(len(tsne_result) - 1):
        plt.plot(tsne_result[i:i+2, 0], tsne_result[i:i+2, 1], c=colors[i])
    
    # Add markers for start and end points
    plt.scatter(tsne_result[0, 0], tsne_result[0, 1], marker='o', s=100, c='green', label='Start')
    plt.scatter(tsne_result[-1, 0], tsne_result[-1, 1], marker='s', s=100, c='red', label='End')
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_trajectory_first_episode.pdf', dpi=100, bbox_inches='tight')
    plt.close()

    print("Latent trajectory plot for the first episode saved.")

def plot_3d_latent_trajectory(latent_reps: np.ndarray, episode_indices: np.ndarray, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    # Filter data for the first episode
    first_episode_mask = episode_indices == 0
    latent_reps = latent_reps[first_episode_mask]

    n_samples = latent_reps.shape[0]  # Number of samples in your dataset
    perplexity = min(30, n_samples - 1)  # Adjust perplexity to be less than n_samples

    # Use t-SNE for this visualization
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(latent_reps)
    
    fig = plt.figure(figsize=(12, 10))  # Larger figure for better visibility
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory for the first episode
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(tsne_result)))
    
    for i in range(len(tsne_result) - 1):
        ax.plot(tsne_result[i:i+2, 0], tsne_result[i:i+2, 1], tsne_result[i:i+2, 2], c=colors[i])
    
    # Add markers for start and end points
    ax.scatter(tsne_result[0, 0], tsne_result[0, 1], tsne_result[0, 2], marker='o', s=100, c='green', label='Start')
    ax.scatter(tsne_result[-1, 0], tsne_result[-1, 1], tsne_result[-1, 2], marker='s', s=100, c='red', label='End')
    
    # Add projections and connecting lines
    min_x, min_y, min_z = np.min(tsne_result, axis=0)
    
    for i in range(len(tsne_result)):
        x, y, z = tsne_result[i]
        # XY plane projection
        ax.scatter(x, y, min_z, c=colors[i], alpha=0.1)
        ax.plot([x, x], [y, y], [z, min_z], 'k:', lw=0.5, alpha=0.3)
        # XZ plane projection
        ax.scatter(x, min_y, z, c=colors[i], alpha=0.1)
        ax.plot([x, x], [y, min_y], [z, z], 'k:', lw=0.5, alpha=0.3)
        # YZ plane projection
        ax.scatter(min_x, y, z, c=colors[i], alpha=0.1)
        ax.plot([x, min_x], [y, y], [z, z], 'k:', lw=0.5, alpha=0.3)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.legend()
    # plt.title('3D Latent Trajectory for First Episode')
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_trajectory_3d_first_episode.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("3D Latent trajectory plot for the first episode saved.")

def plot_latent_correlations(latent_reps: np.ndarray, driving_metrics: Dict[str, np.ndarray], output_dir: Path, n_components: int = 6):
    sample_size = int(0.35 * len(latent_reps))  # 35% of the data
    sampled_indices = np.random.choice(len(latent_reps), size=sample_size, replace=False)

    print("Sampled indices for latent correlation analysis. Total samples:", len(sampled_indices))
    
    sampled_latent_reps = latent_reps[sampled_indices]
    sampled_metrics = {k: v[sampled_indices] for k, v in driving_metrics.items()}
    
    # Normalize the latent representations
    scaler = StandardScaler()
    normalized_latent_reps = scaler.fit_transform(sampled_latent_reps)
    
    # Perform PCA and get explained variance
    pca = PCA(n_components=n_components)
    top_latent_dims = pca.fit_transform(normalized_latent_reps)
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
    
    correlation_matrix = np.zeros((n_components, len(driving_metrics)))
    for i, (metric_name, metric_values) in enumerate(sampled_metrics.items()):
        for j in range(n_components):
            correlation_matrix[j, i], _ = pearsonr(top_latent_dims[:, j], metric_values)
    
    # Plot correlation heatmap
    plt.figure(figsize=(4.0, 3.0))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=list(driving_metrics.keys()),
                yticklabels=[f'PC {i+1}' for i in range(n_components)], fmt='.2f', cbar=True)
    plt.yticks(rotation=0)  # Ensure y-axis labels are not rotated
    # plt.title('Correlation between PCs and Driving Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_correlation_heatmap.pdf', dpi=100, bbox_inches='tight')
    plt.close()

    # Plot explained variance
    plt.figure(figsize=(4.0, 3.0))
    plt.bar(range(1, n_components + 1), explained_variance, align='center', alpha=0.8)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    # plt.title('Explained Variance by Principal Component')
    plt.xticks(range(1, n_components + 1), [f'{i}' for i in range(1, n_components + 1)])
    
    # Add percentage labels on top of each bar
    for i, v in enumerate(explained_variance):
        plt.text(i + 1, v, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'explained_variance.pdf', dpi=100, bbox_inches='tight')
    plt.close()

    print("Latent correlation heatmap and explained variance plots (normalized data) saved.")

@config_wrapper()
def main(cfg):
    parser = argparse.ArgumentParser(description="Analyze latent space of a trained RL agent")
    parser.add_argument('mode', choices=['collect', 'plot'], help="Run mode: collect data or plot results")
    parser.add_argument('--num-episodes', type=int, default=1000, help="Number of episodes to collect data from")
    parser.add_argument('--output-dir', type=str, default='./output/latent_analysis', help="Directory to save output files")
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
        print("Data loaded successfully")
        latent_reps = data['latent_reps']
        episode_indices = data['episode_indices']

        # Generate plots for each metric
        metrics = {
            'speed': data['speeds'],
            'steering': data['steering_angles'],
            'acceleration': data['accelerations'],
            'log. min. distance': np.log(data['min_obstacle_distances'] + 0.01),
            'close vehicles': data['num_close_obstacles']
        }
        
        plot_latent_projections(latent_reps, metrics, output_dir)
        print("Plots saved for 2D latent projections")
        # plot_3d_latent_projections(latent_reps, metrics, output_dir)  # Add this line
        # print("Plots saved for 3D latent projections")
        plot_latent_trajectory(latent_reps, episode_indices, output_dir)
        print("2D Latent trajectory plot saved")
        plot_3d_latent_trajectory(latent_reps, episode_indices, output_dir)  # Add this line
        print("3D Latent trajectory plot saved")
        plot_latent_correlations(latent_reps, metrics, output_dir)
        print("Plot saved for latent correlations") 

        print(f"Analysis complete. Plots saved in {output_dir}")

if __name__ == "__main__":
    main()