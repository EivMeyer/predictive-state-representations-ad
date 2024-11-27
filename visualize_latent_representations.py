import argparse
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

FONT_SIZE = 8

from plotting_setup import setup_plotting, setup_plotting_old

def setup_plotting_local(font_size: int = FONT_SIZE):
    setup_plotting(font_size=font_size)

    # Line widths and marker sizes
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 4

    # Set legend frame
    plt.rcParams['legend.frameon'] = False

    # Set other properties
    plt.rcParams['axes.grid'] = True             # Enable grid if preferred
    plt.rcParams['grid.linewidth'] = 0.5

    # DPI (dots per inch) for clarity in publications
    plt.rcParams['figure.dpi'] = 130

def compute_mutual_information(X: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute average mutual information between each latent dimension and target variable.
    
    Args:
        X: Latent representations (n_samples, n_features)
        y: Target variable (n_samples,)
        n_bins: Number of bins for discretization
    
    Returns:
        Average mutual information across all dimensions
    """
    n_features = X.shape[1]
    mi_per_dim = []
    
    # Compute MI for each dimension separately
    for dim in range(n_features):
        x_dim = X[:, dim]
        
        # Discretize continuous variables
        x_bins = np.linspace(x_dim.min(), x_dim.max(), n_bins+1)
        y_bins = np.linspace(y.min(), y.max(), n_bins+1)
        
        # Compute joint and marginal probabilities
        joint_hist, _, _ = np.histogram2d(x_dim, y, bins=[x_bins, y_bins])
        joint_prob = joint_hist / np.sum(joint_hist)
        
        x_prob = np.sum(joint_prob, axis=1)
        y_prob = np.sum(joint_prob, axis=0)
        
        # Compute mutual information
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_prob[i,j] > 0:
                    mi += joint_prob[i,j] * np.log2(joint_prob[i,j] / (x_prob[i] * y_prob[j]))
        
        mi_per_dim.append(mi)
    
    # Return average MI across all dimensions
    return np.mean(mi_per_dim)

def linear_probe_analysis(latent_reps: np.ndarray, target: np.ndarray, n_splits: int = 5) -> Dict[str, List[float]]:
    """Perform linear probing analysis with multiple train-test splits."""
    metrics = {'r2': [], 'mae': [], 'rmse': []} #, 'mi': []}
    
    for _ in range(n_splits):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(latent_reps, target, test_size=0.2)
        
        # Train linear model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        metrics['r2'].append(r2_score(y_test, y_pred))
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        # metrics['mi'].append(compute_mutual_information(X_test, y_test))
    
    return metrics

def generate_linear_probe_results(latent_reps: np.ndarray, metrics: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Generate linear probing results for all metrics."""
    results = []
    
    for metric_name, metric_values in metrics.items():
        # Perform linear probing analysis
        probe_results = linear_probe_analysis(latent_reps, metric_values)
        
        # Format results
        result = {
            'Metric': metric_name,
            'R2': f"{np.mean(probe_results['r2']):.4f}±{np.std(probe_results['r2']):.4f}",
            'MAE': f"{np.mean(probe_results['mae']):.4f}±{np.std(probe_results['mae']):.4f}",
            'RMSE': f"{np.mean(probe_results['rmse']):.4f}±{np.std(probe_results['rmse']):.4f}",
            # 'MI': f"{np.mean(probe_results['mi']):.4f}±{np.std(probe_results['mi']):.4f}"
        }
        results.append(result)
    
    return pd.DataFrame(results)

def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table from results DataFrame."""
    latex_table = "\\begin{table}[t]\n"
    latex_table += "\\caption{Linear Probing Results}\n"
    latex_table += "\\label{tab:linear_probe}\n"
    latex_table += "\\begin{tabular}{lcccc}\n"
    latex_table += "\\hline\n"
    latex_table += "Metric & $R^2$ & MAE & RMSE\\\\\n"
    latex_table += "\\hline\n"
    
    for _, row in df.iterrows():
        latex_table += f"{row['Metric']} & {row['R2']} & {row['MAE']} & {row['RMSE']}\\\\\n"
    
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"
    
    return latex_table

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
    time_until_termination = []
    
    for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
        # Lists to store episode data temporarily
        episode_latent_reps = []
        episode_speeds = []
        episode_steering = []
        episode_accel = []
        episode_distances = []
        episode_obstacles = []
        
        obs = env.reset()
        done = False
        step = 0
        
        # Collect the full episode
        while not done:
            # action, _ = model.predict(obs, deterministic=True)
            # Instead sample random actions for exploration
            action = [env.action_space.sample()]
            new_obs, _, done, info = env.step(action)

            current_time_step = env.get_attr('ego_vehicle_simulation')[0].current_time_step
            ego_position = env.get_attr('ego_vehicle_simulation')[0].ego_vehicle.state.position
            non_ego_obstacles = env.get_attr('ego_vehicle_simulation')[0].current_non_ego_obstacles
            distances_from_ego = [np.linalg.norm(ego_position - obstacle.state_at_time(current_time_step).position) for obstacle in non_ego_obstacles]
            min_distance = min(distances_from_ego) if distances_from_ego else float('inf')
            num_obstacles_within_25_meter_radius = sum([1 for distance in distances_from_ego if distance <= 25.0])
            
            # Store all episode data
            episode_latent_reps.append(obs[0])  # Assuming the latent representation is the first element
            episode_speeds.append(info[0]['vehicle_current_state']['velocity'])
            episode_steering.append(info[0]['vehicle_current_state']['steering_angle'])
            episode_accel.append(info[0]['vehicle_current_state']['acceleration'])
            episode_distances.append(min_distance)
            episode_obstacles.append(num_obstacles_within_25_meter_radius)
            
            obs = new_obs
            step += 1
        
        # Now we know the episode length, calculate time until termination for each step
        episode_length = len(episode_latent_reps)
        episode_time_to_term = [episode_length - t - 1 for t in range(episode_length)]
        
        # Add all episode data to main lists
        latent_reps.extend(episode_latent_reps)
        speeds.extend(episode_speeds)
        steering_angles.extend(episode_steering)
        accelerations.extend(episode_accel)
        min_obstacle_distances.extend(episode_distances)
        num_close_obstacles.extend(episode_obstacles)
        time_until_termination.extend(episode_time_to_term)
        episode_indices.extend([episode] * episode_length)
    
    return {
        'latent_reps': np.array(latent_reps),
        'speeds': np.array(speeds),
        'steering_angles': np.array(steering_angles),
        'accelerations': np.array(accelerations),
        'episode_indices': np.array(episode_indices),
        'min_obstacle_distances': np.array(min_obstacle_distances),
        'num_close_obstacles': np.array(num_close_obstacles),
        'time_until_termination': np.array(time_until_termination)
    }


def plot_3d_latent_projections(latent_reps: np.ndarray, metrics: Dict[str, np.ndarray], output_dir: Path):

    setup_plotting_local()

    sample_size = int(0.05 * len(latent_reps))  # 10% of the data
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
        
        fig.colorbar(scatter, label=metric_name)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_title(f'3D PCA of Latent Space: {metric_name}')
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
        
        fig.colorbar(scatter, label=metric_name)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
        ax.set_title(f'3D t-SNE of Latent Space: {metric_name}')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_tsne_3d_{metric_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Enhanced 3D plots with connecting lines saved for {metric_name}")

def plot_latent_projections(latent_reps: np.ndarray, metrics: Dict[str, np.ndarray], output_dir: Path):

    setup_plotting_local(font_size=11)
    
    sample_size = min(300, len(latent_reps))  # 300  samples or less
    sampled_indices = random_sample_indices(len(latent_reps), sample_size)
    
    sampled_latent_reps = latent_reps[sampled_indices]
    
    for metric_name, metric_values in metrics.items():
        sampled_metric_values = metric_values[sampled_indices]

        # PCA projection
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(sampled_latent_reps)
        
        plt.figure(figsize=(3.5, 2.5))  # Updated figure size for IEEE 2-column format
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=sampled_metric_values, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label=metric_name)
        # plt.title(f'PCA of Latent Space Colored by {metric_name}')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.tight_layout()
        plt.savefig(output_dir / f'latent_pca_{metric_name}.pdf', dpi=100, bbox_inches='tight')
        plt.close()
        
        # UMAP projection
        # umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
        # umap_result = umap_model.fit_transform(sampled_latent_reps)
        # plt.figure(figsize=(3.4, 2.5))  # Updated figure size for IEEE 2-column format
        # scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=sampled_metric_values, cmap='coolwarm', alpha=0.6)
        # plt.colorbar(scatter, label=metric_name)
        # # plt.title(f'UMAP Projection of Latent Space Colored by {metric_name}')
        # plt.xlabel('UMAP-1')
        # plt.ylabel('UMAP-2')
        # plt.tight_layout()
        # plt.savefig(output_dir / f'latent_umap_{metric_name}.pdf', dpi=100, bbox_inches='tight')
        # plt.close()

        # n_samples = sampled_latent_reps.shape[0]  # Number of samples in your dataset
        # perplexity = min(30, n_samples - 1)  # Adjust perplexity to be less than n_samples
        # tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        # tsne_result = tsne.fit_transform(sampled_latent_reps)
        # plt.figure(figsize=(3.4, 2.5))  # Updated figure size for IEEE 2-column format
        # scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=sampled_metric_values, cmap='coolwarm', alpha=0.6)
        # plt.colorbar(scatter, label=metric_name)
        # # plt.title(f'UMAP Projection of Latent Space Colored by {metric_name}')
        # plt.xlabel('t-SNE 1')
        # plt.ylabel('t-SNE 2')
        # plt.tight_layout()
        # plt.savefig(output_dir / f'latent_tsne_{metric_name}.pdf', dpi=100, bbox_inches='tight')
        # plt.close()

        print(f"Plots saved for {metric_name}")

def plot_latent_trajectory(latent_reps: np.ndarray, episode_indices: np.ndarray, output_dir: Path):
    setup_plotting_local()

    os.makedirs(output_dir, exist_ok=True)

    # Filter data for the first episode
    first_episode_mask = episode_indices == 0
    latent_reps = latent_reps[first_episode_mask]

    n_samples = latent_reps.shape[0]  # Number of samples in your dataset
    perplexity = min(30, n_samples - 1)  # Adjust perplexity to be less than n_samples

    # Use t-SNE for this visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(latent_reps)
    
    plt.figure(figsize=(3.4, 2.5))  # Updated figure size for IEEE 2-column format
    
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
    setup_plotting_local()

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

def plot_latent_correlations(latent_reps: np.ndarray, driving_metrics: Dict[str, np.ndarray], output_dir: Path, n_components_heatmap: int = 12, n_components_variance: int = 12):
    setup_plotting_local()

    sample_size = int(0.35 * len(latent_reps))  # 35% of the data
    sampled_indices = np.random.choice(len(latent_reps), size=sample_size, replace=False)

    print("Sampled indices for latent correlation analysis. Total samples:", len(sampled_indices))
    
    sampled_latent_reps = latent_reps[sampled_indices]
    sampled_metrics = {k: v[sampled_indices] for k, v in driving_metrics.items()}
    
    # Normalize the latent representations
    scaler = StandardScaler()
    normalized_latent_reps = scaler.fit_transform(sampled_latent_reps)
    
    # Perform PCA and get explained variance
    pca_heatmap = PCA(n_components=n_components_heatmap)
    top_latent_dims = pca_heatmap.fit_transform(normalized_latent_reps)
    
    correlation_matrix = np.zeros((n_components_heatmap, len(driving_metrics)))
    for i, (metric_name, metric_values) in enumerate(sampled_metrics.items()):
        for j in range(n_components_heatmap):
            correlation_matrix[j, i], _ = pearsonr(top_latent_dims[:, j], metric_values)
    
    # Plot correlation heatmap
    plt.figure(figsize=(4.0, 2.0))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=list(driving_metrics.keys()),
                yticklabels=[f'PC {i+1}' for i in range(n_components_heatmap)], fmt='.2f', cbar=True)
    plt.yticks(rotation=0)  # Ensure y-axis labels are not rotated
    # plt.title('Correlation between PCs and Driving Metrics')
    plt.tight_layout()
    # Remove grid lines
    plt.grid(False)
    plt.savefig(output_dir / 'latent_correlation_heatmap.pdf', dpi=100, bbox_inches='tight')
    plt.close()

    setup_plotting_local(font_size=7)

    pca_explained_variance = PCA(n_components=n_components_variance)
    pca_explained_variance.fit(normalized_latent_reps)
    explained_variance = pca_explained_variance.explained_variance_ratio_ * 100  # Convert to percentage

    # Plot explained variance
    plt.figure(figsize=(3.5, 1.4))
    plt.bar(range(1, n_components_variance + 1), explained_variance, align='center', alpha=0.8)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance')
    # Remove grid lines
    plt.grid(False)
    # plt.title('Explained Variance by Principal Component')
    plt.xticks(range(1, n_components_variance + 1), [f'{i}' for i in range(1, n_components_variance + 1)])
    # Increase max y-axis limit slightly to make space for percentage labels
    plt.ylim(0, min(100, max(explained_variance) + 5))

    # Add percentage labels on top of each bar
    for i, v in enumerate(explained_variance):
        plt.text(i + 1, v, f'{v:.1f}\%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'explained_variance.pdf', dpi=100, bbox_inches='tight')
    plt.close()

    print("Latent correlation heatmap and explained variance plots (normalized data) saved.")

@config_wrapper()
def main(cfg):
    parser = argparse.ArgumentParser(description="Analyze latent space of a trained RL agent")
    parser.add_argument('mode', choices=['collect', 'plot', 'probing'], help="Run mode: collect data or plot results or do linear probing")
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
        # model_path = sorted(Path(cfg.project_dir).rglob('*.zip'), key=lambda x: x.stat().st_mtime, reverse=True)[0]
        # print(f"Loading model from: {model_path}")
        # model = PPO.load(model_path, env=env)

        # Create new model
        model = PPO(
            "MlpPolicy",
            env
        )

        # Collect latent representations and driving metrics
        data = collect_latent_representations(model, env, args.num_episodes)

        # Save the collected data
        np.savez(output_dir / 'latent_data.npz', **data)
        print(f"Latent representations and driving metrics saved to {output_dir / 'latent_data.npz'}")

    elif args.mode == 'probing':
        # Load data and perform linear probing
        data = np.load(output_dir / 'latent_data.npz')
        latent_reps = data['latent_reps']
        
        # Define metrics for probing
        metrics = {
            'Speed': data['speeds'],
            'Steering': data['steering_angles'],
            'Log NVD': np.log10(data['min_obstacle_distances'] + 0.01),
            'Log TTT': np.log10(data['time_until_termination']*0.04 + 0.1)
        }

        # Generate results
        results_df = generate_linear_probe_results(latent_reps, metrics)
        print("\nLinear Probing Results:")
        print(results_df)
        
        # Save results
        latex_table = generate_latex_table(results_df)
        with open(output_dir / 'linear_probe_table.tex', 'w') as f:
            f.write(latex_table)
        results_df.to_csv(output_dir / 'linear_probe_results.csv')
        print(f"\nSaved to:\n- {output_dir / 'linear_probe_table.tex'}\n- {output_dir / 'linear_probe_results.csv'}")


    elif args.mode == 'plot':
        # Load the collected data
        data = np.load(output_dir / 'latent_data.npz')
        print("Data loaded successfully")

        latent_reps = data['latent_reps']
        episode_indices = data['episode_indices']
        num_episodes = len(np.unique(episode_indices))
        print(f"Number of episodes: {num_episodes}")
        print(f"Number of samples: {len(latent_reps)}")
        print(f"Average samples per episode: {len(latent_reps) / num_episodes:.2f}")

        # Generate plots for each metric
        metrics = {
            'speed': data['speeds'],
            # 'acceleration': data['accelerations'],
            'steering': data['steering_angles'],
            'log NVD': np.log10(data['min_obstacle_distances'] + 0.01),
            # 'close vehicles': data['num_close_obstacles'],
            'log TTT': np.log10(data['time_until_termination']*0.04 + 0.1)
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