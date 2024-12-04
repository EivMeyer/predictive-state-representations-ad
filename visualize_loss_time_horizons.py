import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, mean_absolute_error
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, move_batch_to_device
from utils.config_utils import load_and_merge_config
from models import get_model_class
from utils.file_utils import find_model_path
from utils.training_utils import load_model_state
import torch.nn.functional as F
import argparse

from plotting_setup import setup_plotting
setup_plotting()

STEP_LENGTH = 0.04

def calculate_metrics(model, dataset, device, num_samples=1000):
    """Calculate prediction metrics across different time horizons."""
    model.eval()
    
    # Initialize metric accumulators
    horizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metrics = {
        'latent_error': {k: [] for k in horizons},
        'obs_error': {k: [] for k in horizons},
        'survival_nll': {k: [] for k in horizons},
        'termination_times': [],
        'predicted_termination_times': [],
        'termination_steps': [],
        'predicted_termination_steps': []
    }

    sample_idx = 0
    
    with torch.no_grad():
        # Initialize tqdm for the entire loop over samples
        pbar = tqdm(total=num_samples, desc="Calculating metrics")
    
        for batch_idx in range(len(dataset)):
            if sample_idx >= num_samples:
                break

            storage_batch = dataset[batch_idx]
            storage_batch = move_batch_to_device(storage_batch, device)
            batch_size = storage_batch['observations'].shape[0]

            for start_idx in range(0, batch_size, 1):
                if sample_idx >= num_samples:
                    break
                    
                end_idx = min(start_idx + 1, batch_size)
                batch = {k: v[start_idx:end_idx] for k, v in storage_batch.items()}
            
                # Get model predictions
                outputs = model(batch)
                
                # Extract predictions and targets
                predicted_latents = outputs['predicted_latents'][0]  # [T, D]
                predictions = outputs['predictions'][0]  # [T, C, H, W]
                target_latents = model.calculate_target_latents(batch)[0]  # [T, D]
                targets = batch['next_observations'][0]  # [T, C, H, W]
                
                # Get hazard predictions if available
                hazard = outputs.get('hazard', None)
                if hazard is not None:
                    hazard = hazard[0]  # [T]
                
                # Get actual termination times
                dones = batch['dones'][0]  # [T]
                if dones.any():
                    actual_term_time_steps = dones.nonzero()[0][0].item() + 1
                    metrics['termination_steps'].append(actual_term_time_steps)
                    metrics['termination_times'].append(STEP_LENGTH*actual_term_time_steps)
                    
                    if hazard is not None:
                        hazard_softplus = F.softplus(hazard)
                        # print("Hazard rates after softplus:", hazard_softplus)

                        cumulative_hazard = torch.cumsum(hazard_softplus, dim=0)
                        cumulative_hazard_padded = torch.cat([torch.zeros(1, device=hazard.device), cumulative_hazard])
                        survival_prob = torch.exp(-cumulative_hazard_padded)
                        # print("Survival probabilities:", survival_prob)

                        pred_term_time_steps = survival_prob[:-1].sum().item()
                        pred_term_time_seconds = pred_term_time_steps * STEP_LENGTH
                        # print("Predicted termination time (seconds):", pred_term_time_seconds)

                        metrics['predicted_termination_steps'].append(pred_term_time_steps)
                        metrics['predicted_termination_times'].append(pred_term_time_seconds)
                
                # Calculate metrics for each horizon
                for k in horizons:
                    if k <= len(predictions):
                        # Latent prediction error
                        if target_latents is not None:
                            latent_error = torch.nn.functional.mse_loss(
                                predicted_latents[:k], target_latents[:k]).item()
                            metrics['latent_error'][k].append(latent_error)
                        
                        # Observation reconstruction error
                        obs_error = torch.nn.functional.mse_loss(
                            predictions[:k], targets[:k]).item()
                        metrics['obs_error'][k].append(obs_error)
                        
                        # Survival negative log likelihood
                        if hazard is not None:
                            # Apply Softplus activation to ensure hazard rates are non-negative
                            hazard_softplus = F.softplus(hazard[:k])
                            
                            # Compute cumulative hazard over the correct dimension
                            cumulative_hazard = torch.cumsum(hazard_softplus, dim=0)
                            
                            # Compute survival probabilities
                            survival_prob = torch.exp(-cumulative_hazard)
                            
                            # Compute negative log-likelihood (NLL)
                            nll = -torch.log(survival_prob + 1e-8).mean().item()
                            
                            # Append NLL to metrics
                            metrics['survival_nll'][k].append(nll)

                # Update the global tqdm progress bar
                sample_idx += 1
                pbar.update(1)  # Increment the global progress bar by 1 for each sample processed
    
    return metrics

def compute_summary_statistics(metrics):
    """Compute summary statistics from collected metrics."""
    summary = {
        'horizons': [],
        'latent_error_mean': [],
        'latent_error_std': [],
        'obs_error_mean': [],
        'obs_error_std': [],
        'survival_nll_mean': [],
        'survival_nll_std': []
    }
    
    for k in sorted(metrics['latent_error'].keys()):
        summary['horizons'].append(k)
        
        if metrics['latent_error'][k]:
            summary['latent_error_mean'].append(np.mean(metrics['latent_error'][k]))
            summary['latent_error_std'].append(np.std(metrics['latent_error'][k]))
        else:
            summary['latent_error_mean'].append(np.nan)
            summary['latent_error_std'].append(np.nan)
            
        summary['obs_error_mean'].append(np.mean(metrics['obs_error'][k]))
        summary['obs_error_std'].append(np.std(metrics['obs_error'][k]))
        
        if metrics['survival_nll'][k]:
            summary['survival_nll_mean'].append(np.mean(metrics['survival_nll'][k]))
            summary['survival_nll_std'].append(np.std(metrics['survival_nll'][k]))
        else:
            summary['survival_nll_mean'].append(np.nan)
            summary['survival_nll_std'].append(np.nan)
    
    # Calculate termination prediction metrics
    if metrics['termination_steps'] and metrics['predicted_termination_steps']:
        actual_times = np.array(metrics['termination_steps'])
        pred_times = np.array(metrics['predicted_termination_steps'])

        actual_times_seconds = np.array(metrics['termination_times'])
        pred_times_seconds = np.array(metrics['predicted_termination_times'])
        
        # Calculate MAE for termination timing
        mae = mean_absolute_error(actual_times_seconds, pred_times_seconds)
        
        # Calculate AUC for termination prediction
        # Convert to binary predictions for each time step
        max_time = max(max(actual_times), max(pred_times))
        y_true = np.array([t <= actual_times[:, None] for t in range(1, int(max_time) + 1)]).T
        y_pred = np.array([t <= pred_times[:, None] for t in range(1, int(max_time) + 1)]).T
        auc = roc_auc_score(y_true.ravel(), y_pred.ravel())
        
        summary['termination_mae'] = mae
        summary['termination_auc'] = auc
    else:
        summary['termination_mae'] = np.nan
        summary['termination_auc'] = np.nan
    
    return summary


def plot_metrics(summary, output_dir):
    """Create publication-quality visualization plots for the metrics with confidence intervals."""
    
    # Prepare data
    horizons = summary['horizons']
    metrics_data = []
    
    for idx, horizon in enumerate(horizons):
        metrics_data.append({
            'Horizon': horizon,
            'Value': summary['latent_error_mean'][idx],
            'Lower': summary['latent_error_mean'][idx] - summary['latent_error_std'][idx],
            'Upper': summary['latent_error_mean'][idx] + summary['latent_error_std'][idx],
            'Metric': '$\\mathcal{L}_{lat}$'
        })
        
        metrics_data.append({
            'Horizon': horizon,
            'Value': summary['obs_error_mean'][idx],
            'Lower': summary['obs_error_mean'][idx] - summary['obs_error_std'][idx],
            'Upper': summary['obs_error_mean'][idx] + summary['obs_error_std'][idx],
            'Metric': '$\\mathcal{L}_{obs}$'
        })
        
        metrics_data.append({
            'Horizon': horizon,
            'Value': summary['survival_nll_mean'][idx],
            'Lower': summary['survival_nll_mean'][idx] - summary['survival_nll_std'][idx],
            'Upper': summary['survival_nll_mean'][idx] + summary['survival_nll_std'][idx],
            'Metric': '$-\\log P(T>k)$'
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Plot settings for each metric
    plot_settings = [
        {
            'metric': '$\\mathcal{L}_{lat}$',
            'ylabel': 'MSE',
            'color': sns.color_palette('deep')[0],
            'filename': 'latent_error.pdf'
        },
        {
            'metric': '$\\mathcal{L}_{obs}$',
            'ylabel': 'MSE',
            'color': sns.color_palette('deep')[1],
            'filename': 'obs_error.pdf'
        },
        {
            'metric': '$-\\log P(T>k)$',
            'ylabel': 'NLL',
            'color': sns.color_palette('deep')[2],
            'filename': 'survival_nll.pdf'
        }
    ]
    
    # Create individual plots
    for settings in plot_settings:
        fig, ax = plt.subplots(figsize=(4, 3))  # Size for subfigure
        metric_data = df[df['Metric'] == settings['metric']]
        
        sns.lineplot(
            data=metric_data,
            x='Horizon',
            y='Value',
            color=settings['color'],
            marker='o',
            markersize=4,
            linewidth=1.5,
            ax=ax
        )
        
        ax.fill_between(
            metric_data['Horizon'],
            metric_data['Lower'],
            metric_data['Upper'],
            color=settings['color'],
            alpha=0.15
        )
        
        ax.set_xlabel('Prediction Horizon $k$')
        ax.set_ylabel(settings['ylabel'])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_axisbelow(True)
        
        if metric_data['Lower'].min() >= 0:
            ax.set_ylim(bottom=0)
        
        # Scientific notation for y-axis if values are very small
        ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='y')
        
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig(output_dir / settings['filename'], dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def create_latex_table(summary):
    """Create a LaTeX table of the results."""
    latex_table = """\\begin{table}[t]
\\caption{Prediction Error Over Time Horizons}
\\label{tab:prediction_error}
\\begin{tabular}{cccc}
\\toprule
Horizon ($k$) & $\\mathcal{L}_{lat}$ & $\\mathcal{L}_{obs}$ & $-\\log P(T > k)$ \\\\
\\midrule
"""
    
    for i, k in enumerate(summary['horizons']):
        latex_table += f"{k} & {summary['latent_error_mean'][i]:.3f} & {summary['obs_error_mean'][i]:.3f} & {summary['survival_nll_mean'][i]:.3f} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_table

def main():
    parser = argparse.ArgumentParser(description="Analyze model predictions and hazard estimations")
    parser.add_argument('--model-type', type=str, required=True, help='Class of trained model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to analyze')
    args = parser.parse_args()

    # Load configuration and setup
    cfg = load_and_merge_config()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else cfg.device)
    
    # Initialize dataset and model
    dataset = EnvironmentDataset(cfg)
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(dataset)
    
    ModelClass = get_model_class(args.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    model = ModelClass(obs_shape=obs_shape, action_dim=action_dim, ego_state_dim=ego_state_dim, cfg=cfg).to(device)
    
    model_path = find_model_path(cfg.project_dir, args.model_path)
    if model_path is None:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    load_model_state(
        model_path=model_path,
        model=model,
        device=device,
        strict=False
    )
    
    # Create output directory
    output_dir = Path(cfg.project_dir) / 'prediction_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(model, dataset, device, args.num_samples)
    
    # Compute summary statistics
    print("Computing summary statistics...")
    summary = compute_summary_statistics(metrics)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_metrics(summary, output_dir)
    
    # Generate LaTeX table
    latex_table = create_latex_table(summary)
    with open(output_dir / 'prediction_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Print summary results
    print("\nResults Summary:")
    print("-" * 50)
    print("Prediction Error Over Time Horizons:")
    for i, k in enumerate(summary['horizons']):
        print(f"\nHorizon k={k}:")
        print(f"  Latent Error: {summary['latent_error_mean'][i]:.3f} ± {summary['latent_error_std'][i]:.3f}")
        print(f"  Observation Error: {summary['obs_error_mean'][i]:.3f} ± {summary['obs_error_std'][i]:.3f}")
        print(f"  Survival NLL: {summary['survival_nll_mean'][i]:.3f} ± {summary['survival_nll_std'][i]:.3f}")
    
    print("\nTermination Prediction Performance:")
    print(f"  AUC: {summary['termination_auc']:.3f}")
    print(f"  MAE: {summary['termination_mae']:.3f} seconds")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()