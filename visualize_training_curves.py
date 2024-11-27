import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from scipy.ndimage import gaussian_filter1d
import os

def setup_plotting(font_size=9):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size-1,
        "ytick.labelsize": font_size-1,
        "legend.fontsize": font_size-1,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "figure.dpi": 300
    })

# def smooth_data(x, y, window_size=100):
#     """Smooth data using Gaussian filter and calculate confidence intervals."""
#     # Remove NaN values
#     mask = ~np.isnan(y)
#     x, y = x[mask], y[mask]
    
#     # Calculate rolling std for confidence intervals
#     sigma = window_size / 4  # Convert window size to sigma
#     y_smooth = gaussian_filter1d(y, sigma, mode='reflect')
    
#     # Calculate rolling standard deviation
#     rolling_std = np.array([np.std(y[max(0, i-window_size):min(len(y), i+window_size)]) 
#                            for i in range(len(y))])
#     stderr = rolling_std / np.sqrt(window_size)
    
#     y_upper = y_smooth + 2 * stderr
#     y_lower = y_smooth - 2 * stderr
    
#     return x, y, y_smooth, y_lower, y_upper

def smooth_data(x, y, window_size=100, num_std=0.1):
    """Smooth data using EMA and calculate variability bands."""
    # Remove NaN values
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    
    # EMA smoothing
    alpha = 2 / (window_size + 1)
    y_smooth = pd.Series(y).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    
    # Calculate residuals
    residuals = y - y_smooth
    
    # EWM standard deviation of residuals
    y_ewm_std = pd.Series(residuals).ewm(alpha=alpha, adjust=False).std().to_numpy()
    stderr = y_ewm_std  # Use directly without dividing by sqrt(window_size)
    
    # Variability bands
    y_upper = y_smooth + num_std * stderr
    y_lower = y_smooth - num_std * stderr
    
    return x, y, y_smooth, y_lower, y_upper

def process_data(df, metric_name="train/ep_cumulative_reward"):
    """Extract and process data for each run."""
    runs = {}
    for col in df.columns:
        if metric_name in col and "__MIN" not in col and "__MAX" not in col:
            run_name = col.split(" - ")[0]
            runs[run_name] = {
                'steps': df['Step'].values,
                'values': df[col].values,
            }
    return runs

def get_deterministic_mapping(names, n_colors=10):
    """Create deterministic color mapping from sorted names."""
    return {name: plt.cm.tab10(i % n_colors) for i, name in enumerate(sorted(names))}

def clean_filename(value):
    """Clean a string to be a valid filename."""
    return value.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('²', '2')

def plot_training_curves(runs, output_path, ylabel, window_size=100, num_std=2):
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    
    # Clean and standardize names first
    used_names = set()
    clean_names = {}
    for raw_name in runs.keys():
        name = raw_name.split(' ')[-1].title()
        if name == 'Vae':
            name = 'VAE'
        elif name == 'End-to-end':
            name = 'E2E'
        if name in used_names:
            print(f"Warning: Duplicate name {raw_name}")
            continue
        used_names.add(name)
        clean_names[raw_name] = name
    
    # Create color mapping from clean names
    colors = get_deterministic_mapping(set(clean_names.values()))
    
    # Plot with consistent colors
    for raw_name in sorted(runs.keys()):
        if raw_name not in clean_names:
            continue
        data = runs[raw_name]
        name = clean_names[raw_name]
        
        x, y, y_smooth, y_lower, y_upper = smooth_data(
            data['steps'], 
            data['values'], 
            window_size=window_size,
            num_std=num_std
        )
        
        ax.plot(x, y_smooth, label=name, color=colors[name], alpha=0.8)
        ax.fill_between(x, y_lower, y_upper, color=colors[name], alpha=0.2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', 
             ncol=2, borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def get_available_metrics(df):
    """Extract available metrics from DataFrame columns."""
    metrics = set()
    for col in df.columns:
        if " - " in col and "__MIN" not in col and "__MAX" not in col:
            metric = col.split(" - ")[1]
            metrics.add(metric)
    return sorted(list(metrics))

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from WandB CSV exports')
    parser.add_argument('csv_files', nargs='+', help='CSV files containing training data')
    parser.add_argument('--metric', default=None, help='Metric to plot (auto-detected if not specified)')
    parser.add_argument('--window', type=int, default=100, help='Smoothing window size')
    parser.add_argument('--num-std', type=float, default=2, help='Number of standard deviations for variability bands')
    parser.add_argument('--ylabel', default='Episode Return', help='Y-axis label')
    parser.add_argument('--x-cutoff', type=int, default=None, help='Cut off x-axis at this value')
    parser.add_argument('--font-size', type=int, default=7, help='Font size for plotting')
    args = parser.parse_args()

    setup_plotting(font_size=args.font_size)

    # Read first CSV to detect metrics
    df = pd.read_csv(args.csv_files[0])
    available_metrics = get_available_metrics(df)
    
    if not available_metrics:
        raise ValueError("No metrics found in CSV")
        
    if args.metric is None:
        args.metric = available_metrics[0]
        print(f"Available metrics: {', '.join(available_metrics)}")
        print(f"Auto-selected metric: {args.metric}")

    all_runs = {}
    for csv_file in args.csv_files:
        df = pd.read_csv(csv_file)
        if args.x_cutoff is not None:
            df = df[df['Step'] <= args.x_cutoff]
        runs = process_data(df, args.metric)
        all_runs.update(runs)

    output_file = os.path.join('output', 'training_curves',clean_filename(args.ylabel.lower()) + '.pdf')
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    plot_training_curves(all_runs, output_file, args.ylabel, args.window, args.num_std)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()