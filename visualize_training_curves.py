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

def smooth_data(x, y, window_size=100):
    """Smooth data using Gaussian filter and calculate confidence intervals."""
    # Remove NaN values
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    
    # Calculate rolling std for confidence intervals
    sigma = window_size / 4  # Convert window size to sigma
    y_smooth = gaussian_filter1d(y, sigma)
    
    # Calculate rolling standard deviation
    rolling_std = np.array([np.std(y[max(0, i-window_size):min(len(y), i+window_size)]) 
                           for i in range(len(y))])
    stderr = rolling_std / np.sqrt(window_size)
    
    y_upper = y_smooth + 2 * stderr
    y_lower = y_smooth - 2 * stderr
    
    return x, y_smooth, y_lower, y_upper

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

def plot_training_curves(runs, output_path, window_size=100):
    """Create publication-quality plot of training curves with confidence intervals."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    
    for (name, data), color in zip(runs.items(), colors):
        x, y_smooth, y_lower, y_upper = smooth_data(
            data['steps'], 
            data['values'], 
            window_size=window_size
        )
        
        ax.plot(x, y_smooth, label=name, color=color, alpha=0.8)
        ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Return')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', 
             ncol=2, borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from WandB CSV exports')
    parser.add_argument('csv_files', nargs='+', help='CSV files containing training data')
    parser.add_argument('--output', default='training_curves.pdf', help='Output file path')
    parser.add_argument('--metric', default='train/ep_cumulative_reward', help='Metric to plot')
    parser.add_argument('--window', type=int, default=100, help='Smoothing window size')
    args = parser.parse_args()

    setup_plotting()

    all_runs = {}
    for csv_file in args.csv_files:
        df = pd.read_csv(csv_file)
        runs = process_data(df, args.metric)
        all_runs.update(runs)

    output_file = os.path.join('output', args.output)

    plot_training_curves(all_runs, output_file, args.window)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()