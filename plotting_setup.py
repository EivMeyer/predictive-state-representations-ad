import matplotlib.pyplot as plt
import seaborn as sns

# Initial font size
FONT_SIZE = 14

def setup_plotting_old(font_size=None):
    global FONT_SIZE
    if font_size is not None:
        FONT_SIZE = font_size
    
    # Set style and theme
    plt.style.use('seaborn-v0_8-paper')
    sns.set_theme(context="paper", font_scale=1.2)
    
    # Update rcParams for consistent font sizes and style
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "figure.labelsize": FONT_SIZE,
        "font.family": 'DejaVu Sans',
        "grid.linestyle": '--',
        "grid.alpha": 0.7,
    })

import matplotlib.pyplot as plt

def setup_plotting(font_size=7):
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
        "figure.dpi": 200
    })