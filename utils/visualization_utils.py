import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os

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

def normalize(img):
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min) if img_min != img_max else img

def setup_visualization(input_seq_len, pred_seq_len):
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(30, 20))
    
    # Calculate the number of columns and rows needed
    total_cols = max(input_seq_len, pred_seq_len) + 3  # +3 for training grid
    total_rows = 9  # 3 rows per sample (2 samples) + 3 rows for training grid
    
    gs = GridSpec(total_rows, total_cols, figure=fig)
    
    axes = {
        'input_1': [fig.add_subplot(gs[0, i]) for i in range(input_seq_len)],
        'gt_1': [fig.add_subplot(gs[1, i]) for i in range(pred_seq_len)],
        'pred_1': [fig.add_subplot(gs[2, i]) for i in range(pred_seq_len)],
        'input_2': [fig.add_subplot(gs[3, i]) for i in range(input_seq_len)],
        'gt_2': [fig.add_subplot(gs[4, i]) for i in range(pred_seq_len)],
        'pred_2': [fig.add_subplot(gs[5, i]) for i in range(pred_seq_len)],
        'train': [fig.add_subplot(gs[i:i+2, -3:]) for i in range(0, 6, 2)]
    }
    
    plt.show(block=False)  # Show the figure without blocking
    return fig, axes

def visualize_prediction(fig, axes, observations, ground_truth, prediction, epoch, train_predictions, train_ground_truth, metrics):
    def clear_axes(ax_list):
        for ax in ax_list:
            ax.clear()
            ax.axis('off')

    for ax_list in axes.values():
        clear_axes(ax_list)
    
    input_seq_len = observations.shape[1]
    pred_seq_len = prediction.shape[1]
    
    def plot_sample(sample_num):
        # Display input sequence
        for i, ax in enumerate(axes[f'input_{sample_num}']):
            obs = normalize(prepare_image(observations[sample_num-1, i].cpu().numpy()))
            ax.imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
            ax.set_title(f'Input {sample_num} (t-{input_seq_len-1-i})', fontsize=8)
        
        # Display ground truth and predictions
        for i in range(pred_seq_len):
            gt_np = normalize(prepare_image(ground_truth[sample_num-1, i].cpu().numpy()))
            pred_np = normalize(prepare_image(prediction[sample_num-1, i].cpu().numpy()))
            
            axes[f'gt_{sample_num}'][i].imshow(gt_np, cmap='viridis' if gt_np.ndim == 2 else None)
            axes[f'gt_{sample_num}'][i].set_title(f'GT {sample_num} (t+{i+1})', fontsize=8)
            
            axes[f'pred_{sample_num}'][i].imshow(pred_np, cmap='viridis' if pred_np.ndim == 2 else None)
            axes[f'pred_{sample_num}'][i].set_title(f'Pred {sample_num} (t+{i+1})', fontsize=8)
    
    # Plot samples
    plot_sample(1)
    plot_sample(2)
    
    # Display 3x3 grid of training predictions and ground truths
    num_train_preds = min(9, len(train_predictions))
    for i in range(len(axes['train'])):
        ax = axes['train'][i]
        if i < num_train_preds:
            pred_np = normalize(prepare_image(train_predictions[i, 0].cpu().numpy()))
            gt_np = normalize(prepare_image(train_ground_truth[i, 0].cpu().numpy()))
            
            ax.imshow(np.hstack([gt_np, pred_np]), cmap='viridis' if gt_np.ndim == 2 else None)
            ax.set_title(f'Train {i+1}: GT | Pred', fontsize=8)
        else:
            ax.imshow(np.zeros_like(pred_np), cmap='viridis')
            ax.set_title('N/A', fontsize=8)
    
    # Add overall metrics to suptitle
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    fig.suptitle(f'Prediction Analysis - Epoch {epoch}\n\n{metrics_text}', fontsize=16, fontweight='bold')
    
    # Update the figure without bringing it to the front
    fig.canvas.draw()
    fig.canvas.flush_events()
