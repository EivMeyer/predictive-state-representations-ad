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

def setup_visualization(num_segments, input_seq_len, pred_seq_len):
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(30, 20))
    
    # Calculate the number of columns and rows needed
    total_cols = max(input_seq_len, pred_seq_len * 2) + 3  # *2 for GT and Pred side by side, +3 for training grid
    total_rows = 1 + num_segments * 2 + 3  # Input + (GT+Pred) * num_segments + 3 for training grid
    
    gs = GridSpec(total_rows, total_cols, figure=fig)
    
    axes = {
        'input': [fig.add_subplot(gs[0, i]) for i in range(input_seq_len)],
        'segments': {},
        'train': [fig.add_subplot(gs[i:i+2, -3:]) for i in range(total_rows-3, total_rows, 1)]
    }
    
    # Setup axes for each segment (ground truth and predictions side by side)
    for seg in range(num_segments):
        axes['segments'][seg] = [fig.add_subplot(gs[seg*2+1:seg*2+3, i*2:i*2+2]) for i in range(pred_seq_len)]
    
    plt.show(block=False)  # Show the figure without blocking
    return fig, axes

def visualize_prediction(fig, axes, observations, ground_truth, prediction, epoch, train_predictions, train_ground_truth, metrics):
    def clear_axes(ax_list):
        if isinstance(ax_list, dict):
            for seg_axes in ax_list.values():
                for ax in seg_axes:
                    ax.clear()
                    ax.axis('off')
        else:
            for ax in ax_list:
                ax.clear()
                ax.axis('off')

    for ax_list in axes.values():
        clear_axes(ax_list)

    observations = observations.cpu().numpy().astype(np.float32)
    if not isinstance(ground_truth, dict):
        ground_truth = {'full_frame': ground_truth} 
    if not isinstance(train_ground_truth, dict):
        train_ground_truth = {'full_frame': train_ground_truth}
    ground_truth = {k: v.cpu().numpy().astype(np.float32) for k, v in ground_truth.items()}
    prediction = {k: v.cpu().numpy().astype(np.float32) for k, v in prediction.items()}
    
    # Handle both dictionary and tensor cases for train data
    if isinstance(train_predictions, dict):
        train_predictions = {k: v.cpu().numpy().astype(np.float32) for k, v in train_predictions.items()}
        train_ground_truth = {k: v.cpu().numpy().astype(np.float32) for k, v in train_ground_truth.items()}
    else:
        train_predictions = train_predictions.cpu().numpy().astype(np.float32)
        train_ground_truth = train_ground_truth.cpu().numpy().astype(np.float32)
    
    input_seq_len = observations.shape[1]
    pred_seq_len = list(prediction.values())[0].shape[1]
    num_segments = len(ground_truth)
    
    # Display input sequence
    for i, ax in enumerate(axes['input']):
        obs = normalize(prepare_image(observations[0, i]))
        ax.imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
        ax.set_title(f'Input (t-{input_seq_len-1-i})', fontsize=8)
    
    # Display ground truth and predictions for each segment side by side
    for seg_idx, (seg_name, seg_gt) in enumerate(ground_truth.items()):
        for i in range(pred_seq_len):
            ax = axes['segments'][seg_idx][i]
            gt_np = normalize(prepare_image(seg_gt[0, i]))
            pred_np = normalize(prepare_image(prediction[seg_name][0, i]))
            
            ax.imshow(np.hstack([gt_np, pred_np]), cmap='viridis' if gt_np.ndim == 2 else None)
            ax.set_title(f'{seg_name} (t+{i+1})\nGT | Pred', fontsize=8)
    
    # Display 3x3 grid of training predictions and ground truths
    if isinstance(train_predictions, dict):
        num_train_preds = min(9, list(train_predictions.values())[0].shape[0])
    else:
        num_train_preds = min(9, train_predictions.shape[0])
    
    for i in range(len(axes['train'])):
        ax = axes['train'][i]
        if i < num_train_preds:
            if isinstance(train_predictions, dict):
                pred_np = normalize(prepare_image(np.sum([v[i, 0] for v in train_predictions.values()], axis=0)))
                gt_np = normalize(prepare_image(np.sum([v[i, 0] for v in train_ground_truth.values()], axis=0)))
            else:
                pred_np = normalize(prepare_image(train_predictions[i, 0]))
                gt_np = normalize(prepare_image(train_ground_truth[i, 0]))
            
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