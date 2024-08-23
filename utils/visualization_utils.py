from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np


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

def setup_visualization(seq_length):
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(30, 20))
    gs = GridSpec(8, seq_length + 3, figure=fig)
    
    axes = []
    
    # Input sequence for first sample
    for i in range(seq_length):
        axes.append(fig.add_subplot(gs[0:2, i]))
    
    # Ground truth and prediction for first sample
    axes.append(fig.add_subplot(gs[2:4, :seq_length//2]))
    axes.append(fig.add_subplot(gs[2:4, seq_length//2:seq_length]))
    
    # Input sequence for second sample
    for i in range(seq_length):
        axes.append(fig.add_subplot(gs[4:6, i]))
    
    # Ground truth and prediction for second sample
    axes.append(fig.add_subplot(gs[6:8, :seq_length//2]))
    axes.append(fig.add_subplot(gs[6:8, seq_length//2:seq_length]))
    
    # 3x3 grid for training predictions and ground truths
    for i in range(3):
        for j in range(3):
            # Prediction
            axes.append(fig.add_subplot(gs[i*2:(i+1)*2, seq_length + j]))
            # Ground truth
            axes.append(fig.add_subplot(gs[i*2+1:(i+1)*2+1, seq_length + j]))
    
    plt.show()
    return fig, axes

def visualize_prediction(fig, axes, observations, ground_truth, prediction, epoch, train_predictions, train_ground_truth, metrics):
    for ax in axes:
        ax.clear()
        ax.axis('off')
    
    seq_length = observations.shape[1]
    
    def plot_sample(start_idx, sample_num):
        # Display input sequence
        for i in range(seq_length):
            obs = normalize(prepare_image(observations[sample_num-1, i].cpu().numpy()))
            axes[start_idx + i].imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
            axes[start_idx + i].set_title(f'Input {sample_num} (t-{seq_length-1-i})', fontsize=10)
        
        # Display ground truth
        gt_np = normalize(prepare_image(ground_truth[sample_num-1, 0].cpu().numpy()))
        axes[start_idx + seq_length].imshow(gt_np, cmap='viridis' if gt_np.ndim == 2 else None)
        axes[start_idx + seq_length].set_title(f'Ground Truth {sample_num} (Hold-out)', fontsize=12, fontweight='bold')
        
        # Display prediction
        pred_np = normalize(prepare_image(prediction[sample_num-1].cpu().numpy()))
        axes[start_idx + seq_length + 1].imshow(pred_np, cmap='viridis' if pred_np.ndim == 2 else None)
        axes[start_idx + seq_length + 1].set_title(f'Prediction {sample_num} (Hold-out)', fontsize=12, fontweight='bold')
        
        # Add MSE for this prediction
        mse = np.mean((gt_np - pred_np) ** 2)
        axes[start_idx + seq_length + 1].text(0.5, -0.1, f'MSE: {mse:.4f}',
                                              horizontalalignment='center',
                                              transform=axes[start_idx + seq_length + 1].transAxes)
    
    # Plot first sample
    plot_sample(0, 1)
    
    # Plot second sample
    plot_sample(seq_length + 2, 2)
    
    # Display 3x3 grid of training predictions and ground truths
    num_train_preds = min(9, len(train_predictions))
    for i in range(9):
        ax_pred = axes[-18 + 2*i]
        ax_gt = axes[-17 + 2*i]
        
        if i < num_train_preds:
            pred_np = normalize(prepare_image(train_predictions[i].cpu().numpy()))
            gt_np = normalize(prepare_image(train_ground_truth[i].cpu().numpy()))
            
            ax_pred.imshow(pred_np, cmap='viridis' if pred_np.ndim == 2 else None)
            ax_pred.set_title(f'Train Pred {i+1}', fontsize=10)
            
            ax_gt.imshow(gt_np, cmap='viridis' if gt_np.ndim == 2 else None)
            ax_gt.set_title(f'Train GT {i+1}', fontsize=10)
            
            # Add MSE for this training prediction
            mse = np.mean((gt_np - pred_np) ** 2)
            ax_pred.text(0.5, -0.1, f'MSE: {mse:.4f}',
                         horizontalalignment='center',
                         transform=ax_pred.transAxes,
                         fontsize=8)
        else:
            ax_pred.imshow(np.zeros_like(pred_np), cmap='viridis')
            ax_pred.set_title('N/A', fontsize=10)
            ax_gt.imshow(np.zeros_like(pred_np), cmap='viridis')
            ax_gt.set_title('N/A', fontsize=10)
    
    # Add overall metrics to suptitle
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    fig.suptitle(f'Prediction Analysis - Epoch {epoch}\n\n{metrics_text}', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Pause to update the plot