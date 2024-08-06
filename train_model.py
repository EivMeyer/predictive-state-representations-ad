from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders
import torch
from models.predictive_model_v0 import PredictiveModelV0
from models.predictive_model_v1 import PredictiveModelV1
from models.predictive_model_v2 import PredictiveModelV2
from models.simple_reconstructive_model import SimpleReconstructiveModel
from models.single_step_predictive_model import SingleStepPredictiveModel
from loss_function import CombinedLoss
from torch import nn, optim
from experiment_setup import load_config
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
from matplotlib.gridspec import GridSpec


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
    fig = plt.figure(figsize=(30, 20))  # Increased width to accommodate new grid
    gs = GridSpec(6, seq_length + 3, figure=fig)  # Added 3 columns for new grid
    
    axes = []
    # First sample
    for i in range(seq_length):
        axes.append(fig.add_subplot(gs[0, i]))
    axes.append(fig.add_subplot(gs[1:3, :seq_length//2]))
    axes.append(fig.add_subplot(gs[1:3, seq_length//2:seq_length]))
    
    # Second sample
    for i in range(seq_length):
        axes.append(fig.add_subplot(gs[3, i]))
    axes.append(fig.add_subplot(gs[4:, :seq_length//2]))
    axes.append(fig.add_subplot(gs[4:, seq_length//2:seq_length]))
    
    # 3x3 grid for training predictions
    for i in range(3):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i*2:(i+1)*2, seq_length + j]))
    
    plt.show()
    return fig, axes

def visualize_prediction(fig, axes, observations, ground_truth, prediction, epoch, train_predictions):
    for ax in axes:
        ax.clear()
        ax.axis('off')

    seq_length = observations.shape[1]

    # Display input sequence and results for first sample
    for i in range(seq_length):
        obs = normalize(prepare_image(observations[0, i].cpu().numpy()))
        axes[i].imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
        axes[i].set_title(f'Input 1 t-{seq_length-1-i}')

    ground_truth_np = normalize(prepare_image(ground_truth[0].cpu().numpy()))
    axes[seq_length].imshow(ground_truth_np, cmap='viridis' if ground_truth_np.ndim == 2 else None)
    axes[seq_length].set_title('Ground Truth 1')

    prediction_np = normalize(prepare_image(prediction[0].cpu().numpy()))
    axes[seq_length + 1].imshow(prediction_np, cmap='viridis' if prediction_np.ndim == 2 else None)
    axes[seq_length + 1].set_title('Prediction 1')

    # Display input sequence and results for second sample
    for i in range(seq_length):
        obs = normalize(prepare_image(observations[1, i].cpu().numpy()))
        axes[seq_length + 2 + i].imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
        axes[seq_length + 2 + i].set_title(f'Input 2 t-{seq_length-1-i}')

    ground_truth_np = normalize(prepare_image(ground_truth[1].cpu().numpy()))
    axes[-11].imshow(ground_truth_np, cmap='viridis' if ground_truth_np.ndim == 2 else None)
    axes[-11].set_title('Ground Truth 2')

    prediction_np = normalize(prepare_image(prediction[1].cpu().numpy()))
    axes[-10].imshow(prediction_np, cmap='viridis' if prediction_np.ndim == 2 else None)
    axes[-10].set_title('Prediction 2')

    # Display 3x3 grid of training predictions
    num_train_preds = min(9, len(train_predictions))
    for i in range(9):
        if i < num_train_preds:
            pred_np = normalize(prepare_image(train_predictions[i].cpu().numpy()))
            axes[-9 + i].imshow(pred_np, cmap='viridis' if pred_np.ndim == 2 else None)
            axes[-9 + i].set_title(f'Train Pred {i+1}')
        else:
            axes[-9 + i].imshow(np.zeros_like(pred_np), cmap='viridis')
            axes[-9 + i].set_title(f'N/A')

    plt.suptitle(f'Epoch {epoch}')
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Pause to update the plot


def calculate_prediction_diversity(tensor):
    """
    Calculate the average pairwise difference between predictions in a batch.
    A value close to 0 indicates potential mean collapse.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (batch_size, ...)
    
    Returns:
    float: Average pairwise difference
    """
    # Reshape tensor to (batch_size, -1)
    batch_size = tensor.size(0)
    flattened = tensor.view(batch_size, -1)
    
    # Calculate pairwise differences
    diff_matrix = torch.cdist(flattened, flattened, p=2)
    
    # Calculate mean of upper triangle (excluding diagonal)
    diversity = diff_matrix.triu(diagonal=1).sum() / (batch_size * (batch_size - 1) / 2)
    
    return diversity.item()


def analyze_predictions(predictions, targets):
    """
    Analyze predictions for potential mean collapse and other statistics.
    
    Args:
    predictions (torch.Tensor): Model predictions
    targets (torch.Tensor): Ground truth targets
    
    Returns:
    dict: Dictionary containing various statistics
    """
    pred_diversity = calculate_prediction_diversity(predictions)
    target_diversity = calculate_prediction_diversity(targets)
    
    pred_mean = torch.mean(predictions).item()
    pred_std = torch.std(predictions).item()
    target_mean = torch.mean(targets).item()
    target_std = torch.std(targets).item()
    
    return {
        "prediction_diversity": pred_diversity,
        "target_diversity": target_diversity,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "target_mean": target_mean,
        "target_std": target_std
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, scheduler):
    model.train()
    
    # Get two hold-out samples for visualization
    hold_out_batch = next(iter(val_loader))
    hold_out_obs = hold_out_batch['observations'][:2]  # Get first two samples
    hold_out_target = hold_out_batch['next_observations'][:2]

    # Setup visualization
    seq_length = hold_out_obs.shape[1]
    fig, axes = setup_visualization(seq_length)

    for epoch in range(epochs):
        epoch_stats = {
            "total_loss": [],
            "mse_loss": [],
            "ssim_loss": [],
            "perceptual_loss": [],
            "diversity_loss": [],
            "prediction_diversity": [],
            "target_diversity": [],
            "pred_mean": [],
            "pred_std": [],
            "target_mean": [],
            "target_std": []
        }
        
        for iteration, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            targets = batch['next_observations'][:, 0]
            
            predictions = model(batch)
            loss, loss_components = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()

            # Collect loss statistics
            epoch_stats["total_loss"].append(loss.item())
            epoch_stats["mse_loss"].append(loss_components['mse'])
            epoch_stats["ssim_loss"].append(loss_components['ssim'])
            epoch_stats["perceptual_loss"].append(loss_components['perceptual'])
            epoch_stats["diversity_loss"].append(loss_components['diversity'])
            
            # Collect statistics
            with torch.no_grad():
                stats = analyze_predictions(predictions, targets)
                for key, value in stats.items():
                    epoch_stats[key].append(value)
            
            if (iteration + 1) % 100 == 0:
                print(f"Epoch {epoch}, Iteration {iteration + 1}")
                print(f"  Loss: {loss.item():.4f}")
            
            # Store first 9 training predictions for visualization
            if iteration == 0:
                train_predictions = predictions[:9].detach()

        # Calculate and print mean statistics for the epoch
        print(f"Epoch {epoch} completed:")
        for key, values in epoch_stats.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"  Mean {key}: {mean_value:.4f} (±{std_value:.4f})")

        # Step the scheduler
        scheduler.step()

        # Optional: Log the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr}")

        # Visualize prediction on hold-out sample at the end of each epoch
        model.eval()
        with torch.no_grad():
            hold_out_pred = model(hold_out_batch)
        model.train()

        visualize_prediction(fig, axes, hold_out_obs, hold_out_target, hold_out_pred, epoch, train_predictions)


    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final plot open


def main():
    config = load_config()
    dataset_path = Path(config["project_dir"]) / "dataset"
    
    # Load the full dataset
    full_dataset = EnvironmentDataset(dataset_path)

    # Get data dimensions
    obs_dim, action_dim, ego_state_dim = get_data_dimensions(full_dataset)
    
    # Create train and validation loaders
    batch_size = config["training"]["batch_size"]  # You can adjust this based on your GPU memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_data_loaders(full_dataset, batch_size, device)
    print(f"Training samples: {len(train_loader)}")
    print(f"Validation samples: {len(val_loader)}")
    
    model = PredictiveModelV2(obs_dim, action_dim, ego_state_dim)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) # CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    criterion = CombinedLoss(alpha=1.0, beta=0.0, gamma=0.0, delta=0.01, device=device)
    
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=config["training"]["epochs"], device=device, scheduler=scheduler)

if __name__ == "__main__":
    main()