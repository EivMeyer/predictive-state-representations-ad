from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders
import torch
from models.predictive_model import PredictiveModel
from torch import nn, optim
from experiment_setup import load_config
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from matplotlib.gridspec import GridSpec

def collate_fn(batch, device):
    observations, actions, ego_states, next_observations, _, _ = zip(*batch)
    
    # Convert to tensors and move to GPU in one step
    obs = torch.tensor(np.array(observations), device=device)
    act = torch.tensor(np.array(actions), device=device)
    ego = torch.tensor(np.array(ego_states), device=device)
    next_obs = torch.tensor(np.array(next_observations), device=device)
    
    return obs, act, ego, next_obs

def create_data_loaders(dataset, batch_size, device, train_ratio=0.8):
    """
    Split the dataset into training and validation sets, then create DataLoaders.
    
    Args:
    dataset (Dataset): The full dataset
    batch_size (int): Batch size for the DataLoaders
    train_ratio (float): Ratio of data to use for training (default: 0.8)

    Returns:
    train_loader (DataLoader): DataLoader for the training set
    val_loader (DataLoader): DataLoader for the validation set
    """
    # Calculate the size of each split
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders with custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, device))

    return train_loader, val_loader

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
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, seq_length, figure=fig)
    
    axes = []
    for i in range(seq_length):
        axes.append(fig.add_subplot(gs[0, i]))
    axes.append(fig.add_subplot(gs[1:, :seq_length//2]))
    axes.append(fig.add_subplot(gs[1:, seq_length//2:]))
    
    plt.show()
    return fig, axes

def visualize_prediction(fig, axes, observations, ground_truth, prediction, epoch):
    for ax in axes:
        ax.clear()
        ax.axis('off')

    seq_length = observations.shape[1]

    # Display input sequence
    for i in range(seq_length):
        obs = normalize(prepare_image(observations[0, i].cpu().numpy()))
        axes[i].imshow(obs, cmap='viridis' if obs.ndim == 2 else None)
        axes[i].set_title(f'Input t-{seq_length-1-i}')

    # Display ground truth
    ground_truth_np = normalize(prepare_image(ground_truth.cpu().numpy()))
    axes[-2].imshow(ground_truth_np, cmap='viridis' if ground_truth_np.ndim == 2 else None)
    axes[-2].set_title('Ground Truth')

    # Display prediction
    prediction_np = normalize(prepare_image(prediction.cpu().numpy()))
    axes[-1].imshow(prediction_np, cmap='viridis' if prediction_np.ndim == 2 else None)
    axes[-1].set_title('Prediction')

    plt.suptitle(f'Epoch {epoch}')
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Pause to update the plot

def calculate_variance(tensor):
    # Calculate variance across all dimensions except the batch dimension
    return torch.var(tensor.view(tensor.size(0), -1), dim=1).mean()

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, scheduler):
    model.train()
    
    # Get a hold-out sample for visualization
    hold_out_obs, hold_out_actions, hold_out_ego_states, hold_out_next_obs = next(iter(val_loader))
    hold_out_target = hold_out_next_obs[:, 0].to(device)

    # Setup visualization
    seq_length = hold_out_obs.shape[1]
    fig, axes = setup_visualization(seq_length)

    for epoch in range(epochs):
        total_loss = 0
        total_pred_var = 0
        total_target_var = 0
        num_batches = 0
        
        for iteration, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            targets = batch['next_observations'][:, 0]
            
            predictions = model(batch)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pred_var += calculate_variance(predictions).item()
            total_target_var += calculate_variance(targets).item()
            num_batches += 1
            
            if iteration % 100 == 0:
                print(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        avg_pred_var = total_pred_var / num_batches
        avg_target_var = total_target_var / num_batches
        
        print(f"Epoch {epoch} completed:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Prediction Variance: {avg_pred_var:.4f}")
        print(f"  Average Target Variance: {avg_target_var:.4f}")
        print(f"  Variance Ratio (Pred/Target): {avg_pred_var/avg_target_var:.4f}")

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

        visualize_prediction(fig, axes, hold_out_obs, hold_out_target[0], hold_out_pred[0], epoch)


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
    
    model = PredictiveModel(obs_dim, action_dim, ego_state_dim)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=config["training"]["epochs"], device=device, scheduler=scheduler)

if __name__ == "__main__":
    main()