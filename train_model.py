import hydra
from omegaconf import DictConfig, OmegaConf
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders
import torch
from models.predictive_model_v5 import PredictiveModelV5
from models.simple_reconstructive_model import SimpleReconstructiveModel
from models.single_step_predictive_model import SingleStepPredictiveModel
from loss_function import CombinedLoss
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import time
from utils.visualization_utils import setup_visualization, visualize_prediction
import torch.multiprocessing as mp


class AdaptiveLogger:
    def __init__(self, base_batch_size=32, base_log_interval=50):
        self.base_batch_size = base_batch_size
        self.base_log_interval = base_log_interval
        self.start_time = time.time()
        self.total_samples = 0
        self.last_log_time = self.start_time

    def should_log(self, iteration, batch_size):
        current_time = time.time()
        time_since_last_log = current_time - self.last_log_time
        
        # Adjust log interval based on batch size
        adjusted_interval = max(1, int(self.base_log_interval * (self.base_batch_size / batch_size)))
        
        # Log if enough iterations have passed or if enough time has passed (e.g., at least 10 seconds)
        if iteration % adjusted_interval == 0 or time_since_last_log >= 10:
            self.last_log_time = current_time
            return True
        return False

    def log(self, epoch, iteration, loss, batch_size):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.total_samples += batch_size

        speed_samples = self.total_samples / elapsed_time
        speed_batches = iteration / elapsed_time

        print(f"Epoch {epoch}, Iteration {iteration}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Speed: {speed_samples:.2f} samples/second ({speed_batches:.2f} batches/second)")
        print(f"  Allocated GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached GPU Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")



def get_model_class(model_type):
    model_classes = {
        "PredictiveModelV5": PredictiveModelV5,
        "SimpleReconstructiveModel": SimpleReconstructiveModel,
        "SingleStepPredictiveModel": SingleStepPredictiveModel
    }
    return model_classes.get(model_type)



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
    flattened = tensor.reshape(batch_size, -1)
    
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

def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, scheduler, max_grad_norm, device):
    logger = AdaptiveLogger(base_batch_size=32, base_log_interval=50)

    model.train()
    
    # Get two hold-out samples for visualization
    hold_out_batch = next(iter(val_loader))
    hold_out_batch = move_batch_to_device(hold_out_batch, device)
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

        start_time = time.time()
        total_iterations = 0
        
        for iteration, batch in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            
            targets = batch['next_observations'][:, 0]
            
            predictions = model(batch)
            loss, loss_components = criterion(predictions, targets)
            
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
            
            total_iterations += 1
            if logger.should_log(iteration, len(batch)):
                logger.log(epoch, iteration, loss, len(batch))
            
            # Store first 9 training predictions for visualization
            if iteration == len(train_loader) - 1:
                train_predictions = predictions[:9].detach()
                train_ground_truth = targets[:9].detach()

        # Calculate and print mean statistics for the epoch
        print(f"Epoch {epoch} completed:")
        epoch_averages = {}
        for key, values in epoch_stats.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            epoch_averages[key] = {"mean": mean_value, "std": std_value}
            print(f"  Mean {key}: {mean_value:.4f} (±{std_value:.4f})")

        # Log all metrics to wandb
        wandb.log(epoch_averages, step=epoch)

        # Step the scheduler
        scheduler.step()

        # Log the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr}")
        wandb.log({"learning_rate": current_lr}, step=epoch)

        # Calculate and log overall epoch speed
        epoch_time = time.time() - start_time
        epoch_speed = len(train_loader) / epoch_time
        print(f"Epoch speed: {epoch_speed:.2f} iterations/second")
        wandb.log({"epoch_speed": epoch_speed}, step=epoch)

        # Visualize prediction on hold-out sample at the end of each epoch
        model.eval()
        with torch.no_grad():
            hold_out_pred = model(hold_out_batch)

        # Log images to wandb
        wandb.log({
            "hold_out_prediction": wandb.Image(fig),
        }, step=epoch)
        
        # Visualization
        metrics = {
            'MSE (train)': epoch_averages['mse_loss']["mean"],
            'Diversity (train)': epoch_averages['prediction_diversity']["mean"],
            # Add any other metrics you're tracking
        }
        visualize_prediction(fig, axes, hold_out_obs, hold_out_target, hold_out_pred, epoch, train_predictions, train_ground_truth, metrics)
        
        model.train()

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final plot open

    # Finish wandb run
    wandb.finish()

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Set the multiprocessing start method to 'spawn'
    if __name__ == '__main__':
        mp.set_start_method('spawn', force=True)

    dataset_path = Path(cfg.project_dir) / "dataset"
    
    # Load the full dataset
    full_dataset = EnvironmentDataset(dataset_path, downsample_factor=cfg.training.downsample_factor)

    # Get data dimensions
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(full_dataset)
    
    # Create train and validation loaders
    batch_size = cfg.training.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_data_loaders(full_dataset, batch_size, cfg.training.train_ratio, cfg.training.num_workers, cfg.training.pin_memory)
    print(f"Training minibatches: {len(train_loader)}")
    print(f"Validation minibatches: {len(val_loader)}")
    
    # Get the model class based on the config
    ModelClass = get_model_class(cfg.training.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {cfg.training.model_type}")
    
    model = ModelClass(obs_shape=obs_shape, action_dim=action_dim, ego_state_dim=ego_state_dim)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

    criterion = CombinedLoss(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, device=device)

    wandb.init(project="PredictiveStateRepresentations-AD", config=OmegaConf.to_container(cfg, resolve=True))
    
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=cfg.training.epochs, scheduler=scheduler, max_grad_norm=cfg.training.max_grad_norm, device=device)

if __name__ == "__main__":
    main()