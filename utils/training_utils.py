import torch
import time
from omegaconf import DictConfig, OmegaConf

def init_wandb(cfg: DictConfig):
    if cfg.wandb.enabled:
        import wandb
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))
        print(f"Initialized wandb project: {cfg.wandb.project}")
        return wandb
    else:
        # Return a dummy object that does nothing when called
        class DummyWandB:
            def __getattr__(self, _):
                return lambda *args, **kwargs: None
        print("WandB is disabled")
        return DummyWandB()
    

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


def analyze_predictions(predictions, targets, calculate_diversity=False):
    """
    Analyze predictions for potential mean collapse and other statistics.
    
    Args:
    predictions (torch.Tensor): Model predictions
    targets (torch.Tensor): Ground truth targets
    calculate_diversity (bool): Flag indicating whether to calculate diversity
    
    Returns:
    dict: Dictionary containing various statistics
    """
    result = {}
    
    if calculate_diversity:
        pred_diversity = calculate_prediction_diversity(predictions)
        target_diversity = calculate_prediction_diversity(targets)
        result["prediction_diversity"] = pred_diversity
        result["target_diversity"] = target_diversity
    
    pred_mean = torch.mean(predictions).item()
    pred_std = torch.std(predictions).item()
    target_mean = torch.mean(targets).item()
    target_std = torch.std(targets).item()
    
    result["pred_mean"] = pred_mean
    result["pred_std"] = pred_std
    result["target_mean"] = target_mean
    result["target_std"] = target_std
    
    return result