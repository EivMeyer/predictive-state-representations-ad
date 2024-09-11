import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler

class NoScheduler(_LRScheduler):
    def __init__(self, optimizer):
        super(NoScheduler, self).__init__(optimizer)

    def step(self):
        pass

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


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