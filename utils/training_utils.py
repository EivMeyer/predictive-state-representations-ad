import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from pathlib import Path
from torch import nn
import torch
from typing import Any, Dict, Optional
from torch.optim import Optimizer
import logging
import torch
import hashlib
from collections import OrderedDict
import numpy as np
from typing import Union, Dict, Optional
import json
import math

class NoScheduler(_LRScheduler):
    def __init__(self, optimizer):
        super(NoScheduler, self).__init__(optimizer)

    def step(self):
        pass

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def get_linear_warmup_cosine_decay_scheduler(optimizer, total_steps, warmup_steps=10000):
    """
    Returns a LambdaLR scheduler with linear warmup and cosine decay.
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to apply the scheduler to.
        warmup_steps (int): Number of steps for linear warmup.
        total_steps (int): Total number of training steps.
    Returns:
        torch.optim.lr_scheduler.LambdaLR: The scheduler.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < total_steps:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            # After total_steps, keep the learning rate at 0
            return 0.0
    return LambdaLR(optimizer, lr_lambda)

def init_wandb(cfg: DictConfig, project_postfix: str = ""):
    if cfg.wandb.enabled:
        project_postfix = '-' + project_postfix if project_postfix else ""
        import wandb
        wandb.init(project=cfg.wandb.project + "-" + cfg.environment + project_postfix, config=OmegaConf.to_container(cfg, resolve=True), mode="offline" if cfg.wandb.offline else "online")
        run_postfix = cfg.wandb.run if cfg.wandb.run else ""
        if run_postfix != "":
            wandb.run.name = wandb.run.name + " " + run_postfix
            wandb.run.save()
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

def load_model_state(model_path: Path, model: nn.Module, device: torch.device, strict: bool = False) -> Dict[str, Any]:
    """
    Helper function to load a model's state from a checkpoint.
    
    Args:
        model_path: Path to the checkpoint file
        model: Model instance to load state into
        device: Device to load the state to
        strict: Whether to strictly enforce state_dict key matching
        
    Returns:
        Dict containing checkpoint data
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_save_state' in checkpoint:
        # New format with nested states
        model.load_save_state(checkpoint['model_save_state'], strict=strict)
        return checkpoint
    elif 'model_state_dict' in checkpoint:
        # Legacy format
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        return checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format in {model_path}")


def load_optimizer_state(
    optimizer: Optimizer,
    model: torch.nn.Module,
    checkpoint_state: Dict[str, Any],
    hyperparameters: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> bool:
    """
    Robustly load optimizer state, handling partial matches and different parameter shapes.
    
    Args:
        optimizer: The current optimizer instance
        model: The model associated with the optimizer
        checkpoint_state: The loaded checkpoint state dictionary
        hyperparameters: Optional dictionary of hyperparameters to override loaded values
        verbose: Whether to print detailed logging information
    
    Returns:
        bool: True if at least some state was loaded successfully, False otherwise
    
    Example:
        checkpoint = torch.load('checkpoint.pth')
        success = load_optimizer_state(
            optimizer,
            model,
            checkpoint['optimizer_state_dict'],
            hyperparameters={
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay,
                'betas': (config.beta1, config.beta2)
            }
        )
    """
    def log_msg(msg: str):
        if verbose:
            logging.info(msg)
        
    if not isinstance(checkpoint_state, dict):
        log_msg("Checkpoint state must be a dictionary")
        return False

    try:
        # Create mapping of parameter IDs to names for the current model
        param_id_to_name = {id(p): name for name, p in model.named_parameters()}
        current_optim_state = optimizer.state_dict()
        
        # Initialize containers for the new optimizer state
        new_state = {'state': {}, 'param_groups': []}
        total_params = 0
        matched_params = 0
        
        # Process each parameter group
        for checkpoint_group, current_group in zip(checkpoint_state['param_groups'], 
                                                 current_optim_state['param_groups']):
            # Copy all group options except params
            new_group = {k: v for k, v in current_group.items() if k != 'params'}
            new_params = []
            params_to_skip = []
            
            # Check each parameter in the group
            for checkpoint_param_id, current_param in zip(checkpoint_group['params'], 
                                                        current_group['params']):
                total_params += 1
                current_param_name = param_id_to_name.get(current_param)
                if current_param_name is None:
                    continue
                
                # Get parameter shapes
                current_shape = model.state_dict()[current_param_name].shape
                checkpoint_shape = None
                
                # Find shape in checkpoint state
                if checkpoint_param_id in checkpoint_state['state']:
                    for v in checkpoint_state['state'][checkpoint_param_id].values():
                        if isinstance(v, torch.Tensor):
                            checkpoint_shape = v.shape
                            break
                
                if checkpoint_shape == current_shape:
                    # Shapes match, copy the state
                    matched_params += 1
                    new_params.append(current_param)
                    if checkpoint_param_id in checkpoint_state['state']:
                        new_state['state'][current_param] = {
                            k: v.clone() if isinstance(v, torch.Tensor) else v
                            for k, v in checkpoint_state['state'][checkpoint_param_id].items()
                        }
                else:
                    params_to_skip.append(current_param_name)
            
            if params_to_skip:
                log_msg(f"Skipping optimizer state for parameters due to shape mismatch: {', '.join(params_to_skip)}")
            
            new_group['params'] = new_params
            new_state['param_groups'].append(new_group)
        
        # Load the partially matched state
        optimizer.load_state_dict(new_state)
        match_percentage = (matched_params / total_params) * 100 if total_params > 0 else 0
        log_msg(f"Loaded optimizer state for {matched_params}/{total_params} parameters ({match_percentage:.1f}%)")
        
        # Override hyperparameters if provided
        if hyperparameters:
            for param_group in optimizer.param_groups:
                for key, value in hyperparameters.items():
                    if key in param_group:
                        param_group[key] = value
            log_msg("Applied hyperparameter overrides")
        
        return matched_params > 0

    except Exception as e:
        log_msg(f"Error loading optimizer state: {str(e)}")
        return False


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    map_location: str = 'cpu',
    strict: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Load a checkpoint with robust handling of model and optimizer states.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: Optional scheduler to load state into
        hyperparameters: Optional dictionary of optimizer hyperparameters to override
        map_location: Device to map tensors to
        strict: Whether to strictly enforce model state loading
        verbose: Whether to print detailed logging information
    
    Returns:
        Dict containing loading results and metadata
        
    Example:
        results = load_checkpoint(
            Path('checkpoint.pth'),
            model,
            optimizer,
            scheduler,
            hyperparameters={
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay,
                'betas': (config.beta1, config.beta2)
            },
            verbose=True
        )
        start_epoch = results['epoch'] + 1
    """
    def log_msg(msg: str):
        if verbose:
            logging.info(msg)
    
    results = {
        'success': False,
        'epoch': 0,
        'model_loaded': False,
        'optimizer_loaded': False,
        'scheduler_loaded': False
    }
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            try:
                if strict:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    results['model_loaded'] = True
                else:
                    # Load only matching parameters
                    model_state = model.state_dict()
                    matched_state_dict = {}
                    for name, param in checkpoint['model_state_dict'].items():
                        if name in model_state and param.shape == model_state[name].shape:
                            matched_state_dict[name] = param
                        else:
                            log_msg(f"Skipping model parameter {name}: shape mismatch or not in model")
                    
                    model.load_state_dict(matched_state_dict, strict=False)
                    results['model_loaded'] = True
                log_msg("Model state loaded successfully")
            except Exception as e:
                log_msg(f"Error loading model state: {str(e)}")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            results['optimizer_loaded'] = load_optimizer_state(
                optimizer, 
                model, 
                checkpoint['optimizer_state_dict'],
                hyperparameters,
                verbose
            )
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                results['scheduler_loaded'] = True
                log_msg("Scheduler state loaded successfully")
            except Exception as e:
                log_msg(f"Error loading scheduler state: {str(e)}")
        
        # Get training metadata
        results['epoch'] = checkpoint.get('epoch', 0)
        results['success'] = True
        
        return results
        
    except Exception as e:
        log_msg(f"Error loading checkpoint: {str(e)}")
        return results
    

def compute_model_checksum(model: Union[torch.nn.Module, Dict], include_names: bool = True, verbose: bool = False) -> str:
    """
    Compute a checksum for a PyTorch model's state.
    
    Args:
        model: Either a PyTorch model or a state dict
        include_names: Whether to include parameter names in the hash
        verbose: Whether to print detailed parameter information
    
    Returns:
        str: Hexadecimal checksum of the model state
    """
    if isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
    else:
        state_dict = model

    # Initialize hasher
    hasher = hashlib.sha256()
    
    # Sort parameters by name for consistency
    sorted_items = sorted(state_dict.items())
    
    total_params = 0
    param_stats = OrderedDict()
    
    for name, param in sorted_items:
        if isinstance(param, torch.Tensor):
            # Convert tensor to numpy and flatten
            param_np = param.detach().cpu().numpy().flatten()
            total_params += param_np.size
            
            # Compute statistics for this parameter
            param_stats[name] = {
                'shape': list(param.shape),
                'mean': float(np.mean(param_np)),
                'std': float(np.std(param_np)),
                'min': float(np.min(param_np)),
                'max': float(np.max(param_np)),
                'num_params': param_np.size
            }
            
            # Update hash with parameter values
            hasher.update(param_np.tobytes())
            
            # Optionally include parameter name in hash
            if include_names:
                hasher.update(name.encode())
    
    if verbose:
        print("\nModel Parameter Statistics:")
        print(f"Total parameters: {total_params:,}")
        for name, stats in param_stats.items():
            print(f"\n{name}:")
            print(f"  Shape: {stats['shape']}")
            print(f"  Parameters: {stats['num_params']:,}")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    checksum = hasher.hexdigest()
    
    return checksum

def verify_model_compatibility(model1: torch.nn.Module, model2: torch.nn.Module, 
                             verbose: bool = False) -> bool:
    """
    Verify that two models have compatible architectures by comparing their state dict keys.
    
    Args:
        model1: First PyTorch model
        model2: Second PyTorch model
        verbose: Whether to print detailed information about differences
    
    Returns:
        bool: True if models are compatible, False otherwise
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    missing_keys = keys1 - keys2
    unexpected_keys = keys2 - keys1
    
    if verbose:
        if missing_keys:
            print("\nMissing keys in second model:")
            for key in sorted(missing_keys):
                print(f"  {key}")
        
        if unexpected_keys:
            print("\nUnexpected keys in second model:")
            for key in sorted(unexpected_keys):
                print(f"  {key}")
        
        common_keys = keys1 & keys2
        print(f"\nCommon parameters: {len(common_keys)}")
        
        shape_mismatches = []
        for key in common_keys:
            shape1 = tuple(state_dict1[key].shape)
            shape2 = tuple(state_dict2[key].shape)
            if shape1 != shape2:
                shape_mismatches.append((key, shape1, shape2))
        
        if shape_mismatches:
            print("\nShape mismatches:")
            for key, shape1, shape2 in shape_mismatches:
                print(f"  {key}: {shape1} vs {shape2}")
    
    return len(missing_keys) == 0 and len(unexpected_keys) == 0

def save_model_metadata(model: torch.nn.Module, filepath: str, 
                       extra_info: Optional[Dict] = None) -> None:
    """
    Save model metadata including checksum and parameter statistics.
    
    Args:
        model: PyTorch model
        filepath: Where to save the metadata JSON
        extra_info: Additional information to include in metadata
    """
    state_dict = model.state_dict()
    
    metadata = {
        'checksum': compute_model_checksum(model),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'parameter_shapes': {name: list(param.shape) 
                           for name, param in state_dict.items()},
        'parameter_stats': {}
    }
    
    # Compute statistics for each parameter
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            param_np = param.detach().cpu().numpy()
            metadata['parameter_stats'][name] = {
                'mean': float(np.mean(param_np)),
                'std': float(np.std(param_np)),
                'min': float(np.min(param_np)),
                'max': float(np.max(param_np))
            }
    
    if extra_info:
        metadata.update(extra_info)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def count_parameters(model):
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_by_group(model):
    """
    Count parameters grouped by model component prefixes.
    
    Args:
        model (torch.nn.Module): PyTorch model
    
    Returns:
        dict: Parameter counts by group prefix
    """
    groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        group = name.split('.')[0]
        params = param.numel()
        
        if group not in groups:
            groups[group] = 0
        groups[group] += params
        
    return groups

def print_parameter_summary(model):
    """Print parameter counts by group and total."""
    groups = count_parameters_by_group(model)
    
    print("\nParameters by group:")
    print("-" * 50)
    total = 0
    for group, count in sorted(groups.items()):
        print(f"{group:30} {count:,}")
        total += count
    
    print("-" * 50)
    print(f"{'Total':30} {total:,}")
    
    return total  # Return total count for convenience


def compute_rl_checksums(
    rl_model,
    verbose: bool = False
) -> Dict[str, str]:
    """
    Compute checksums for RL and/or SRL models without modifying them.
    
    Args:
        rl_model: The RL model (optional)  
        srl_model: The SRL model (optional)
        verbose: Whether to print detailed parameter info
        
    Returns:
        Dict containing checksums for each component
    """
    checksums = {}
    
    if rl_model is not None:
        policy_checksum = compute_model_checksum(rl_model.policy, verbose=verbose)
        checksums['rl_policy'] = policy_checksum
        
        if hasattr(rl_model.policy, 'representation_model'):
            # Special case for end-to-end training
            checksums['rl_representation'] = compute_model_checksum(
                rl_model.policy.representation_model, 
                verbose=verbose
            )
    
    if verbose:
        print("\nChecksums:")
        for name, checksum in checksums.items():
            print(f"{name}: {checksum}")
            
    return checksums