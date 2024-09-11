from omegaconf import DictConfig
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders, move_batch_to_device
from utils.file_utils import find_model_path
from utils.config_utils import config_wrapper
import torch
from torch import Tensor
from torch import nn, optim
from torch.utils.data import DataLoader
from models import BasePredictiveModel, get_model_class
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time
from utils.visualization_utils import setup_visualization, visualize_prediction
import torch.multiprocessing as mp
from utils.training_utils import analyze_predictions, init_wandb
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Any, Optional
import os
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau



class Trainer:
    def __init__(self, 
                cfg: DictConfig, 
                model: BasePredictiveModel, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                scheduler: optim.lr_scheduler._LRScheduler, 
                device: torch.device, 
                wandb: Any):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.wandb = wandb
        self.best_val_loss: float = float('inf')
        self.start_epoch: int = 0
        self.current_epoch: int = 0
        self.run_name: str = ""
        
        self.hold_out_batch: Optional[dict[str, torch.Tensor]] = None
        self.hold_out_obs: Optional[torch.Tensor] = None
        self.hold_out_target: Optional[torch.Tensor] = None
        self.train_predictions: Optional[torch.Tensor] = None
        self.train_ground_truth: Optional[torch.Tensor] = None
        self.val_stats: Dict[str, List[float]] = {}
        
        self.fig: Optional[Any] = None
        self.axes: Optional[Any] = None
        
        self.setup_visualization()

        self.use_amp = cfg.training.use_amp if hasattr(cfg.training, 'use_amp') else False
        self.scaler = GradScaler(init_scale=2**10, growth_factor=2**(1/4), backoff_factor=0.5, growth_interval=100)  if self.use_amp else None

        self.val_batch_size = self._calculate_val_batch_size()

    def _calculate_val_batch_size(self) -> int:
        if self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            current_memory = torch.cuda.memory_allocated(self.device)
            available_memory = total_memory - current_memory
            
            # Use approximately 20% of available memory per batch
            target_memory = available_memory * 0.2

            # Estimate memory per sample (this is a rough estimate and might need adjustment)
            sample_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            estimated_size_per_sample = sample_size * 2  # Factor of 2 for forward and backward pass

            val_batch_size = max(1, int(target_memory / estimated_size_per_sample))
            print(f"Calculated validation batch size: {val_batch_size}")
            return val_batch_size
        else:
            # For CPU, use a default batch size
            return 32

    def train_epoch(self) -> Dict[str, List[float]]:
        self.model.train()
        epoch_stats: Dict[str, List[float]] = {}
        
        # Get a single validation batch for quick validation
        val_batch = next(iter(self.val_loader))
        val_batch = move_batch_to_device(val_batch, self.device)
        val_batch = {k: v[0:1] for k, v in val_batch.items()}
        
        running_avg_loss = None
        alpha = 0.01  # Smoothing factor for running average

        total_batches = len(self.train_loader) if self.cfg.training.batches_per_epoch is None else self.cfg.training.batches_per_epoch

        pbar = tqdm(enumerate(self.train_loader), total=total_batches, leave=False, disable=not self.cfg.verbose)
        for iteration, full_batch in pbar:
            full_batch = move_batch_to_device(full_batch, self.device)
            batch_size = full_batch['observations'].shape[0]
            
            for _ in range(self.cfg.training.iterations_per_batch):
                for start_idx in range(0, batch_size, self.cfg.training.minibatch_size):
                    end_idx = min(start_idx + self.cfg.training.minibatch_size, batch_size)
                    batch = {k: v[start_idx:end_idx] for k, v in full_batch.items()}
                    
                    self.optimizer.zero_grad()
                    
                    if self.use_amp:
                        with autocast():
                            model_outputs = self.model.forward(batch)
                            loss, loss_components = self.model.compute_loss(batch, model_outputs)
                        
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        model_outputs = self.model.forward(batch)
                        loss, loss_components = self.model.compute_loss(batch, model_outputs)
                        
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
                        self.optimizer.step()
                    
                    predictions = model_outputs['predictions']
                    targets = batch['next_observations']
                    
                    self.update_stats(epoch_stats, loss_components, loss.item())
                    stats = analyze_predictions(predictions, targets)
                    self.update_stats(epoch_stats, stats)
                    
                    # Update running average loss
                    if running_avg_loss is None:
                        running_avg_loss = loss.item()
                    else:
                        running_avg_loss = alpha * loss.item() + (1 - alpha) * running_avg_loss
                    
                    # Compute validation error if configured
                    if self.cfg.training.track_val_loss:
                        with torch.no_grad():
                            val_outputs = self.model.forward(val_batch)
                            val_loss, _ = self.model.compute_loss(val_batch, val_outputs)
                    
                    if self.cfg.verbose:
                        # Get current learning rate and optimizer info
                        param_group = self.optimizer.param_groups[0]
                        current_lr = param_group['lr']
                        optimizer_type = self.optimizer.__class__.__name__
                        
                        # Prepare optimizer info string
                        optimizer_info = f"Optimizer: {optimizer_type}, LR: {current_lr:.6f}"
                        if 'betas' in param_group:
                            optimizer_info += f", Betas: {param_group['betas']}"
                        if 'momentum' in param_group:
                            optimizer_info += f", Momentum: {param_group['momentum']:.4f}"

                        # Update progress bar
                        pbar.set_description(
                            f"Train Loss: {loss.item():.6f}, Avg Loss: {running_avg_loss:.6f}, {optimizer_info}" +
                            (f", Val Loss: {val_loss.item():.6f}" if self.cfg.training.track_val_loss else "")
                        )
            
            if self.cfg.training.create_plots and iteration == len(self.train_loader) - 1:
                self.train_predictions = predictions[:9].detach()
                self.train_ground_truth = targets[:9].detach()
        
        return epoch_stats

    def validate(self) -> Dict[str, List[float]]:
        self.model.eval()
        val_stats: Dict[str, List[float]] = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = move_batch_to_device(batch, self.device)
                batch_size = batch['observations'].shape[0]
                
                for start_idx in range(0, batch_size, self.val_batch_size):
                    end_idx = min(start_idx + self.val_batch_size, batch_size)
                    minibatch = {k: v[start_idx:end_idx] for k, v in batch.items()}
                    
                    if self.use_amp:
                        with autocast():
                            model_outputs = self.model.forward(minibatch)
                            loss, loss_components = self.model.compute_loss(minibatch, model_outputs)
                    else:
                        model_outputs = self.model.forward(minibatch)
                        loss, loss_components = self.model.compute_loss(minibatch, model_outputs)
                    
                    predictions = model_outputs['predictions']
                    targets = minibatch['next_observations']
                    
                    self.update_stats(val_stats, loss_components, loss.item())
                    stats = analyze_predictions(predictions, targets)
                    self.update_stats(val_stats, stats)
        
        return val_stats

    def setup_visualization(self) -> None:
        if self.cfg.training.create_plots:
            hold_out_batch = next(iter(self.val_loader))
            hold_out_batch = move_batch_to_device(hold_out_batch, self.device)
            self.hold_out_batch = hold_out_batch
            self.hold_out_obs = hold_out_batch['observations'][:2]
            self.hold_out_target = hold_out_batch['next_observations'][:2]
            seq_length = self.hold_out_obs.shape[1]
            num_predictions = self.hold_out_target.shape[1]
            self.fig, self.axes = setup_visualization(seq_length, num_predictions)

    def update_stats(self, stats_dict: Dict[str, List[float]], new_stats: Dict[str, float], total_loss: Optional[float] = None) -> None:
        for key, value in new_stats.items():
            if key not in stats_dict:
                stats_dict[key] = []
            stats_dict[key].append(value)
        if total_loss is not None:
            if "total_loss" not in stats_dict:
                stats_dict["total_loss"] = []
            stats_dict["total_loss"].append(total_loss)

    def log_epoch_stats(self, train_stats: Dict[str, List[float]], val_stats: Dict[str, List[float]]) -> None:
        print(f"Epoch {self.current_epoch} completed:")
        epoch_averages: Dict[str, Any] = {}

        def process_values(values):
            if isinstance(values, list):
                if all(isinstance(v, torch.Tensor) for v in values):
                    # List of tensors
                    values_cpu = torch.stack(values).cpu().numpy()
                elif all(isinstance(v, (int, float)) for v in values):
                    # List of numbers
                    values_cpu = np.array(values)
                else:
                    raise TypeError(f"Unsupported type in list: {type(values[0])}")
            elif isinstance(values, torch.Tensor):
                values_cpu = values.cpu().numpy()
            else:
                raise TypeError(f"Unsupported type: {type(values)}")
            return values_cpu

        for key, values in train_stats.items():
            values_cpu =  process_values(values)
            mean_value = np.mean(values_cpu)
            std_value = np.std(values_cpu)
            epoch_averages[f"train/{key}"] = {"mean": mean_value, "std": std_value}
            print(f"  Mean train {key}: {mean_value:.4f} (±{std_value:.4f})")

        print("Validation results:")
        for key, values in val_stats.items():
            values_cpu =  process_values(values)
            mean_value = np.mean(values)
            epoch_averages[f"val/{key}"] = mean_value
            print(f"  Val {key}: {mean_value:.4f}")

        self.wandb.log(epoch_averages, step=self.current_epoch)

    def save_model(self, is_best: bool = False) -> None:
        save_dict = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': np.mean(self.val_stats['total_loss']),
        }

        if is_best:
            path = Path(self.cfg.project_dir) / "models" / self.run_name / "best_model.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_dict, path)
            print(f"New best model saved with validation loss: {save_dict['val_loss']:.4f}")
        elif self.current_epoch % self.cfg.training.save_interval == 0:
            if self.cfg.training.overwrite_checkpoints:
                path = Path(self.cfg.project_dir) / "models" / self.run_name / "model_latest.pth"
            else:
                path = Path(self.cfg.project_dir) / "models" / self.run_name / f"model_epoch_{self.current_epoch+1}.pth"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_dict, path)
            print(f"Model saved to {path}")
            self.wandb.save(str(path))

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            self.current_epoch = epoch
            start_time = time.time()

            train_stats = self.train_epoch()
            self.val_stats = self.validate()
            self.log_epoch_stats(train_stats, self.val_stats)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr}")
            self.wandb.log({"learning_rate": current_lr}, step=epoch)

            epoch_time = time.time() - start_time
            epoch_speed = len(self.train_loader) / epoch_time
            print(f"Epoch speed: {epoch_speed:.2f} iterations/second")
            self.wandb.log({"epoch_speed": epoch_speed}, step=epoch)

            if self.cfg.training.create_plots:
                self.visualize_predictions()

            val_loss = np.mean(self.val_stats['total_loss'])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(is_best=True)

            self.save_model()

        self.save_final_model()

    def visualize_predictions(self) -> None:
        self.model.eval()
        with torch.no_grad():
            hold_out_pred = self.model(self.hold_out_batch)['predictions']

        self.wandb.log({
            "hold_out_prediction": self.wandb.Image(self.fig),
        }, step=self.current_epoch)

        metrics = {
            'Loss (val)': np.mean(self.val_stats['total_loss']),
        }
        visualize_prediction(self.fig, self.axes, self.hold_out_obs, self.hold_out_target, hold_out_pred, 
                            self.current_epoch, self.train_predictions, self.train_ground_truth, metrics)

    def save_final_model(self) -> None:
        final_model_path = Path(self.cfg.project_dir) / "models" / self.run_name / "final_model.pth"
        torch.save({
            'epoch': self.cfg.training.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': np.mean(self.val_stats['total_loss']),
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
        self.wandb.save(str(final_model_path))

    def load_checkpoint(self, checkpoint_path: Path, load_scheduler: bool, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Recreate the optimizer with the current configuration
        self.optimizer = get_optimizer(self.model, self.cfg)

        # Load the optimizer state, ignoring missing or extra parameters
        optimizer_state = checkpoint['optimizer_state_dict']
        current_optimizer_state = self.optimizer.state_dict()

        # Only load state for parameters that exist in both states
        common_params = set(optimizer_state['param_groups'][0]['params']) & set(current_optimizer_state['param_groups'][0]['params'])
        
        for param in common_params:
            if param in optimizer_state['state']:
                current_optimizer_state['state'][param] = optimizer_state['state'][param]

        self.optimizer.load_state_dict(current_optimizer_state)

        # Override optimizer learning rate, betas and weight_decay:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.cfg.training.optimizer.learning_rate
            if 'betas' in param_group:
                param_group['betas'] = (self.cfg.training.optimizer.beta1, self.cfg.training.optimizer.beta2)
            param_group['weight_decay'] = self.cfg.training.optimizer.weight_decay

        # Recreate the scheduler with the current configuration
        self.scheduler = get_scheduler(self.optimizer, self.cfg)

        if load_scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except ValueError:
                print("Warning: Failed to load scheduler state. Starting with a fresh scheduler.")

        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {self.start_epoch}")


def get_optimizer(model, cfg):
    optimizer_type = cfg.training.optimizer.type
    lr = cfg.training.optimizer.learning_rate
    weight_decay = cfg.training.optimizer.weight_decay

    if optimizer_type == "Adam":
        betas = (cfg.training.optimizer.beta1, cfg.training.optimizer.beta2)
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "AdamW":
        betas = (cfg.training.optimizer.beta1, cfg.training.optimizer.beta2)
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "SGD":
        momentum = cfg.training.optimizer.momentum
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "RMSprop":
        alpha = cfg.training.optimizer.alpha
        momentum = cfg.training.optimizer.momentum
        return RMSprop(model.parameters(), lr=lr, alpha=alpha, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def get_scheduler(optimizer, cfg):
    scheduler_type = cfg.training.scheduler.type

    if scheduler_type == "StepLR":
        return StepLR(optimizer, step_size=cfg.training.scheduler.step_size, gamma=cfg.training.scheduler.gamma)
    elif scheduler_type == "ExponentialLR":
        return ExponentialLR(optimizer, gamma=cfg.training.scheduler.gamma)
    elif scheduler_type == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=cfg.training.scheduler.T_max, eta_min=cfg.training.scheduler.eta_min)
    elif scheduler_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=cfg.training.scheduler.factor, 
                                 patience=cfg.training.scheduler.patience, threshold=cfg.training.scheduler.threshold)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    

@config_wrapper()
def main(cfg: DictConfig) -> None:
    if __name__ == '__main__':
        mp.set_start_method('spawn', force=True)

    wandb = init_wandb(cfg)

    dataset_path = Path(cfg.project_dir) / "dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{wandb.run.name}_{cfg.training.model_type}" if cfg.wandb.enabled else f"{timestamp}_{cfg.training.model_type}"
    run_dir = Path(cfg.project_dir) / "models" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    full_dataset = EnvironmentDataset(dataset_path, downsample_factor=cfg.training.downsample_factor)
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(full_dataset)
    
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else cfg.device)
    train_loader, val_loader = create_data_loaders(
        dataset=full_dataset, 
        batch_size=cfg.training.batch_size, 
        val_size=cfg.training.val_size, 
        num_workers=cfg.training.num_workers, 
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor,
        batches_per_epoch=cfg.training.batches_per_epoch
    )
    print(f"Training on {device}")
    print(f"Total batches: {len(full_dataset)}")
    print(f"Train set: {len(train_loader)}")
    print(f"Validation set: {len(val_loader)}")
    
    ModelClass = get_model_class(cfg.training.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {cfg.training.model_type}")
    
    model: BasePredictiveModel = ModelClass(
        obs_shape=obs_shape, 
        action_dim=action_dim, 
        ego_state_dim=ego_state_dim,
        cfg=cfg
    ).to(device)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    trainer = Trainer(cfg, model, train_loader, val_loader, optimizer, scheduler, device, wandb)
    trainer.run_name = run_name

    if cfg.training.warmstart_model:
        if cfg.training.warmstart_model == "latest":
            checkpoint_path = max(
                (os.path.join(root, name) for root, _, files in os.walk("./output/models") for name in files if name == "model_latest.pth"),
                key=os.path.getmtime
            )
        else:
            checkpoint_path = find_model_path(cfg.project_dir, cfg.training.warmstart_model)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Model file not found: {cfg.training.warmstart_model}. "
                                    f"Searched in {cfg.project_dir} and its subdirectories.")
        print(f"Loading model from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path, cfg.training.warmstart_load_scheduler_state, device)
    
    trainer.train()

if __name__ == "__main__":
    main()