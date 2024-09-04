import hydra
from omegaconf import DictConfig
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders, move_batch_to_device
from utils.file_utils import find_model_path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import BasePredictiveModel, get_model_class
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time
from utils.visualization_utils import setup_visualization, visualize_prediction
import torch.multiprocessing as mp
from utils.training_utils import AdaptiveLogger, analyze_predictions, init_wandb
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

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
        self.logger = AdaptiveLogger()
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

    def train_epoch(self) -> Dict[str, List[float]]:
        self.model.train()
        epoch_stats: Dict[str, List[float]] = {}
        
        for iteration, batch in tqdm(enumerate(self.train_loader), leave=False, total=len(self.train_loader)):
            batch = move_batch_to_device(batch, self.device)
            targets = batch['next_observations']
            
            self.optimizer.zero_grad()
            
            model_outputs = self.model.forward(batch)
            predictions = model_outputs['predictions']
            loss, loss_components = self.model.compute_loss(batch, model_outputs)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
            self.optimizer.step()

            self.update_stats(epoch_stats, loss_components, loss.item())
            stats = analyze_predictions(predictions, targets)
            self.update_stats(epoch_stats, stats)
            
            if self.cfg.training.stdout_logging and self.logger.should_log(iteration, len(batch)):
                self.logger.log(self.current_epoch, iteration, loss, len(batch))
            
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
                
                targets = batch['next_observations']

                model_outputs = self.model.forward(batch)
                predictions = model_outputs['predictions']
                loss, loss_components = self.model.compute_loss(batch, model_outputs)
                
                self.update_stats(val_stats, loss_components, loss.item())
                stats = analyze_predictions(predictions, targets)
                self.update_stats(val_stats, stats)
        
        return val_stats

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

        for key, values in train_stats.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            epoch_averages[f"train/{key}"] = {"mean": mean_value, "std": std_value}
            print(f"  Mean train {key}: {mean_value:.4f} (±{std_value:.4f})")

        print("Validation results:")
        for key, values in val_stats.items():
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
            torch.save(save_dict, path)
            print(f"New best model saved with validation loss: {save_dict['val_loss']:.4f}")
        elif self.current_epoch % self.cfg.training.save_interval == 0:
            if self.cfg.training.overwrite_checkpoints:
                path = Path(self.cfg.project_dir) / "models" / self.run_name / "model_latest.pth"
            else:
                path = Path(self.cfg.project_dir) / "models" / self.run_name / f"model_epoch_{self.current_epoch+1}.pth"
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
            'MSE (train)': np.mean(self.val_stats['mse']),
            'MSE (val)': np.mean(self.val_stats['mse']),
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

    def load_checkpoint(self, checkpoint_path: Path, device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {self.start_epoch}")

@hydra.main(version_base=None, config_path=".", config_name="config")
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
        train_ratio=cfg.training.train_ratio, 
        num_workers=cfg.training.num_workers, 
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor
    )
    print(f"Training on {device}")
    print(f"Training minibatches: {len(train_loader)}")
    print(f"Validation minibatches: {len(val_loader)}")
    
    ModelClass = get_model_class(cfg.training.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {cfg.training.model_type}")
    
    model: BasePredictiveModel = ModelClass(
        obs_shape=obs_shape, 
        action_dim=action_dim, 
        ego_state_dim=ego_state_dim,
        cfg=cfg
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-4)

    trainer = Trainer(cfg, model, train_loader, val_loader, optimizer, scheduler, device, wandb)
    trainer.run_name = run_name

    if cfg.training.warmstart_model:
        checkpoint_path = find_model_path(cfg.project_dir, cfg.training.warmstart_model)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Model file not found: {cfg.training.warmstart_model}. "
                                    f"Searched in {cfg.project_dir} and its subdirectories.")
        trainer.load_checkpoint(checkpoint_path, device)
    
    trainer.train()

if __name__ == "__main__":
    main()