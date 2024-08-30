import hydra
from omegaconf import DictConfig, OmegaConf
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders, move_batch_to_device
import torch
from models.base_predictive_model import BasePredictiveModel
from models.predictive_model_v8 import PredictiveModelV8
from loss_function import CombinedLoss
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
import numpy as np
from typing import Type
import matplotlib.pyplot as plt
import time
from utils.visualization_utils import setup_visualization, visualize_prediction
import torch.multiprocessing as mp
from utils.training_utils import AdaptiveLogger, analyze_predictions, init_wandb
from datetime import datetime


def get_model_class(model_type) -> Type[BasePredictiveModel]:
    model_classes = {
        "PredictiveModelV8": PredictiveModelV8,
    }
    return model_classes.get(model_type)


def train_model(
    model: BasePredictiveModel, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion: CombinedLoss, 
    epochs, scheduler, max_grad_norm, device, wandb, create_plots, stdout_logging, model_save_dir, save_interval, overwrite_checkpoints):
    logger = AdaptiveLogger()

    model.train()

    # Get two hold-out samples for visualization
    hold_out_batch = next(iter(val_loader))
    hold_out_batch = move_batch_to_device(hold_out_batch, device)
    hold_out_obs = hold_out_batch['observations'][:2]  # Get first two samples
    hold_out_target = hold_out_batch['next_observations'][:2]

    # Setup visualization
    if create_plots:
        seq_length = hold_out_obs.shape[1]
        num_predictions = hold_out_target.shape[1]
        fig, axes = setup_visualization(seq_length, num_predictions)

    for epoch in range(epochs):
        epoch_stats = {}

        start_time = time.time()
        total_iterations = 0
        
        for iteration, batch in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
            batch = move_batch_to_device(batch, device)

            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            targets = batch['next_observations']
            observations = batch['observations']
            ego_states = batch['ego_states']
            representations = model.encode(observations, ego_states)
            predictions = model.decode(representations)

            # Calculate loss
            loss, loss_components = criterion(predictions, targets, representations)
            
            # Backward pass
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update weights
            optimizer.step()

            # Automatically collect all loss components
            for key, value in loss_components.items():
                if key not in epoch_stats:
                    epoch_stats[key] = []
                epoch_stats[key].append(value)

            # Add overall loss to epoch_stats
            if "total_loss" not in epoch_stats:
                epoch_stats["total_loss"] = []
            epoch_stats["total_loss"].append(loss.item())
            
            # Collect statistics
            with torch.no_grad():
                stats = analyze_predictions(predictions, targets)
                for key, value in stats.items():
                    if key not in epoch_stats:
                        epoch_stats[key] = []
                    epoch_stats[key].append(value)
            
            # Intermediate logging
            if stdout_logging and logger.should_log(iteration, len(batch)):
                logger.log(epoch, iteration, loss, len(batch))
            
            # Store first 9 training predictions for visualization
            if create_plots and iteration == len(train_loader) - 1:
                train_predictions = predictions[:9].detach()
                train_ground_truth = targets[:9].detach()

            total_iterations += 1

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
        if create_plots:
            model.eval()
            with torch.no_grad():
                hold_out_pred = model(hold_out_batch)

            # Log images to wandb
            wandb.log({
                "hold_out_prediction": wandb.Image(fig),
            }, step=epoch)
        
            # Visualization
            metrics = {
                'MSE (train)': epoch_averages['mse']["mean"],
                # Add any other metrics you're tracking
            }
            visualize_prediction(fig, axes, hold_out_obs, hold_out_target, hold_out_pred, epoch, train_predictions, train_ground_truth, metrics)

        # Save model at specified interval
        if epoch % save_interval == 0:
            if overwrite_checkpoints:
                model_path = model_save_dir / "model_latest.pth"
            else:
                model_path = model_save_dir / f"model_epoch_{epoch+1}.pth"
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_path)
            
            print(f"Model saved to {model_path}")
            wandb.save(str(model_path))  # Log the saved model to wandb
        
        model.train()

    # Save the final model after training
    final_model_path = model_save_dir / "final_model.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    wandb.save(str(final_model_path))  # Log the final model to wandb

    # Finish wandb run
    wandb.finish()

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Set the multiprocessing start method to 'spawn'
    if __name__ == '__main__':
        mp.set_start_method('spawn', force=True)

    wandb = init_wandb(cfg)

    dataset_path = Path(cfg.project_dir) / "dataset"

    # Create a unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if wandb.run is not None:
        run_name = f"{timestamp}_{wandb.run.name}"
    else:
        run_name = timestamp

    run_dir = Path(cfg.project_dir) / "models" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the full dataset
    full_dataset = EnvironmentDataset(dataset_path, downsample_factor=cfg.training.downsample_factor)

    # Get data dimensions
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(full_dataset)
    
    # Create train and validation loaders
    batch_size = cfg.training.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_data_loaders(
        dataset=full_dataset, 
        batch_size=batch_size, 
        train_ratio=cfg.training.train_ratio, 
        num_workers=cfg.training.num_workers, 
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor
    )
    print(f"Training on {device}")
    print(f"Training minibatches: {len(train_loader)}")
    print(f"Validation minibatches: {len(val_loader)}")
    
    # Get the model class based on the config
    ModelClass = get_model_class(cfg.training.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {cfg.training.model_type}")
    
    model = ModelClass(
        obs_shape=obs_shape, 
        action_dim=action_dim, 
        ego_state_dim=ego_state_dim, 
        num_frames_to_predict=cfg.dataset.t_pred,
        hidden_dim=cfg.training.hidden_dim,
    )
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-4)

    criterion = CombinedLoss(
        mse_weight=cfg.training.loss.mse_weight,
        l1_weight=cfg.training.loss.l1_weight,
        diversity_weight=cfg.training.loss.diversity_weight,
        latent_l1_weight=cfg.training.loss.latent_l1_weight,
        latent_l2_weight=cfg.training.loss.latent_l2_weight,
        temporal_decay=cfg.training.loss.temporal_decay,
    )
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=cfg.training.epochs,
        scheduler=scheduler,
        max_grad_norm=cfg.training.max_grad_norm,
        device=device,
        wandb=wandb,
        create_plots=cfg.training.create_plots,
        stdout_logging=cfg.training.stdout_logging,
        model_save_dir=run_dir,
        save_interval=cfg.training.save_interval,
        overwrite_checkpoints=cfg.training.overwrite_checkpoints
    )

if __name__ == "__main__":
    main()