import torch
from omegaconf import DictConfig
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, create_data_loaders, move_batch_to_device
from utils.file_utils import find_model_path
from utils.config_utils import config_wrapper
from models import BasePredictiveModel, get_model_class
from pathlib import Path
import numpy as np
from utils.visualization_utils import setup_visualization, visualize_prediction
from utils.training_utils import analyze_predictions, load_model_state, compute_model_checksum
import matplotlib.pyplot as plt
import argparse
import os

from plotting_setup import setup_plotting
setup_plotting()

class ModelVisualizer:
    def __init__(self, cfg: DictConfig, model: BasePredictiveModel, train_loader, val_loader, device):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.fig = None
        self.axes = None
        self.setup_visualization()

    def setup_visualization(self):
        if self.cfg.training.create_plots:
            hold_out_batch = next(iter(self.val_loader))
            hold_out_batch = move_batch_to_device(hold_out_batch, self.device)
            hold_out_batch = {k: v[0:2] for k, v in hold_out_batch.items()}  # Take only first two samples
            self.hold_out_batch = hold_out_batch
            self.hold_out_obs = hold_out_batch['observations'][:2]
            self.hold_out_target = hold_out_batch['next_observations'][:2]
            seq_length = self.hold_out_obs.shape[1]
            num_predictions = self.hold_out_target.shape[1]
            self.fig, self.axes = setup_visualization(seq_length, num_predictions)
            plt.ion()  # Turn on interactive mode

    def visualize_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            # Select first two samples from batch for visualization
            vis_batch = {k: v[0:2] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_output = self.model(vis_batch)
            predictions = model_output['predictions']
            
            # Calculate statistics and metrics
            stats = analyze_predictions(predictions, vis_batch['next_observations'])
            metrics = {
                'Mean Pred': stats['pred_mean'],
                'Pred Std': stats['pred_std'],
                'Mean Target': stats['target_mean'],
                'Target Std': stats['target_std']
            }

            if hasattr(self.model, 'predict_done_probability'):
                done_probs = self.model.predict_done_probability(model_output['hazard'])
            else:
                done_probs = None

            # Visualize the predictions
            visualize_prediction(
                self.fig, 
                self.axes,
                vis_batch['observations'],
                vis_batch['next_observations'],
                predictions,
                0,  # epoch number (not relevant here)
                predictions,  # using the same predictions for the training grid
                vis_batch['next_observations'],
                metrics,
                done_probs.cpu().numpy() if done_probs is not None else None,
                vis_batch['dones'][:, -self.model.num_frames_to_predict:].cpu().numpy() if 'dones' in vis_batch else None
            )

            plt.draw()
            plt.pause(0.1)  # Small pause to update the plot

    def run_visualization(self, loader, save_dir=None):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i, batch in enumerate(loader):
            batch = move_batch_to_device(batch, self.device)
            self.visualize_batch(batch)
            
            if save_dir:
                self.fig.savefig(os.path.join(save_dir, f'batch_{i:04d}.png'))

            # Wait for user input
            response = input(f"\nBatch {i+1}/{len(loader)}\nPress Enter to continue, 's' to save, or 'q' to quit: ")
            if response.lower() == 'q':
                break
            elif response.lower() == 's':
                save_path = os.path.join(save_dir if save_dir else '.', f'batch_{i:04d}.png')
                self.fig.savefig(save_path)
                print(f"Saved figure to {save_path}")

        plt.ioff()
        plt.close('all')

@config_wrapper()
def main(cfg: DictConfig):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize model predictions across dataset")
    parser.add_argument('--model-type', type=str, required=True, help='Type of model to visualize')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations (optional)')
    parser.add_argument('--dataset', choices=['train', 'val'], default='val', help='Which dataset to visualize')
    args = parser.parse_args()

    # Set up device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = EnvironmentDataset(cfg)
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(full_dataset)
    print(f"Dataset loaded with {len(full_dataset)} samples")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset=full_dataset,
        batch_size=cfg.training.batch_size,
        val_size=cfg.training.val_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor,
        batches_per_epoch=cfg.training.batches_per_epoch
    )

    # Initialize model
    ModelClass = get_model_class(args.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {args.model_type}")
    print(f"Using model: {ModelClass.__name__}")

    model = ModelClass(
        obs_shape=obs_shape,
        action_dim=action_dim,
        ego_state_dim=ego_state_dim,
        cfg=cfg
    ).to(device)

    # Load model checkpoint
    checkpoint_path = find_model_path(cfg.project_dir, args.model_path)
    print(f"Looking for model in {checkpoint_path}")
    if checkpoint_path is None:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    print(f"Loading model from {checkpoint_path}")

    load_model_state(
        model_path=checkpoint_path,
        model=model,
        device=device,
        strict=True
    )
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    model_checksum = compute_model_checksum(model, include_names=False, verbose=False)
    print(f"Model loaded with checksum: {model_checksum}")

    # Create visualizer
    visualizer = ModelVisualizer(cfg, model, train_loader, val_loader, device)

    # Run visualization
    loader = train_loader if args.dataset == 'train' else val_loader
    print(f"\nVisualizing {args.dataset} dataset")
    print("Controls:")
    print("- Press Enter to advance to next batch")
    print("- Press 's' to save current visualization")
    print("- Press 'q' to quit")
    visualizer.run_visualization(loader, args.save_dir)

if __name__ == "__main__":
    main()