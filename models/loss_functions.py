import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.5, l1_weight=0.3, diversity_weight=0.1, 
                 latent_l1_weight=0.05, latent_l2_weight=0.05, temporal_decay=0.9,
                 perceptual_weight=0.1, num_scales=3, use_sample_weights=True,
                 r_weight=1.0, g_weight=1.0, b_weight=1.0, warmup_iterations=1000, momentum=0.99):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.diversity_weight = diversity_weight
        self.latent_l1_weight = latent_l1_weight
        self.latent_l2_weight = latent_l2_weight
        self.temporal_decay = temporal_decay
        self.perceptual_weight = perceptual_weight
        self.num_scales = num_scales
        self.use_sample_weights = use_sample_weights
        self.r_weight = r_weight
        self.g_weight = g_weight
        self.b_weight = b_weight

        if self.use_sample_weights:
            # Setup for running statistics for sample weights
            self.warmup_iterations = warmup_iterations
            self.momentum = momentum
            self.register_buffer('iteration_count', torch.tensor(0))
            self.register_buffer('running_mean', None)
            self.register_buffer('running_std', None)

    def update_running_stats(self, batch_mean, batch_std):
        if self.running_mean is None:
            self.running_mean = batch_mean
            self.running_std = batch_std
        else:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_std = self.momentum * self.running_std + (1 - self.momentum) * batch_std

    def compute_sample_weights(self, observations, debug=False):
        batch_size, seq_len, channels, height, width = observations.shape
        
        # Compute channel-wise mean and std for each sample
        sample_means = observations.view(batch_size, seq_len, channels, -1).mean(dim=3)
        sample_stds = observations.view(batch_size, seq_len, channels, -1).std(dim=3)
        
        # Compute batch statistics
        batch_mean = sample_means.mean(dim=(0, 1))
        batch_std = sample_stds.mean(dim=(0, 1))
        
        # Update running statistics
        self.update_running_stats(batch_mean, batch_std)
        
        # Compute channel-wise deviations
        mean_deviation = torch.abs(sample_means - self.running_mean.unsqueeze(0).unsqueeze(0))
        std_deviation = torch.abs(sample_stds - self.running_std.unsqueeze(0).unsqueeze(0))
        
        # Combine deviations across channels and time
        total_deviation = (mean_deviation + std_deviation).sum(dim=(1, 2))
        
        # Normalize weights
        weights = total_deviation / total_deviation.sum()
        
        if debug:
            self.plot_observation_sequences(observations, weights, sample_means, sample_stds, batch_mean, batch_std)
        
        return weights
    
    def plot_observation_sequences(self, observations, weights, sample_means, sample_stds, batch_mean, batch_std):
        batch_size, seq_len, channels, height, width = observations.shape
        
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle("Observation Sequences, Weights, and Channel Statistics", fontsize=16)
        
        for i in range(batch_size):
            row = i // grid_size
            col = i % grid_size
            ax = axes[row, col]
            
            # Plot the middle frame of the sequence
            mid_frame = seq_len // 2
            img = observations[i, mid_frame].permute(1, 2, 0).cpu().numpy()
            
            if channels == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
            
            # Prepare channel statistics text
            stats_text = f"Weight: {weights[i]:.4f}\n"
            for c in range(channels):
                stats_text += f"Ch{c} - μ: {sample_means[i, mid_frame, c]:.2f} (β: {batch_mean[c]:.2f}), "
                stats_text += f"σ: {sample_stds[i, mid_frame, c]:.2f} (β: {batch_std[c]:.2f})\n"
            
            ax.set_title(stats_text, fontsize=8)
            ax.axis('off')
        
        # Remove any unused subplots
        for i in range(batch_size, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.show()

    def compute_temporal_weights(self, seq_len, device):
        weights = torch.tensor([self.temporal_decay ** i for i in range(seq_len)], device=device)
        return weights / weights.sum()

    def diversity_loss(self, pred):
        # Compute pairwise distance between all predictions. 
        # Detailed explanation:
        # - pred_flat reshapes the predictions to [batch_size, seq_len, -1]
        # - dist_matrix computes the pairwise distance between all predictions
        # - mean_dist computes the mean distance between all predictions
        # - return -mean_dist to encourage diversity
        # - Note: We use torch.cdist instead of torch.norm because it supports batched inputs

        batch_size, seq_len = pred.shape[:2]
        pred_flat = pred.view(batch_size, seq_len, -1)
        dist_matrix = torch.cdist(pred_flat, pred_flat)
        mean_dist = dist_matrix.sum() / (batch_size * (batch_size - 1))
        return -mean_dist

    def latent_l1_loss(self, latent):
        return torch.mean(torch.abs(latent))

    def latent_l2_loss(self, latent):
        return torch.mean(latent ** 2)

    def gradient_magnitude(self, x):
        shape = x.shape
        x = x.view(-1, *x.shape[2:])
        
        # Apply Sobel filter to each channel independently
        sobel_x = self.sobel_filter_x().to(x.device)
        sobel_y = self.sobel_filter_y().to(x.device)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
        
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Convert back to original shape
        return grad_magnitude.view(*shape)

    def sobel_filter_x(self):
        return torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def sobel_filter_y(self):
        return torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def forward(self, pred, target, latent):
        device = pred.device

        if pred.ndim == 3 and target.ndim == 3:
            # Expand dimensions to match image format (insert height and width dimensions)
            pred = pred.unsqueeze(-1).unsqueeze(-1)
            target = target.unsqueeze(-1).unsqueeze(-1)
        
        batch_size, seq_len, channels, height, width = pred.shape
        
        temporal_weights = self.compute_temporal_weights(seq_len, device=device)
        temporal_weights = temporal_weights.unsqueeze(0).repeat(batch_size, 1)

        if self.use_sample_weights and width > 1 and height > 1:
            sample_weights = self.compute_sample_weights(target).to(device)
        else:
            sample_weights = torch.ones(batch_size, device=device)

        total_loss = 0
        loss_components = {}

        for scale in range(self.num_scales):
            batch_size, seq_len, channels, height, width = pred.shape

            if scale > 0:
                pred = F.avg_pool2d(pred.flatten(end_dim=1), kernel_size=2, stride=2).view(batch_size, seq_len, channels, height // 2, width // 2)
                target = F.avg_pool2d(target.flatten(end_dim=1), kernel_size=2, stride=2).view(batch_size, seq_len, channels, height // 2, width // 2)

            channel_weights = torch.tensor([self.r_weight, self.g_weight, self.b_weight], device=device).view(1, 1, 3, 1, 1) if channels == 3 else 1

            if self.mse_weight > 0:
                # MSE loss with temporal and sample weighting, and RGB channel weighting
                mse_loss = ((pred - target) ** 2 * channel_weights).mean(dim=(2, 3, 4))
                weighted_mse_loss = (mse_loss * temporal_weights).sum(dim=-1) * sample_weights
                weighted_mse_loss = weighted_mse_loss.mean()
                total_loss += self.mse_weight * weighted_mse_loss
                loss_components[f'mse_scale_{scale}'] = weighted_mse_loss.item()

            if self.l1_weight > 0:
                # L1 loss with temporal and sample weighting, and RGB channel weighting
                l1_loss = (torch.abs(pred - target) * channel_weights).mean(dim=(2, 3, 4))
                weighted_l1_loss = (l1_loss * temporal_weights).sum(dim=-1) * sample_weights
                weighted_l1_loss = weighted_l1_loss.mean()
                total_loss += self.l1_weight * weighted_l1_loss
                loss_components[f'l1_scale_{scale}'] = weighted_l1_loss.item()

            if self.perceptual_weight > 0 and width > 1 and height > 1:
                # Perceptual loss (gradient-based) with temporal and sample weighting
                pred_grad = self.gradient_magnitude(pred)
                target_grad = self.gradient_magnitude(target)
                perceptual_loss = ((pred_grad - target_grad) ** 2).mean(dim=(2, 3, 4))
                weighted_perceptual_loss = (perceptual_loss * temporal_weights).sum(dim=-1) * sample_weights
                weighted_perceptual_loss = weighted_perceptual_loss.mean()
                total_loss += self.perceptual_weight * weighted_perceptual_loss
                loss_components[f'perceptual_scale_{scale}'] = weighted_perceptual_loss.item()

            if width == 1 or height == 1:
                break

        if self.diversity_weight > 0:
            diversity_loss = self.diversity_loss(pred.unsqueeze(1))
            total_loss += self.diversity_weight * diversity_loss
            loss_components['diversity'] = diversity_loss.item()

        if self.latent_l1_weight > 0:
            latent_l1_loss = self.latent_l1_loss(latent)
            total_loss += self.latent_l1_weight * latent_l1_loss
            loss_components['latent_l1'] = latent_l1_loss.item()

        if self.latent_l2_weight > 0:
            latent_l2_loss = self.latent_l2_loss(latent)
            total_loss += self.latent_l2_weight * latent_l2_loss
            loss_components['latent_l2'] = latent_l2_loss.item()

        loss_components['total'] = total_loss.item()

        return total_loss, loss_components
    
class VQVAELoss(nn.Module):
    def __init__(self, commitment_cost=0.25):
        super(VQVAELoss, self).__init__()
        self.commitment_cost = commitment_cost
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, pred, target, latent, vq_loss):
        # Compute MSE loss between predictions and targets
        mse_loss = self.mse_loss(pred, target)

        # Combine MSE loss with VQ-VAE loss
        total_loss = mse_loss + vq_loss

        loss_components = {
            'mse': mse_loss.item(),
            'vq_loss': vq_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_components