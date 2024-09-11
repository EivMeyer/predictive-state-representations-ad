import torch
from torch import nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.5, l1_weight=0.3, diversity_weight=0.1, 
                 latent_l1_weight=0.05, latent_l2_weight=0.05, temporal_decay=0.9,
                 perceptual_weight=0.1, num_scales=3):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.diversity_weight = diversity_weight
        self.latent_l1_weight = latent_l1_weight
        self.latent_l2_weight = latent_l2_weight
        self.temporal_decay = temporal_decay
        self.perceptual_weight = perceptual_weight
        self.num_scales = num_scales

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
        # Ensure x is 4D: [batch_size, channels, height, width]
        if x.dim() == 5:
            x = x.view(-1, *x.shape[2:])
        
        # Apply Sobel filter to each channel independently
        sobel_x = self.sobel_filter_x().to(x.device)
        sobel_y = self.sobel_filter_y().to(x.device)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
        
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    def sobel_filter_x(self):
        return torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def sobel_filter_y(self):
        return torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def forward(self, pred, target, latent):
        device = pred.device
        
        # Handle case where pred and target have an extra dimension (sequence length of 1)
        if pred.dim() == 5 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        batch_size, channels, height, width = pred.shape
        
        temporal_weights = self.compute_temporal_weights(1, device=device)  # Use 1 as seq_len since we've squeezed it
        temporal_weights = temporal_weights.view(1, 1, 1, 1)

        total_loss = 0
        loss_components = {}

        for scale in range(self.num_scales):
            if scale > 0:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)

            # MSE loss with temporal weighting
            mse_loss = ((pred - target) ** 2).mean()
            weighted_mse_loss = mse_loss * temporal_weights.squeeze()
            total_loss += self.mse_weight * weighted_mse_loss
            loss_components[f'mse_scale_{scale}'] = weighted_mse_loss.item()

            # L1 loss with temporal weighting
            l1_loss = torch.abs(pred - target).mean()
            weighted_l1_loss = l1_loss * temporal_weights.squeeze()
            total_loss += self.l1_weight * weighted_l1_loss
            loss_components[f'l1_scale_{scale}'] = weighted_l1_loss.item()

            # Perceptual loss (gradient-based) with temporal weighting
            pred_grad = self.gradient_magnitude(pred)
            target_grad = self.gradient_magnitude(target)
            perceptual_loss = ((pred_grad - target_grad) ** 2).mean()
            weighted_perceptual_loss = perceptual_loss * temporal_weights.squeeze()
            total_loss += self.perceptual_weight * weighted_perceptual_loss
            loss_components[f'perceptual_scale_{scale}'] = weighted_perceptual_loss.item()

        diversity_loss = self.diversity_loss(pred.unsqueeze(1))  # Add sequence dimension back for diversity loss
        total_loss += self.diversity_weight * diversity_loss
        loss_components['diversity'] = diversity_loss.item()

        latent_l1_loss = self.latent_l1_loss(latent)
        total_loss += self.latent_l1_weight * latent_l1_loss
        loss_components['latent_l1'] = latent_l1_loss.item()

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