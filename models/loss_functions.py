# loss_functions.py
import torch
from torch import nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.5, l1_weight=0.3, diversity_weight=0.1, 
                 latent_l1_weight=0.05, latent_l2_weight=0.05, temporal_decay=0.9):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.diversity_weight = diversity_weight
        self.latent_l1_weight = latent_l1_weight
        self.latent_l2_weight = latent_l2_weight
        self.temporal_decay = temporal_decay

    def compute_temporal_weights(self, seq_len, device):
        weights = torch.tensor([self.temporal_decay ** i for i in range(seq_len)], device=device)
        return weights / weights.sum()

    def diversity_loss(self, pred):
        batch_size, seq_len = pred.shape[:2]
        pred_flat = pred.view(batch_size, seq_len, -1)
        dist_matrix = torch.cdist(pred_flat, pred_flat)
        mean_dist = dist_matrix.sum() / (batch_size * (batch_size - 1))
        return -mean_dist

    def latent_l1_loss(self, latent):
        return torch.mean(torch.abs(latent))

    def latent_l2_loss(self, latent):
        return torch.mean(latent ** 2)

    def forward(self, pred, target, latent):
        device = pred.device
        batch_size, seq_len = pred.shape[:2]
        
        temporal_weights = self.compute_temporal_weights(seq_len, device=device)
        temporal_weights = temporal_weights.view(1, -1, 1, 1, 1)

        mse_loss = ((pred - target) ** 2).mean(dim=(2, 3, 4))
        weighted_mse_loss = (mse_loss * temporal_weights.squeeze()).sum(dim=1).mean()

        l1_loss = torch.abs(pred - target).mean(dim=(2, 3, 4))
        weighted_l1_loss = (l1_loss * temporal_weights.squeeze()).sum(dim=1).mean()

        diversity_loss = self.diversity_loss(pred)
        latent_l1_loss = self.latent_l1_loss(latent)
        latent_l2_loss = self.latent_l2_loss(latent)

        total_loss = (
            self.mse_weight * weighted_mse_loss +
            self.l1_weight * weighted_l1_loss +
            self.diversity_weight * diversity_loss +
            self.latent_l1_weight * latent_l1_loss +
            self.latent_l2_weight * latent_l2_loss
        )

        loss_components = {
            'mse': weighted_mse_loss.item(),
            'l1': weighted_l1_loss.item(),
            'diversity': diversity_loss.item(),
            'latent_l1': latent_l1_loss.item(),
            'latent_l2': latent_l2_loss.item(),
            'total': total_loss.item()
        }

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