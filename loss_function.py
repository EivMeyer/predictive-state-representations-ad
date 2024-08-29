import torch
from torch import nn

class CombinedLoss(torch.nn.Module):
    def __init__(self, mse_weight=0.5, l1_weight=0.3, diversity_weight=0.2, temporal_decay=0.9):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.diversity_weight = diversity_weight
        self.temporal_decay = temporal_decay

    def compute_temporal_weights(self, seq_len, device):
        weights = torch.tensor([self.temporal_decay ** i for i in range(seq_len)], device=device)
        return weights / weights.sum()  # Normalize so they sum to 1

    def diversity_loss(self, pred):
        batch_size, seq_len = pred.shape[:2]
        pred_flat = pred.view(batch_size, seq_len, -1)
        dist_matrix = torch.cdist(pred_flat, pred_flat)
        mean_dist = dist_matrix.sum() / (batch_size * (batch_size - 1))
        return -mean_dist

    def forward(self, pred, target):
        device = pred.device
        batch_size, seq_len = pred.shape[:2]
        
        # Compute temporal weights
        temporal_weights = self.compute_temporal_weights(seq_len, device=device)
        temporal_weights = temporal_weights.view(1, -1, 1, 1, 1)  # Shape: (1, seq_len, 1, 1, 1)

        # Compute MSE loss
        mse_loss = ((pred - target) ** 2).mean(dim=(2, 3, 4))  # Shape: (batch_size, seq_len)
        weighted_mse_loss = (mse_loss * temporal_weights.squeeze()).sum(dim=1).mean()

        # Compute L1 loss
        l1_loss = torch.abs(pred - target).mean(dim=(2, 3, 4))  # Shape: (batch_size, seq_len)
        weighted_l1_loss = (l1_loss * temporal_weights.squeeze()).sum(dim=1).mean()

        # Compute diversity loss
        diversity_loss = self.diversity_loss(pred)

        # Combine losses
        total_loss = (
            self.mse_weight * weighted_mse_loss +
            self.l1_weight * weighted_l1_loss +
            self.diversity_weight * diversity_loss
        )

        loss_components = {
            'mse': weighted_mse_loss.item(),
            'l1': weighted_l1_loss.item(),
            'diversity': diversity_loss.item()
        }

        return total_loss, loss_components