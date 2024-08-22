import torch
import torch.nn as nn
import numpy as np

class SimpleReconstructiveModel(nn.Module):
    def __init__(self, obs_shape, *args, **kwargs):
        super().__init__()
        
        obs_dim = np.prod(obs_shape)  # Total dimension of observation

        self.obs_shape = obs_shape
        self.obs_dim = obs_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, batch):
        observations = batch['observations']
        batch_size, seq_len, height, width, channels = observations.shape
        
        # Take only the last observation from the sequence
        last_obs = observations[:, -1, :, :, :]
        
        # Process the last observation
        x = last_obs.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        x = self.encoder(x)
        x = self.decoder(x)
        
        # Reshape to match the expected output shape
        prediction = x.permute(0, 2, 3, 1)  # Change back to (batch_size, height, width, channels)
        
        return prediction