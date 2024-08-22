import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PredictiveModelV3(nn.Module):
    def __init__(self, obs_shape, hidden_dim=64, *args, **kwargs):
        super().__init__()
        
        obs_dim = np.prod(obs_shape)  # Total dimension of observation

        self.obs_shape = obs_shape
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Batch norm for input
        self.input_bn = nn.BatchNorm2d(3)  # Assuming 3 channel input (RGB)

        # Simple CNN for processing individual observations
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Flatten CNN output
        self.flatten = nn.Flatten()
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(32 * 8 * 8, hidden_dim, batch_first=True)
        
        # Batch norm after LSTM
        self.lstm_bn = nn.BatchNorm1d(hidden_dim)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, obs_dim),
            nn.Sigmoid()  # Ensure output is in [0, 1] range for image prediction
        )

    def forward(self, batch):
        observations = batch['observations']
        batch_size, seq_len, height, width, channels = observations.shape
        
        # Reshape and apply input batch norm
        obs_reshaped = observations.view(batch_size * seq_len, channels, height, width)
        obs_normalized = self.input_bn(obs_reshaped)
        
        # Process each observation through CNN
        cnn_output = self.cnn(obs_normalized)
        cnn_flat = self.flatten(cnn_output)
        cnn_seq = cnn_flat.view(batch_size, seq_len, -1)
        
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(cnn_seq)
        
        # Use the last LSTM output for prediction
        final_hidden = lstm_out[:, -1, :]
        
        # Apply batch norm to LSTM output
        final_hidden_normalized = self.lstm_bn(final_hidden)
        
        # Generate prediction
        prediction = self.output(final_hidden_normalized)
        prediction = prediction.view(batch_size, height, width, channels)
        
        return prediction

# Helper function to add noise to predictions
def add_prediction_noise(prediction, noise_scale=0.02):
    return prediction + noise_scale * torch.randn_like(prediction)

# Diversity loss
def diversity_loss(predictions):
    batch_size = predictions.size(0)
    flattened = predictions.view(batch_size, -1)
    pairwise_distances = torch.pdist(flattened)
    return -torch.mean(pairwise_distances)  # Negative because we want to maximize diversity