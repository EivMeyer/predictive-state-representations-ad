import torch
import torch.nn as nn
import numpy as np

class PredictiveModelV0(nn.Module):
    def __init__(self, obs_shape, action_dim, ego_state_dim, hidden_dim=64):
        super().__init__()
        
        obs_dim = np.prod(obs_shape)  # Total dimension of observation

        self.obs_shape = obs_shape
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ego_state_dim = ego_state_dim

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # Reducing pooling stages
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((32, 32))  # Larger feature maps
        )
        
        self.obs_flatten = nn.Flatten()
        
        # LSTM for processed observations
        self.obs_lstm = nn.LSTM(128 * 32 * 32, hidden_dim, batch_first=True)
        
        # LSTM for action sequences
        self.action_lstm = nn.LSTM(action_dim, hidden_dim, batch_first=True)
        
        # LSTM for ego state sequences
        self.ego_state_lstm = nn.LSTM(ego_state_dim, hidden_dim, batch_first=True)
        
        combined_dim = hidden_dim * 3  # Output from 3 LSTMs
        
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, obs_dim),
            nn.Sigmoid()  # Ensure output is in [0, 1] range for image prediction
        )

    def forward(self, batch):
        observations, actions, ego_states = batch['observations'], batch['actions'], batch['ego_states']
        batch_size, seq_len, height, width, channels = observations.shape
        
        assert actions.shape == (batch_size, seq_len, self.action_dim), f"Expected actions shape {(batch_size, seq_len, self.action_dim)}, got {actions.shape}"
        assert ego_states.shape == (batch_size, seq_len, self.ego_state_dim), f"Expected ego_states shape {(batch_size, seq_len, self.ego_state_dim)}, got {ego_states.shape}"
        
        # Process observations
        obs_reshaped = observations.permute(0, 1, 4, 2, 3).contiguous()
        obs_reshaped = obs_reshaped.view(-1, channels, height, width)

        obs_features = self.obs_encoder(obs_reshaped)
        obs_features = self.obs_flatten(obs_features)
        obs_features = obs_features.view(batch_size, seq_len, -1)
        
        # Process through LSTMs
        _, (obs_hidden, _) = self.obs_lstm(obs_features)
        _, (action_hidden, _) = self.action_lstm(actions)
        _, (ego_hidden, _) = self.ego_state_lstm(ego_states)
        
        # Combine LSTM outputs
        combined = torch.cat([obs_hidden[-1], action_hidden[-1], ego_hidden[-1]], dim=1)
        assert combined.shape == (batch_size, self.obs_lstm.hidden_size * 3), f"Expected combined shape {(batch_size, self.obs_lstm.hidden_size * 3)}, got {combined.shape}"
        
        # Make final prediction
        prediction = self.predictor(combined)
        assert prediction.shape == (batch_size, self.obs_dim), f"Expected prediction shape {(batch_size, self.obs_dim)}, got {prediction.shape}"
        
        prediction = prediction.view(batch_size, height, width, channels)
        assert prediction.shape == (batch_size, height, width, channels), f"Expected final prediction shape {(batch_size, height, width, channels)}, got {prediction.shape}"
        
        return prediction