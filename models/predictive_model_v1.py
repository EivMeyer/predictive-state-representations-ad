import torch
import torch.nn as nn

class PredictiveModelV1(nn.Module):
    def __init__(self, obs_dim, action_dim, ego_state_dim, hidden_dim=64):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ego_state_dim = ego_state_dim

        # Improved CNN encoder
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.obs_flatten = nn.Flatten()
        
        # GRU for sequence processing
        self.sequence_gru = nn.GRU(128 * 8 * 8 + action_dim + ego_state_dim, hidden_dim, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, obs_dim),
            nn.Sigmoid()  # Ensure output is in [0, 1] range for image prediction
        )

    def forward(self, batch):
        observations, actions, ego_states = batch['observations'], batch['actions'], batch['ego_states']
        batch_size, seq_len, height, width, channels = observations.shape
        
        # Process observations
        obs_reshaped = observations.permute(0, 1, 4, 2, 3).contiguous()
        obs_reshaped = obs_reshaped.view(-1, channels, height, width)
        obs_features = self.obs_encoder(obs_reshaped)
        obs_features = self.obs_flatten(obs_features)
        obs_features = obs_features.view(batch_size, seq_len, -1)
        
        # Combine features
        combined_features = torch.cat([obs_features, actions, ego_states], dim=-1)
        
        # Process sequence
        gru_out, _ = self.sequence_gru(combined_features)
        
        # Apply attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Decode
        final_features = attn_out[:, -1, :]  # Use the last timestep
        prediction = self.decoder(final_features)
        
        prediction = prediction.view(batch_size, height, width, channels)
        
        return prediction