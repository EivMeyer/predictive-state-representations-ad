import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class PredictiveModelV2(nn.Module):
    def __init__(self, obs_dim, action_dim, ego_state_dim, hidden_dim=256, nhead=8, num_layers=6):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ego_state_dim = ego_state_dim

        # CNN for processing observations
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        cnn_output_size = 128 * 8 * 8
        
        # Linear layer to combine CNN output with actions and ego states
        self.input_fc = nn.Linear(cnn_output_size + action_dim + ego_state_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, obs_dim),
            nn.Sigmoid()  # Ensure output is in [0, 1] range for image prediction
        )

    def forward(self, batch):
        observations, actions, ego_states = batch['observations'], batch['actions'], batch['ego_states']
        batch_size, seq_len, height, width, channels = observations.shape
        
        # Correct handling of dimensions for CNN
        obs_reshaped = observations.permute(0, 1, 4, 2, 3).contiguous()
        obs_reshaped = obs_reshaped.view(-1, channels, height, width)
        
        cnn_output = self.cnn(obs_reshaped)
        cnn_output = cnn_output.view(batch_size, seq_len, -1)
        
        # Combine CNN output with actions and ego states
        combined = torch.cat([cnn_output, actions, ego_states], dim=-1)
        encoder_input = self.input_fc(combined)
        
        # Add positional encoding
        encoder_input = self.pos_encoder(encoder_input.permute(1, 0, 2))
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(encoder_input)
        
        # Use the last output for prediction
        final_hidden = transformer_output[-1]
        
        # Generate prediction
        prediction = self.output_fc(final_hidden)
        prediction = prediction.view(batch_size, height, width, channels)
        
        return prediction