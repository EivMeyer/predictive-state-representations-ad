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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PredictiveModelV4(nn.Module):
    def __init__(self, obs_dim, action_dim, ego_state_dim, hidden_dim=256, nhead=8, num_layers=6):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ego_state_dim = ego_state_dim

        # CNN for processing observations with Residual Blocks
        self.cnn = nn.Sequential(
            ResidualBlock(3, 32),
            nn.MaxPool2d(2, 2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        cnn_output_size = 128 * 8 * 8
        
        # Linear layer to combine CNN output with actions and ego states
        self.input_fc = nn.Linear(cnn_output_size + action_dim + ego_state_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers
        self.output_fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.output_bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.output_fc2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, obs_dim),
            nn.Sigmoid()
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
        
        # Apply BatchNorm
        encoder_input = encoder_input.view(-1, encoder_input.size(-1))
        encoder_input = self.input_bn(encoder_input)
        encoder_input = encoder_input.view(batch_size, seq_len, -1)
        
        # Add positional encoding
        encoder_input = self.pos_encoder(encoder_input.permute(1, 0, 2))
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(encoder_input)
        transformer_output = transformer_output.permute(1, 0, 2)
        
        # Output layers
        output = self.output_fc1(transformer_output[:, -1])
        output = output.view(-1, output.size(-1))
        output = self.output_bn1(output)
        output = output.view(batch_size, -1)
        output = F.relu(output)
        prediction = self.output_fc2(output)
        
        prediction = prediction.view(batch_size, height, width, channels)
        
        return prediction