import torch
import torch.nn as nn
import math
from models.base_predictive_model import BasePredictiveModel

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PredictiveModelV8Lightweight(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, num_frames_to_predict, hidden_dim, nhead=8, num_layers=4):
        super().__init__(obs_shape, action_dim, ego_state_dim, num_frames_to_predict, hidden_dim)

        self.encoder = self._make_encoder(obs_shape)
        
        # Calculate the size of the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[-3], obs_shape[-2], obs_shape[-1])
            encoder_output_size = self.encoder(dummy_input).shape[1]

        self.encoder_projector = nn.Linear(encoder_output_size, hidden_dim)
        self.ego_state_projector = nn.Linear(ego_state_dim, hidden_dim)

        self.pos_encoder = LearnablePositionalEncoding(hidden_dim, max_len=num_frames_to_predict)

        # Lightweight Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        # Decoder
        self.decoder = self._make_decoder(obs_shape)

        self._initialize_weights()

    def _make_encoder(self, obs_shape):
        return nn.Sequential(
            self._make_encoder_block(obs_shape[-3], 32),
            self._make_encoder_block(32, 64),
            self._make_encoder_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def _make_decoder(self, obs_shape):
        if obs_shape[-2:] == (64, 64):
            return nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, 128, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(obs_shape[-3]),
            )
        elif obs_shape[-2:] == (128, 128):
            return nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(obs_shape[-3]),
            )
        else:
            raise NotImplementedError(f"Decoder not implemented for shape {obs_shape[-2:]}. Only (64, 64) and (128, 128) are supported.")


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, observations, ego_states):
        batch_size, seq_len, channels, height, width = observations.shape

        # Process observations
        observations_reshaped = observations.view(-1, channels, height, width)
        encoder_features = self.encoder(observations_reshaped)
        encoder_features = self.encoder_projector(encoder_features)
        encoder_features = encoder_features.view(batch_size, seq_len, self.hidden_dim)

        # Process ego states
        ego_features = self.ego_state_projector(ego_states)

        # Combine encoder features and ego features
        combined_features = encoder_features + ego_features

        # Add learnable positional encoding
        src = self.pos_encoder(combined_features)

        return src

    def decode(self, memory):
        batch_size, seq_len, _ = memory.shape

        # Prepare decoder input (use zeros as initial input)
        decoder_input = torch.zeros(batch_size, self.num_frames_to_predict, self.hidden_dim, device=memory.device)
        decoder_input = self.pos_encoder(decoder_input)

        # Generate future frame predictions
        output = self.transformer(memory, decoder_input)

        # Reshape for convolutional decoder
        output = output.view(-1, self.hidden_dim, 1, 1)

        # Apply convolutional decoder
        predictions = self.decoder(output)

        # Reshape to [batch_size, num_frames_to_predict, channels, height, width]
        predictions = predictions.view(batch_size, self.num_frames_to_predict, self.obs_shape[-3], self.obs_shape[-2], self.obs_shape[-1])

        return predictions
