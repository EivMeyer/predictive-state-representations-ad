import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class PredictiveModelV8(nn.Module):
    def __init__(self, obs_shape, action_dim, ego_state_dim, hidden_dim=256, nhead=16, num_encoder_layers=8, num_decoder_layers=8, num_frames_to_predict=5):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim
        self.num_frames_to_predict = num_frames_to_predict
        self.ego_state_dim = ego_state_dim
        
        # Wider CNN encoder
        self.encoder = nn.Sequential(
            self._make_encoder_block(3, 64),
            self._make_encoder_block(64, 128),
            self._make_encoder_block(128, 256),
            self._make_encoder_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Calculate the size of the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, obs_shape[-2], obs_shape[-1])
            encoder_output_size = self.encoder(dummy_input).shape[1]
        
        self.encoder_projector = nn.Linear(encoder_output_size, hidden_dim)
        self.ego_state_projector = nn.Linear(ego_state_dim, hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=num_frames_to_predict)
        
        # Deeper transformer encoder and decoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=1024, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=1024, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

        # Wider convolutional decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
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
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
        )
        
        self._initialize_weights()

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
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

        # Add positional encoding and apply transformer encoder
        src = self.pos_encoder(combined_features.permute(1, 0, 2))
        memory = self.transformer_encoder(src)

        # Return only the last memory state
        return memory[-1]

    def forward(self, batch):
        observations = batch['observations']
        ego_states = batch['ego_states']

        last_memory = self.encode(observations, ego_states)

        # Prepare decoder input
        decoder_input = self.pos_encoder(last_memory.unsqueeze(0).repeat(self.num_frames_to_predict, 1, 1))

        # Generate future frame predictions using only the last memory state
        output = self.transformer_decoder(decoder_input, last_memory.unsqueeze(0))

        # Reshape for convolutional decoder
        output = output.permute(1, 0, 2).contiguous()
        output = output.view(-1, self.hidden_dim, 1, 1)

        # Apply convolutional decoder
        predictions = self.decoder(output)

        # Reshape to [batch_size, num_frames_to_predict, channels, height, width]
        predictions = predictions.view(observations.shape[0], self.num_frames_to_predict, self.obs_shape[-3], self.obs_shape[-2], self.obs_shape[-1])

        return predictions