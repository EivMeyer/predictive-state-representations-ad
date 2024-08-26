import torch
import torch.nn as nn
import numpy as np

# Goal: Simplest possible extension to V6 model to predict future frames.

class PredictiveModelV7(nn.Module):
    def __init__(self, obs_shape, action_dim, ego_state_dim, hidden_dim=64, num_frames_to_predict=5):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim
        self.num_frames_to_predict = num_frames_to_predict
        
        # CNN encoder with BatchNorm (same as before)
        self.encoder = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Second Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Third Convolutional Block with Dilated Convolutions
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Final Convolutional Block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Calculate the size of the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, obs_shape[-2], obs_shape[-1])
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.view(1, -1).size(1)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # BatchNorm for LSTM output
        self.lstm_bn = nn.BatchNorm1d(self.hidden_dim)
        
        # Frame predictor (converts LSTM hidden state to next frame)
        self.frame_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Decoder with BatchNorm
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

        # Special initialization for the last layer
        nn.init.xavier_uniform_(self.decoder[-2].weight, gain=0.01)

    def forward(self, batch):
        observations = batch['observations']
        batch_size, seq_len, channels, height, width = observations.shape

        # Reshape to process all time steps at once
        observations_reshaped = observations.view(batch_size * seq_len, channels, height, width)
        
        # Process all observations through the encoder
        encoded = self.encoder(observations_reshaped)
        
        # Reshape the encoded output
        encoded = encoded.view(batch_size, seq_len, -1)

        # Process through LSTM
        _, (h_n, _) = self.lstm(encoded)
        h_t = h_n[-1]
        
        # Generate multiple frames (vectorized)
        h_t = h_t.unsqueeze(1).expand(-1, self.num_frames_to_predict, -1)
        h_t = h_t.reshape(-1, self.hidden_dim)
        
        # Apply frame predictor to all time steps at once
        h_t = self.frame_predictor(h_t)
        
        # Apply BatchNorm
        h_t = self.lstm_bn(h_t)
        
        # Reshape for decoder
        x = h_t.view(-1, self.hidden_dim, 1, 1)
        
        # Generate predictions using the decoder
        predictions = self.decoder(x)
        
        # Reshape predictions to (batch_size, num_frames_to_predict, channels, height, width)
        predictions = predictions.view(batch_size, self.num_frames_to_predict, channels, height, width)
        
        return predictions