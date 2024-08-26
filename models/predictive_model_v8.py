import torch
import torch.nn as nn
import math

# Goal: Adapted from PredictiveModelV7, featuring dilated convolutions and multiple convolutional blocks to capture complex spatial features from input images. Also utilizes a transformer encoder-decoder structure to process temporal dependencies in the input sequence and generate multi-step predictions.

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

class PredictiveModelV8(nn.Module):
    def __init__(self, obs_shape, action_dim, ego_state_dim, hidden_dim=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, num_frames_to_predict=5):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim
        self.num_frames_to_predict = num_frames_to_predict
        
        # CNN encoder (from V7)
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
            encoder_output_size = self.encoder(dummy_input).shape[1]
        
        # Linear layer to project encoder output to hidden_dim
        self.encoder_projector = nn.Linear(encoder_output_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Transformer decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

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

    def forward(self, batch):
        observations = batch['observations']
        batch_size, seq_len, channels, height, width = observations.shape
        
        # Process all observations through the encoder
        observations_reshaped = observations.view(-1, channels, height, width)
        encoder_features = self.encoder(observations_reshaped)
        encoder_features = self.encoder_projector(encoder_features)
        encoder_features = encoder_features.view(batch_size, seq_len, self.hidden_dim)
        
        # Add positional encoding
        src = self.pos_encoder(encoder_features.permute(1, 0, 2))
        
        # Pass through transformer encoder
        memory = self.transformer_encoder(src)
        
        # Prepare decoder input for all future frames at once
        decoder_input = memory[-1].unsqueeze(0).repeat(self.num_frames_to_predict, 1, 1)
        decoder_input = self.pos_encoder(decoder_input)
        
        # Generate future frame predictions in parallel
        output = self.transformer_decoder(decoder_input, memory)
        
        # Reshape output for decoding
        output = output.view(-1, self.hidden_dim, 1, 1)  # Reshape to (N, C, H, W) format for ConvTranspose2d

        # Decode predicted features back into image space
        predictions = self.decoder(output)
        
        # Reshape output to (batch_size, num_frames_to_predict, channels, height, width)
        predictions = predictions.view(batch_size, self.num_frames_to_predict, channels, height, width)
        
        return predictions