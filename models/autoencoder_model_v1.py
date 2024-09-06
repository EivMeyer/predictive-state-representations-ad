import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel

class AutoEncoderModelV1(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)

        self.encoder = self._make_encoder(obs_shape)

        # Calculate the size of the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[-3], obs_shape[-2], obs_shape[-1])
            encoder_output_size = self.encoder(dummy_input).shape[1]

        self.latent_dim = self.hidden_dim
        self.fc_latent = nn.Linear(encoder_output_size, self.latent_dim)

        # Decoder
        self.decoder = self._make_decoder(obs_shape)

        self._initialize_weights()

    def _make_encoder(self, obs_shape):
        return nn.Sequential(
            self._make_encoder_block(obs_shape[-3], 64),
            self._make_encoder_block(64, 128),
            self._make_encoder_block(128, 256),
            self._make_encoder_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

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

    def _make_decoder(self, obs_shape):
        if obs_shape[-2:] == (64, 64):
            return nn.Sequential(
                nn.Linear(self.latent_dim, 512 * 4 * 4),
                nn.ReLU(inplace=True),
                nn.Unflatten(1, (512, 4, 4)),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(obs_shape[-3]),
            )
        elif obs_shape[-2:] == (128, 128):
            return nn.Sequential(
                nn.Linear(self.latent_dim, 512 * 4 * 4),
                nn.ReLU(inplace=True),
                nn.Unflatten(1, (512, 4, 4)),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
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
                nn.ConvTranspose2d(32, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),
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

    def encode(self, batch):
        observation = batch['observations'][:, -1] if batch['observations'].ndim == 5 else batch['observations']
        encoder_features = self.encoder(observation)
        latent = self.fc_latent(encoder_features)
        return latent

    def decode(self, batch, latent):
        return self.decoder(latent)

    def forward(self, batch):
        latent = self.encode(batch)
        reconstruction = self.decode(batch, latent).unsqueeze(1)  # Add sequence length dimension
        return {'predictions': reconstruction, 'encoded_state': latent}

    def compute_loss(self, batch, model_output):
        reconstruction = model_output['predictions']
        target_observation = batch['observations'][:, (-1,)]  # Take only the last observation

        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(reconstruction, target_observation, reduction='mean')

        loss_components = {
            'mse': mse_loss.item(),
        }

        return mse_loss, loss_components