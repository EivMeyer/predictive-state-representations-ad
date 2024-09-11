import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel
from models.loss_functions import CombinedLoss

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.leaky_relu(out)

class AutoEncoderModelV0(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)

        self.encoder = self._make_encoder(obs_shape)

        # Calculate the size of the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[-3], obs_shape[-2], obs_shape[-1])
            encoder_output_size = self.encoder(dummy_input).shape[1]

        self.latent_dim = self.hidden_dim 
        self.fc_mu = nn.Linear(encoder_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_size, self.latent_dim)

        # Decoder
        self.decoder = self._make_decoder(obs_shape)

        self._initialize_weights()

        # Initialize CombinedLoss
        self.combined_loss = CombinedLoss(
            mse_weight=cfg.training.loss.mse_weight,
            l1_weight=cfg.training.loss.l1_weight,
            diversity_weight=cfg.training.loss.diversity_weight,
            latent_l1_weight=0,  # Set to 0 as we're using KL divergence
            latent_l2_weight=0,  # Set to 0 as we're using KL divergence
            temporal_decay=cfg.training.loss.temporal_decay,
            perceptual_weight=cfg.training.loss.perceptual_weight,
            num_scales=cfg.training.loss.num_scales
        )

        # KL divergence weight
        self.kl_weight = cfg.models.AutoEncoderModelV0.kl_weight

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
            ResidualBlock(in_channels, out_channels, stride=2),
            ResidualBlock(out_channels, out_channels)
        )

    def _make_decoder(self, obs_shape):
        if obs_shape[-2:] == (64, 64):
            return nn.Sequential(
                nn.Linear(self.latent_dim, 512 * 4 * 4),
                nn.LeakyReLU(inplace=True),
                nn.Unflatten(1, (512, 4, 4)),
                self._make_decoder_block(512, 256),
                self._make_decoder_block(256, 128),
                self._make_decoder_block(128, 64),
                nn.ConvTranspose2d(64, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),
                nn.Sigmoid()
            )
        elif obs_shape[-2:] == (128, 128):
            return nn.Sequential(
                nn.Linear(self.latent_dim, 512 * 4 * 4),
                nn.LeakyReLU(inplace=True),
                nn.Unflatten(1, (512, 4, 4)),
                self._make_decoder_block(512, 256),
                self._make_decoder_block(256, 128),
                self._make_decoder_block(128, 64),
                self._make_decoder_block(64, 32),
                nn.ConvTranspose2d(32, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError(f"Decoder not implemented for shape {obs_shape[-2:]}. Only (64, 64) and (128, 128) are supported.")

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                # Use Xavier initialization for weights
                nn.init.xavier_uniform_(m.weight)  # or use nn.init.xavier_normal_ for a different distribution
                
                # Fine-tune bias initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # Initialize biases to a small positive value to aid learning
                
            elif isinstance(m, nn.InstanceNorm2d):
                # Initialize InstanceNorm weights and biases appropriately
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)  # Keep InstanceNorm weights initialized to 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def encode(self, batch):
        observation = batch['observations'][:, -1]  # Take only the last observation
        encoder_features = self.encoder(observation)
        mu = self.fc_mu(encoder_features)
        logvar = self.fc_logvar(encoder_features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, batch, z):
        return self.decoder(z)

    def forward(self, batch):
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(batch, z).unsqueeze(1)  # Add sequence length dimension
        return {'predictions': reconstruction, 'encoded_state': mu, 'logvar': logvar}

    def compute_loss(self, batch, model_output):
        reconstruction = model_output['predictions']
        mu = model_output['encoded_state']
        logvar = model_output['logvar']
        target_observation = batch['observations'][:, (-1,)]  # Take only the last observation

        # Compute CombinedLoss
        combined_loss, loss_components = self.combined_loss(reconstruction, target_observation, mu)

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = combined_loss + self.kl_weight * kl_loss

        # Update loss components
        loss_components['kl_loss'] = kl_loss.item()
        loss_components['kl_weight'] = self.kl_weight
        loss_components['total_loss'] = total_loss.item()

        return total_loss, loss_components
