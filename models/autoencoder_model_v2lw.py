import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel
from models.loss_functions import CombinedLoss

class LightweightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Single conv layer instead of residual block
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels)
    
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)))

class AutoEncoderModelV2LW(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)
        
        # Reduced latent dimension
        self.latent_dim = self.hidden_dim // 2  
        
        self.encoder = self._make_encoder(obs_shape)
        
        # Calculate encoder output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[-3], obs_shape[-2], obs_shape[-1])
            encoder_output_size = self.encoder(dummy_input).shape[1]
        
        # Simpler projection layers
        self.fc_mu = nn.Linear(encoder_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_size, self.latent_dim)
        
        # Decoder
        self.decoder = self._make_decoder(obs_shape)
        
        self._initialize_weights()
        
        # Simplified loss with fewer components
        self.combined_loss = CombinedLoss(
            mse_weight=cfg.training.loss.mse_weight,
            l1_weight=cfg.training.loss.l1_weight,
            diversity_weight=0.0,  # Removed diversity loss
            latent_l1_weight=0,
            latent_l2_weight=0,
            temporal_decay=0.0,  # Removed temporal decay
            perceptual_weight=0.0,  # Removed perceptual loss
            num_scales=1
        )
        
        # Reduced KL weight
        self.kl_weight = cfg.models.AutoEncoderModelV0.kl_weight * 0.5

    def _make_encoder(self, obs_shape):
        return nn.Sequential(
            # Fewer channels and simplified blocks
            self._make_encoder_block(obs_shape[-3], 32),
            self._make_encoder_block(32, 64),
            self._make_encoder_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def _make_encoder_block(self, in_channels, out_channels):
        return LightweightConvBlock(in_channels, out_channels, stride=2)

    def _make_decoder(self, obs_shape):
        if obs_shape[-2:] == (64, 64):
            return nn.Sequential(
                nn.Linear(self.latent_dim, 128 * 4 * 4),
                nn.LeakyReLU(inplace=True),
                nn.Unflatten(1, (128, 4, 4)),
                self._make_decoder_block(128, 64),    # 4x4 → 8x8
                self._make_decoder_block(64, 32),     # 8x8 → 16x16
                self._make_decoder_block(32, 16),     # 16x16 → 32x32
                nn.ConvTranspose2d(16, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 → 64x64
                nn.Sigmoid()
            )
        elif obs_shape[-2:] == (128, 128):
            return nn.Sequential(
                nn.Linear(self.latent_dim, 128 * 4 * 4),
                nn.LeakyReLU(inplace=True),
                nn.Unflatten(1, (128, 4, 4)),
                self._make_decoder_block(128, 64),    # 4x4 → 8x8
                self._make_decoder_block(64, 32),     # 8x8 → 16x16
                self._make_decoder_block(32, 16),     # 16x16 → 32x32
                self._make_decoder_block(16, 8),      # 32x32 → 64x64
                nn.ConvTranspose2d(8, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 → 128x128
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
                nn.init.kaiming_normal_(m.weight)  # Changed to Kaiming initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, batch):
        observation = batch['observations'][:, -1] if batch['observations'].ndim == 5 else batch['observations']
        encoder_features = self.encoder(observation)
        mu = self.fc_mu(encoder_features)
        logvar = self.fc_logvar(encoder_features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # During inference, just return the mean

    def decode(self, batch, z):
        return self.decoder(z)

    def forward(self, batch):
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(batch, z).unsqueeze(1)
        return {'predictions': reconstruction, 'encoded_state': mu, 'logvar': logvar}

    def compute_loss(self, batch, model_output):
        reconstruction = model_output['predictions']
        mu = model_output['encoded_state']
        logvar = model_output['logvar']
        target_observation = batch['observations'][:, (-1,)]

        combined_loss, loss_components = self.combined_loss(reconstruction, target_observation, mu)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = combined_loss + self.kl_weight * kl_loss

        loss_components.update({
            'kl_loss': kl_loss.item(),
            'kl_weight': self.kl_weight,
            'total_loss': total_loss.item()
        })

        return total_loss, loss_components