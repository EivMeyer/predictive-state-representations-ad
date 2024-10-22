import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel
from models.loss_functions import CombinedLoss

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main path with wider kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        
        # Enhanced skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        
        # Detail preservation branch
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        main = F.leaky_relu(self.bn1(self.conv1(x)))
        main = self.bn2(self.conv2(main))
        detail = self.detail_branch(x)
        return F.leaky_relu(main + self.shortcut(x) + detail)

class AutoEncoderModelV0M2(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)
        
        self.latent_dim = self.hidden_dim
        self.encoder = self._make_enhanced_encoder(obs_shape)
        
        # Calculate encoder output size with both pooling paths
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[-3], obs_shape[-2], obs_shape[-1])
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[1] // 2  # Divide by 2 because output contains both mu and logvar
        
        # VAE components
        self.fc_mu = nn.Linear(encoder_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_size, self.latent_dim)
        
        # Improved decoder
        self.decoder = self._make_enhanced_decoder(obs_shape)
        
        self._initialize_weights()
        
        # Initialize CombinedLoss with adjusted weights
        self.combined_loss = CombinedLoss(
            mse_weight=cfg.training.loss.mse_weight * 0.7,  # Reduced MSE weight
            l1_weight=cfg.training.loss.l1_weight * 0.7,    # Reduced L1 weight
            diversity_weight=cfg.training.loss.diversity_weight,
            latent_l1_weight=0,
            latent_l2_weight=0,
            temporal_decay=cfg.training.loss.temporal_decay,
            perceptual_weight=cfg.training.loss.perceptual_weight * 1.5,  # Increased perceptual weight
            num_scales=1
        )
        
        # KL divergence weight
        self.kl_weight = cfg.models.AutoEncoderModelV0.kl_weight

    def _make_enhanced_encoder(self, obs_shape):
        return nn.Sequential(
            # Initial feature extraction with larger kernel
            nn.Conv2d(obs_shape[-3], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Enhanced residual blocks
            EnhancedResidualBlock(64, 128, stride=2),
            EnhancedResidualBlock(128, 256, stride=2),
            EnhancedResidualBlock(256, 512, stride=2),
            
            # Parallel pooling paths
            ParallelPooling(),
            nn.Flatten()
        )
    
    def _make_enhanced_decoder(self, obs_shape):
        class DetailPreservingBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
                self.detail = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
                
            def forward(self, x):
                main = self.main(x)
                return main + self.detail(main)
        
        if obs_shape[-2:] == (64, 64):
            return nn.Sequential(
                nn.Linear(self.latent_dim, 512 * 4 * 4),
                nn.LeakyReLU(0.2),
                nn.Unflatten(1, (512, 4, 4)),
                DetailPreservingBlock(512, 256),
                DetailPreservingBlock(256, 128),
                DetailPreservingBlock(128, 64),
                nn.Sequential(
                    nn.ConvTranspose2d(64, obs_shape[-3], 4, 2, 1),
                    nn.Conv2d(obs_shape[-3], obs_shape[-3], 3, 1, 1),
                    nn.Sigmoid()
                )
            )
        elif obs_shape[-2:] == (128, 128):
            return nn.Sequential(
                nn.Linear(self.latent_dim, 512 * 4 * 4),
                nn.LeakyReLU(0.2),
                nn.Unflatten(1, (512, 4, 4)),
                DetailPreservingBlock(512, 256),
                DetailPreservingBlock(256, 128),
                DetailPreservingBlock(128, 64),
                DetailPreservingBlock(64, 32),
                nn.Sequential(
                    nn.ConvTranspose2d(32, obs_shape[-3], 4, 2, 1),
                    nn.Conv2d(obs_shape[-3], obs_shape[-3], 3, 1, 1),
                    nn.Sigmoid()
                )
            )
        else:
            raise NotImplementedError(f"Decoder not implemented for shape {obs_shape[-2:]}. Only (64, 64) and (128, 128) are supported.")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use fan_in mode for better stability in deep networks
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m in (self.fc_mu, self.fc_logvar):
                    # Initialize VAE projection layers with smaller weights
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # Other linear layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, batch):
        if batch['observations'].ndim == 5:
            observation = batch['observations'][:, -1]
        else:
            observation = batch['observations']
            
        encoder_features = self.encoder(observation)
        encoder_features_split = torch.split(encoder_features, encoder_features.size(1)//2, dim=1)
        mu = self.fc_mu(encoder_features_split[0])
        logvar = self.fc_logvar(encoder_features_split[1])
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

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

        # Compute reconstruction loss with adjusted weights
        combined_loss, loss_components = self.combined_loss(reconstruction, target_observation, mu)

        # KL divergence loss with warm-up
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = combined_loss + self.kl_weight * kl_loss

        # Update loss components dictionary
        loss_components['kl_loss'] = kl_loss.item()
        loss_components['kl_weight'] = self.kl_weight
        loss_components['total_loss'] = total_loss.item()

        return total_loss, loss_components

class ParallelPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
    def forward(self, x):
        avg_features = self.avg_pool(x)
        max_features = self.max_pool(x)
        return torch.cat([avg_features, max_features], dim=1)