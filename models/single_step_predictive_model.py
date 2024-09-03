import torch
import torch.nn as nn
from models.base_predictive_model import BasePredictiveModel
from models.loss_functions import CombinedLoss

class SingleStepPredictiveModel(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)

        self.encoder = self._make_encoder(obs_shape)
        self.decoder = self._make_decoder(obs_shape)

        # Calculate the size of the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[-3], obs_shape[-2], obs_shape[-1])
            encoder_output_size = self.encoder(dummy_input).shape[1]

        self.encoder_projector = nn.Linear(encoder_output_size, self.hidden_dim)

        # Loss function
        self.loss_function = CombinedLoss(
            mse_weight=self.cfg.training.loss.mse_weight,
            l1_weight=self.cfg.training.loss.l1_weight,
            diversity_weight=self.cfg.training.loss.diversity_weight,
            latent_l1_weight=self.cfg.training.loss.latent_l1_weight,
            latent_l2_weight=self.cfg.training.loss.latent_l2_weight,
            temporal_decay=self.cfg.training.loss.temporal_decay,
        )

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
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, batch):
        observations = batch['observations']
        
        # Take only the last observation from the sequence
        last_obs = observations[:, -1, :, :, :]
        
        # Process observation through encoder
        encoded_state = self.encoder_projector(self.encoder(last_obs))
        
        return encoded_state

    def decode(self, batch, encoded_state):
        # Reshape for convolutional decoder
        output = encoded_state.view(-1, self.hidden_dim, 1, 1)

        # Apply convolutional decoder
        prediction = self.decoder(output)

        # Add an extra dimension for num_frames_to_predict (which is 1 in this case)
        prediction = prediction.unsqueeze(1)

        return prediction

    def compute_loss(self, batch, encoded_state, predictions):
        target_observations = batch['next_observations']

        # Calculate loss
        loss, loss_components = self.loss_function(predictions, target_observations, encoded_state)

        return loss, loss_components