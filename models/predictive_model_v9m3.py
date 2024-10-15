import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel
from models.autoencoder_model_v0 import AutoEncoderModelV0
from models.loss_functions import CombinedLoss
from utils.file_utils import find_model_path
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class PredictiveModelV9M3(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg, 
                 pretrained_model_path=None, nhead=16, num_encoder_layers=8, num_decoder_layers=8):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)

        pretrained_model_path = find_model_path(cfg.project_dir, cfg.models.PredictiveModelV9.pretrained_model_path) if pretrained_model_path is None else pretrained_model_path
        assert pretrained_model_path is not None, "Pretrained model path must be provided"

        # Load pre-trained autoencoder
        if cfg.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = cfg.device
        self.autoencoder = AutoEncoderModelV0(obs_shape, action_dim, ego_state_dim, cfg)
        self.autoencoder.load_state_dict(torch.load(pretrained_model_path, map_location=device)['model_state_dict'], strict=False)
        self.autoencoder.eval()
        
        # Freeze the encoder part of the autoencoder
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False

        # Create a trainable copy of the decoder
        self.trainable_decoder = copy.deepcopy(self.autoencoder.decoder)

        # Determine the size of the latent space
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            dummy_ego_state = torch.zeros(1, ego_state_dim)
            dummy_batch = {'observations': dummy_input, 'ego_states': dummy_ego_state}
            dummy_encoding = self.autoencoder.encode(dummy_batch)
            if isinstance(dummy_encoding, tuple):
                dummy_encoding = dummy_encoding[0]
            latent_dim = dummy_encoding.shape[1]

        self.latent_dim = latent_dim
        self.hidden_dim = cfg.training.hidden_dim

        # Projectors
        self.latent_projector = nn.Linear(latent_dim, self.hidden_dim) if latent_dim != self.hidden_dim else nn.Identity()
        self.ego_state_projector = nn.Linear(ego_state_dim, self.hidden_dim)

        # Readout token
        self.readout_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Separate positional encodings for encoder and decoder
        self.pos_encoder_encoding = PositionalEncoding(self.hidden_dim, max_len=self.num_frames_to_predict + 1)  # +1 for readout token
        self.pos_encoder_decoding = PositionalEncoding(self.hidden_dim, max_len=self.num_frames_to_predict)

        # Transformer encoder and decoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=nhead, dim_feedforward=1024, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        decoder_layers = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=nhead, dim_feedforward=1024, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

        # Output projectors
        self.output_projector = nn.Linear(self.hidden_dim, latent_dim)
        self.hazard_projector = nn.Linear(self.hidden_dim, 1)  # Projector for hazard function

        # Loss functions
        self.loss_function_latent = CombinedLoss(
            mse_weight=self.cfg.training.loss.mse_weight,
            l1_weight=self.cfg.training.loss.l1_weight,
            diversity_weight=self.cfg.training.loss.diversity_weight,
            latent_l1_weight=self.cfg.training.loss.latent_l1_weight,
            latent_l2_weight=self.cfg.training.loss.latent_l2_weight,
            temporal_decay=self.cfg.training.loss.temporal_decay,
            use_sample_weights=False
        )

        self.loss_function_observations = CombinedLoss(
            mse_weight=self.cfg.training.loss.mse_weight,
            l1_weight=self.cfg.training.loss.l1_weight,
            diversity_weight=0.0,
            latent_l1_weight=0.0,
            latent_l2_weight=0.0,
            temporal_decay=self.cfg.training.loss.temporal_decay,
            use_sample_weights=False
        )

        self._initialize_weights()

        # Create parameter groups for different learning rates
        self.decoder_params = list(self.trainable_decoder.parameters())
        self.other_params = [p for n, p in self.named_parameters() if not any(p is dp for dp in self.decoder_params)]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, batch):
        observations = batch['observations']
        ego_states = batch['ego_states']

        batch_size, seq_len, channels, height, width = observations.shape

        # Process observations
        observations_reshaped = observations.view(-1, channels, height, width)
        with torch.no_grad():
            latent_features = self.autoencoder.encode({'observations': observations_reshaped, 'ego_states': ego_states.view(-1, ego_states.shape[-1])})
            if isinstance(latent_features, tuple):
                latent_features = latent_features[0]
        latent_features = latent_features.view(batch_size, seq_len, -1)
        latent_features = self.latent_projector(latent_features)

        # Process ego states
        ego_features = self.ego_state_projector(ego_states)

        # Combine latent features and ego features
        combined_features = latent_features + ego_features

        # Add readout token
        readout_tokens = self.readout_token.expand(batch_size, -1, -1)
        combined_features = torch.cat([combined_features, readout_tokens], dim=1)

        # Add positional encoding
        src = self.pos_encoder_encoding(combined_features.permute(1, 0, 2))

        # Apply transformer encoder
        memory = self.transformer_encoder(src)

        # Return only the readout token's final state
        return memory[-1]

    def decode(self, batch, memory):
        batch_size, hidden_dim = memory.shape

        # Prepare decoder input
        decoder_input = self.pos_encoder_decoding(memory.unsqueeze(0).repeat(self.num_frames_to_predict, 1, 1))

        # Generate future latent predictions using only the last memory state
        output = self.transformer_decoder(decoder_input, memory.unsqueeze(0))

        # Project output back to latent space
        predicted_latents = self.output_projector(output.permute(1, 0, 2))

        # Predict hazard function
        hazard = self.hazard_projector(output.permute(1, 0, 2)).squeeze(-1)

        return predicted_latents, hazard
    
    def decode_image(self, batch, encoded_state):
        predicted_latents, hazard = self.decode(batch, encoded_state)

        predictions = self.trainable_decoder(predicted_latents.view(-1, self.latent_dim))
        predictions = predictions.view(batch['observations'].shape[0], self.num_frames_to_predict, *self.obs_shape[-3:])

        return predictions, hazard

    def forward(self, batch):
        encoded_state = self.encode(batch)
        predicted_latents, hazard = self.decode(batch, encoded_state)

        # Decode latents to observations using the trainable decoder
        predictions = self.trainable_decoder(predicted_latents.view(-1, self.latent_dim))
        predictions = predictions.view(batch['observations'].shape[0], self.num_frames_to_predict, *self.obs_shape[-3:])

        return {
            "predicted_latents": predicted_latents,
            "encoded_state": encoded_state,
            "predictions": predictions,
            "hazard": hazard
        }

    def compute_loss(self, batch, model_output):
        predictions = model_output['predictions']
        predicted_latents = model_output['predicted_latents']
        encoded_state = model_output['encoded_state']
        hazard = model_output['hazard']
        target_observations = batch['next_observations']
        target_done = batch['dones'][:, -self.num_frames_to_predict:]

        # Create a mask for valid (not done) timesteps
        valid_mask = torch.cumprod(1 - target_done.int(), dim=1)

        # Get target latents using the pretrained autoencoder
        with torch.no_grad():
            batch_size, seq_len, channels, height, width = target_observations.shape
            target_observations_reshaped = target_observations.view(-1, channels, height, width)
            target_ego_states = batch['ego_states'][:, -self.num_frames_to_predict:, :]
            target_ego_states_reshaped = target_ego_states.view(-1, target_ego_states.shape[-1])
            target_batch = {
                'observations': target_observations_reshaped,
                'ego_states': target_ego_states_reshaped
            }
            target_latents = self.autoencoder.encode(target_batch)
            if isinstance(target_latents, tuple):
                target_latents = target_latents[0]
            target_latents = target_latents.view(batch_size, seq_len, -1)

        # Calculate losses with masking
        loss_obs, loss_components_obs = self.loss_function_observations(
            predictions * valid_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            target_observations * valid_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            encoded_state
        )
        loss_latent, loss_components_latent = self.loss_function_latent(
            predicted_latents * valid_mask.unsqueeze(-1),
            target_latents * valid_mask.unsqueeze(-1),
            encoded_state
        )

        # Normalize losses by the number of valid timesteps
        num_valid = valid_mask.sum()
        loss_obs = loss_obs * valid_mask.numel() / num_valid
        loss_latent = loss_latent * valid_mask.numel() / num_valid

        # Survival analysis loss
        cumulative_hazard = torch.cumsum(F.softplus(hazard), dim=1)
        not_done_mask = 1 - target_done.float()
        survival_likelihood = torch.exp(-cumulative_hazard) * not_done_mask + (1 - torch.exp(-cumulative_hazard)) * target_done.float()
        loss_survival = -torch.log(survival_likelihood + 1e-8).mean()

        # Combine losses
        total_loss = loss_obs + loss_latent + loss_survival

        # Combine loss components
        loss_components = {
            **{f"obs_{k}": v * valid_mask.numel() / num_valid for k, v in loss_components_obs.items()},
            **{f"latent_{k}": v * valid_mask.numel() / num_valid for k, v in loss_components_latent.items()},
            "survival_loss": loss_survival.item()
        }

        return total_loss, loss_components

    def predict_survival_probability(self, hazard):
        cumulative_hazard = torch.cumsum(F.softplus(hazard), dim=1)
        survival_probability = torch.exp(-cumulative_hazard)
        return survival_probability

    def predict_done_probability(self, hazard):
        survival_probability = self.predict_survival_probability(hazard)
        done_probability = 1 - survival_probability
        return done_probability