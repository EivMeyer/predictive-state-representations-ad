import torch
import torch.nn as nn
from models.base_predictive_model import BasePredictiveModel
from models.autoencoder_model_v1 import AutoEncoderModelV1
from models.loss_functions import CombinedLoss
from utils.file_utils import find_model_path

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class PredictiveModelV9(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg, 
                 pretrained_model_path=None, nhead=16, num_encoder_layers=8, num_decoder_layers=8):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)

        pretrained_model_path = find_model_path(cfg.project_dir, cfg.models.PredictiveModelV9.pretrained_model_path) if pretrained_model_path is None else pretrained_model_path
        assert pretrained_model_path is not None, "Pretrained model path must be provided"

        # Load pre-trained autoencoder
        self.autoencoder = AutoEncoderModelV1(obs_shape, action_dim, ego_state_dim, cfg)
        self.autoencoder.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
        self.autoencoder.eval()
        
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # Determine the size of the latent space
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            dummy_ego_state = torch.zeros(1, ego_state_dim)
            dummy_batch = {'observations': dummy_input, 'ego_states': dummy_ego_state}
            latent_dim = self.autoencoder.encode(dummy_batch).shape[1]

        self.latent_dim = latent_dim
        self.hidden_dim = cfg.training.hidden_dim

        # Projectors
        self.latent_projector = nn.Linear(latent_dim, self.hidden_dim) if latent_dim != self.hidden_dim else nn.Identity()
        self.ego_state_projector = nn.Linear(ego_state_dim, self.hidden_dim)

        self.pos_encoder = PositionalEncoding(self.hidden_dim, max_len=self.num_frames_to_predict)

        # Transformer encoder and decoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=nhead, dim_feedforward=1024, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        decoder_layers = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=nhead, dim_feedforward=1024, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

        # Output projector
        self.output_projector = nn.Linear(self.hidden_dim, latent_dim)

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
        latent_features = latent_features.view(batch_size, seq_len, -1)
        latent_features = self.latent_projector(latent_features)

        # Process ego states
        ego_features = self.ego_state_projector(ego_states)

        # Combine latent features and ego features
        combined_features = latent_features + ego_features

        # Add positional encoding and apply transformer encoder
        src = self.pos_encoder(combined_features.permute(1, 0, 2))
        memory = self.transformer_encoder(src)

        # Return only the last memory state
        return memory[-1]

    def decode(self, batch, memory):
        batch_size, hidden_dim = memory.shape

        # Prepare decoder input
        decoder_input = self.pos_encoder(memory.unsqueeze(0).repeat(self.num_frames_to_predict, 1, 1))

        # Generate future latent predictions using only the last memory state
        output = self.transformer_decoder(decoder_input, memory.unsqueeze(0))

        # Project output back to latent space
        predicted_latents = self.output_projector(output.permute(1, 0, 2))

        return predicted_latents

    def forward(self, batch):
        encoded_state = self.encode(batch)
        predicted_latents = self.decode(batch, encoded_state)

        # Decode latents to observations
        predictions = self.autoencoder.decode(batch, predicted_latents.view(-1, self.latent_dim))
        predictions = predictions.view(batch['observations'].shape[0], self.num_frames_to_predict, *self.obs_shape[-3:])

        return {
            "predicted_latents": predicted_latents,
            "encoded_state": encoded_state,
            "predictions": predictions
        }

    def compute_loss(self, batch, model_output):
        predictions = model_output['predictions']
        predicted_latents = model_output['predicted_latents']
        encoded_state = model_output['encoded_state']
        target_observations = batch['next_observations']

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
            target_latents = target_latents.view(batch_size, seq_len, -1)

        # Calculate loss
        loss, loss_components = self.loss_function(predicted_latents, target_latents, encoded_state)

        return loss, loss_components