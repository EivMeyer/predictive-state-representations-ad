import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel
from models.autoencoder_model_v0 import AutoEncoderModelV0
from models.autoencoder_model_v0m1 import AutoEncoderModelV0M1
from models.loss_functions import CombinedLoss
from utils.file_utils import find_model_path
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class VariationalLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, output_dim)
        self.fc_logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class PredictiveModelV9M2(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg, 
                 pretrained_model_path=None, nhead=16, num_encoder_layers=8, num_decoder_layers=8):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)

        self.seq_len, self.channels, self.height, self.width = obs_shape

        pretrained_model_path = find_model_path(cfg.project_dir, cfg.models.PredictiveModelV9M2.pretrained_model_path) if pretrained_model_path is None else pretrained_model_path
        assert pretrained_model_path is not None, "Pretrained model path must be provided"

        # Load pre-trained autoencoder
        if cfg.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = cfg.device
        self.autoencoder = AutoEncoderModelV0M1(obs_shape, action_dim, ego_state_dim, cfg)
        try:
            self.autoencoder.load_state_dict(torch.load(pretrained_model_path, map_location=device)['model_state_dict'], strict=False)
        except RuntimeError as e:
            print(f"Could not load the full state dict from {pretrained_model_path}. Using empty state dict instead.")
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
            latent_dim = torch.prod(torch.tensor(dummy_encoding.shape[1:])).item()
            latent_shape = dummy_encoding.shape[1:]

        self.latent_dim = latent_dim
        self.latent_shape = latent_shape
        self.hidden_dim = cfg.training.hidden_dim

        # Projectors
        self.latent_projector = nn.Linear(latent_dim, self.hidden_dim) if latent_dim != self.hidden_dim else nn.Identity()
        self.ego_state_projector = nn.Linear(ego_state_dim, self.hidden_dim)

        # Readout token
        self.readout_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Separate positional encodings for encoder and decoder
        self.pos_encoder_encoding = PositionalEncoding(self.hidden_dim, max_len=self.seq_len + 1)  # +1 for readout token
        self.pos_encoder_decoding = PositionalEncoding(self.hidden_dim, max_len=self.num_frames_to_predict)

        # Transformer encoder and decoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=nhead, dim_feedforward=1024, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        decoder_layers = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=nhead, dim_feedforward=1024, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

        # Output projector
        self.output_projector = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Variational Information Bottleneck
        self.vib = VariationalLayer(self.hidden_dim, self.hidden_dim)

        # Variational layers for each decoding step
        self.variational_layers = nn.ModuleList([
            VariationalLayer(self.hidden_dim, self.latent_dim) 
            for _ in range(self.num_frames_to_predict)
        ])

        # Future action conditioning
        self.action_projector = nn.Linear(action_dim, self.hidden_dim)
        self.condition_on_future_actions = cfg.models.PredictiveModelV9M2.condition_on_future_actions

        # Loss function for latent space
        self.loss_function_latent = CombinedLoss(
            mse_weight=self.cfg.training.loss.mse_weight,
            l1_weight=self.cfg.training.loss.l1_weight,
            diversity_weight=self.cfg.training.loss.diversity_weight,
            latent_l1_weight=self.cfg.training.loss.latent_l1_weight,
            latent_l2_weight=self.cfg.training.loss.latent_l2_weight,
            temporal_decay=self.cfg.training.loss.temporal_decay,
            use_sample_weights=False
        )

        # Loss function for observations 
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

        # Apply Variational Information Bottleneck to the readout token
        vib_encoded, vib_mu, vib_logvar = self.vib(memory[-1])

        return vib_encoded, vib_mu, vib_logvar

    def decode(self, batch, memory, sample=True):
        batch_size, hidden_dim = memory.shape

        # Prepare decoder input
        decoder_input = self.pos_encoder_decoding(memory.unsqueeze(0).repeat(self.num_frames_to_predict, 1, 1))

        # Incorporate future actions if provided and configured to use them
        if self.condition_on_future_actions:
            action_features = self.action_projector(batch['next_actions'])
            decoder_input += action_features.permute(1, 0, 2)

        # Generate future latent predictions
        output = self.transformer_decoder(decoder_input, memory.unsqueeze(0))

        # Project output and apply variational layers for each decoding step
        predicted_latents = []
        vae_mus = []
        vae_logvars = []
        for i in range(self.num_frames_to_predict):
            hidden = self.output_projector(output[i])
            z, mu, logvar = self.variational_layers[i](hidden)
            if not sample:
                z = mu
            predicted_latents.append(z)
            vae_mus.append(mu)
            vae_logvars.append(logvar)

        predicted_latents = torch.stack(predicted_latents, dim=1)
        vae_mus = torch.stack(vae_mus, dim=1)
        vae_logvars = torch.stack(vae_logvars, dim=1)

        return predicted_latents, vae_mus, vae_logvars

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch, sample=True):
        encoded_state, vib_mu, vib_logvar = self.encode(batch)
        predicted_latents, vae_mus, vae_logvars = self.decode(batch, encoded_state, sample)

        # Decode latents to observations using the trainable decoder
        predictions = self.trainable_decoder(predicted_latents.view(-1, *self.latent_shape))
        predictions = predictions.view(batch['observations'].shape[0], self.num_frames_to_predict, *self.obs_shape[-3:])

        return {
            "predicted_latents": predicted_latents,
            "encoded_state": encoded_state,
            "predictions": predictions,
            "vib_mu": vib_mu,
            "vib_logvar": vib_logvar,
            "vae_mus": vae_mus,
            "vae_logvars": vae_logvars
        }

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
    def decode_image(self, batch, encoded_state):
        predicted_latents, _, _ = self.decode(batch, encoded_state)

        predictions = self.trainable_decoder(predicted_latents.view(-1, self.latent_dim))
        predictions = predictions.view(batch['observations'].shape[0], self.num_frames_to_predict, *self.obs_shape[-3:])

        return predictions

    def compute_loss(self, batch, model_output):
        predictions = model_output['predictions']
        predicted_latents = model_output['predicted_latents']
        encoded_state = model_output['encoded_state']
        vib_mu = model_output['vib_mu']
        vib_logvar = model_output['vib_logvar']
        vae_mus = model_output['vae_mus']
        vae_logvars = model_output['vae_logvars']
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
            if isinstance(target_latents, tuple):
                target_latents = target_latents[0]
            target_latents = target_latents.view(batch_size, seq_len, -1)

        # Calculate loss
        loss, loss_components = self.loss_function_observations(predictions, target_observations, encoded_state)

        # Add latent space loss
        latent_loss, latent_loss_components = self.loss_function_latent(predicted_latents, target_latents, encoded_state)
        loss += latent_loss
        loss_components.update({f"latent_{k}": v for k, v in latent_loss_components.items()})

        # Add KL divergence loss for Variational Information Bottleneck
        vib_kl_loss = self.kl_divergence(vib_mu, vib_logvar).mean()
        loss += self.cfg.models.PredictiveModelV9M2.vib_weight * vib_kl_loss
        loss_components['vib_kl_loss'] = vib_kl_loss.item()

        # Add KL divergence loss for variational decoding
        vae_kl_loss = self.kl_divergence(vae_mus, vae_logvars).mean()
        loss += self.cfg.models.PredictiveModelV9M2.vae_weight * vae_kl_loss
        loss_components['vae_kl_loss'] = vae_kl_loss.item()

        return loss, loss_components
    
    def get_parameter_groups(self):
        return [
            {'params': self.other_params},
            {'params': self.decoder_params, 'lr': self.cfg.models.PredictiveModelV9M2.decoder_learning_rate}
        ]