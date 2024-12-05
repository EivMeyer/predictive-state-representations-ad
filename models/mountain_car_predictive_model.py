import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel

class MountainCarPredictiveModel(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg, 
                 nhead=8, num_encoder_layers=4, num_decoder_layers=4):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)
        
        # In MountainCar, state is a 2d vector
        self.state_dim = 2
        self.hidden_dim = cfg.training.hidden_dim
        
        # Encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=nhead, 
            dim_feedforward=1024, 
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, 
            nhead=nhead, 
            dim_feedforward=1024, 
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, 
            num_layers=num_decoder_layers
        )
        
        # State predictor and hazard predictor
        self.state_predictor = nn.Linear(self.hidden_dim, self.state_dim)
        self.hazard_predictor = nn.Linear(self.hidden_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, batch):
        # Take only the last observation
        observations = batch['observations'][:, -1]
        batch_size = observations.shape[0]

        # Encode state
        encoded = self.state_encoder(observations)
        memory = self.transformer_encoder(encoded.unsqueeze(0))
        return memory[0]  # [batch_size, hidden_dim]

    def decode(self, batch, memory):
        batch_size = memory.shape[0]

        # Generate future predictions
        decoder_input = memory.unsqueeze(0).repeat(self.num_frames_to_predict, 1, 1)
        transformer_output = self.transformer_decoder(decoder_input, memory.unsqueeze(0))
        
        # Predict states and hazard
        predicted_states = self.state_predictor(transformer_output.permute(1, 0, 2))
        hazard = self.hazard_predictor(transformer_output.permute(1, 0, 2)).squeeze(-1)

        return predicted_states, hazard

    def forward(self, batch):
        encoded_state = self.encode(batch)
        predicted_states, hazard = self.decode(batch, encoded_state)

        return {
            "encoded_state": encoded_state,
            "predictions": predicted_states,
            "hazard": hazard
        }

    def compute_loss(self, batch, model_output):
        predictions = model_output['predictions']
        hazard = model_output['hazard']
        target_states = batch['next_observations']
        target_done = batch['dones'][:, -self.num_frames_to_predict:]

        # Create mask for valid (not done) timesteps
        valid_mask = torch.cumprod(1 - target_done.int(), dim=1)

        # MSE loss for state predictions
        mse_loss = F.mse_loss(
            predictions * valid_mask.unsqueeze(-1), 
            target_states * valid_mask.unsqueeze(-1),
            reduction='none'
        ).mean()

        # Survival analysis loss
        cumulative_hazard = torch.cumsum(F.softplus(hazard), dim=1)
        not_done_mask = 1 - target_done.float()
        survival_likelihood = torch.exp(-cumulative_hazard) * not_done_mask + (1 - torch.exp(-cumulative_hazard)) * target_done.float()
        survival_loss = -torch.log(survival_likelihood + 1e-8).mean()

        # Combine losses
        total_loss = mse_loss + self.cfg.models.PredictiveModelV9M3.survival_loss_weight * survival_loss

        loss_components = {
            'mse_loss': mse_loss.item(),
            'survival_loss': survival_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_components

    def predict_done_probability(self, hazard):
        cumulative_hazard = torch.cumsum(F.softplus(hazard), dim=1)
        survival_probability = torch.exp(-cumulative_hazard)
        return 1 - survival_probability