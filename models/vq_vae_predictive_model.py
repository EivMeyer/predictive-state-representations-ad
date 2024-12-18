import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel
from models.autoencoder_model_v1 import AutoEncoderModelV1
from models.autoencoder_model_v0 import AutoEncoderModelV0
from models.loss_functions import VQVAELoss
from utils.file_utils import find_model_path
from typing import Dict, Any
from copy import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.ema_w.data.normal_()
        self.decay = decay

    def forward(self, inputs):
        # Compute distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.embedding.num_embeddings * 1e-5) * n)
            
            dw = torch.matmul(encodings.t(), inputs)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity

class VQVAEPredictiveModel(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg, 
                 num_embeddings=512, embedding_dim=64, commitment_cost=0.25, 
                 pretrained_model_path=None, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)
        
        pretrained_model_path = find_model_path(cfg.project_dir, cfg.models.VQVAEPredictiveModel.pretrained_model_path) if pretrained_model_path is None else pretrained_model_path

        assert pretrained_model_path is not None, "Pretrained model path must be provided"

        self.autoencoder = AutoEncoderModelV0(obs_shape, action_dim, ego_state_dim, cfg)
        self.autoencoder.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'], strict=False)
        self.autoencoder.eval()
        
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        self.vq_vae = VQVAE(num_embeddings, embedding_dim, commitment_cost)
        
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=self.hidden_dim, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=self.hidden_dim, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)
        
        self.fc_encoder = nn.Linear(self.hidden_dim, embedding_dim)
        self.fc_decoder = nn.Linear(embedding_dim, embedding_dim)

        self.loss_function = VQVAELoss(commitment_cost=commitment_cost)

    def encode(self, batch):
        observations = batch['observations']
        ego_states = batch['ego_states']

        batch_size, seq_len, channels, height, width = observations.shape
        observations = observations.view(-1, channels, height, width)
        ego_states = ego_states.view(-1, ego_states.shape[-1])

        flattened_temporal_batch = copy(batch)
        flattened_temporal_batch['observations'] = observations.view(batch_size * seq_len, -1, channels, height, width)
        
        with torch.no_grad():
            encoded_latents = self.autoencoder.encode(flattened_temporal_batch)
            if isinstance(encoded_latents, tuple):
                encoded_latents = encoded_latents[0]
        
        encoded_latents = encoded_latents.view(batch_size, seq_len, -1)
        encoded_latents = self.fc_encoder(encoded_latents)
        
        # Add positional encoding
        encoded_latents = self.pos_encoder(encoded_latents.permute(1, 0, 2))
        
        # Apply transformer encoder
        memory = self.transformer_encoder(encoded_latents)
        
        return memory[-1]  # Return only the last memory state

    def decode(self, batch, memory):
        batch_size = memory.shape[0]
        
        # Prepare decoder input
        decoder_input = self.pos_encoder(memory.unsqueeze(0).repeat(self.num_frames_to_predict, 1, 1))
        
        # Generate future frame predictions
        output = self.transformer_decoder(decoder_input, memory.unsqueeze(0))
        
        # Apply final linear layer
        predicted_latents = self.fc_decoder(output.permute(1, 0, 2))
        
        return predicted_latents

    def forward(self, batch):
        observations = batch['observations']

        encoded_state = self.encode(batch)
        vq_loss, quantized, perplexity = self.vq_vae(encoded_state)
        predicted_latents = self.decode(batch, quantized)

        predictions = self.autoencoder.decode(batch, predicted_latents.view(-1, self.hidden_dim)).view_as(observations)

        return {
            "predicted_latents": predicted_latents,
            "encoded_state": encoded_state,
            "quantized": quantized,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "predictions": predictions
        }

    def compute_loss(self, batch, model_output):
        predicted_latents = model_output["predicted_latents"]
        encoded_state = model_output["encoded_state"]
        vq_loss = model_output["vq_loss"]
        
        # Get target latents using the pretrained single-step model
        with torch.no_grad():
            target_observations = batch["next_observations"]
            target_ego_states = batch["ego_states"][:, -self.num_frames_to_predict:, :]
            batch_size, seq_len, channels, height, width = target_observations.shape
            target_observations = target_observations.view(-1, 1, channels, height, width)
            target_ego_states = target_ego_states.view(-1, target_ego_states.shape[-1])
            target_batch = {
                'observations': target_observations,
                'ego_states': target_ego_states
            }
            target_latents = self.autoencoder.encode(target_batch)
            if isinstance(target_latents, tuple):
                target_latents = target_latents[0]
            target_latents = target_latents.view(batch_size, seq_len, -1)
        
        loss, loss_components = self.loss_function(predicted_latents, target_latents, encoded_state, vq_loss)
        return loss, loss_components