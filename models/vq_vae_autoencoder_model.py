import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_predictive_model import BasePredictiveModel

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VQVAEAutoEncoderModel(BasePredictiveModel):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg):
        super().__init__(obs_shape, action_dim, ego_state_dim, cfg)

        self.encoder = self._make_encoder(obs_shape)
        self.decoder = self._make_decoder(obs_shape)

        self.num_embeddings = 512  # You can adjust this
        self.embedding_dim = 64    # This should match your desired latent dimension
        self.commitment_cost = 0.25

        self.pre_vq_conv = nn.Conv2d(512, self.embedding_dim, kernel_size=1, stride=1)
        self.vq_vae = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.commitment_cost)
        self.post_vq_conv = nn.Conv2d(self.embedding_dim, 512, kernel_size=1, stride=1)

        self._initialize_weights()

    def _make_encoder(self, obs_shape):
        return nn.Sequential(
            self._make_encoder_block(obs_shape[-3], 64),
            self._make_encoder_block(64, 128),
            self._make_encoder_block(128, 256),
            self._make_encoder_block(256, 512),
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
        return nn.Sequential(
            self._make_decoder_block(512, 256),
            self._make_decoder_block(256, 128),
            self._make_decoder_block(128, 64),
            nn.ConvTranspose2d(64, obs_shape[-3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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
        z = self.encoder(observation)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self.vq_vae(z)
        return quantized, loss, perplexity, encodings

    def decode(self, batch, quantized):
        z = self.post_vq_conv(quantized)
        return self.decoder(z)

    def forward(self, batch):
        quantized, vq_loss, perplexity, encodings = self.encode(batch)
        reconstruction = self.decode(batch, quantized).unsqueeze(1)  # Add sequence length dimension
        return {
            'predictions': reconstruction,
            'encoded_state': quantized,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'encodings': encodings
        }

    def compute_loss(self, batch, model_output):
        reconstruction = model_output['predictions']
        vq_loss = model_output['vq_loss']
        target_observation = batch['observations'][:, (-1,)]  # Take only the last observation

        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(reconstruction, target_observation, reduction='mean')

        # Total loss
        total_loss = mse_loss + vq_loss

        loss_components = {
            'mse': mse_loss.item(),
            'vq_loss': vq_loss.item(),
            'perplexity': model_output['perplexity'].item(),
        }

        return total_loss, loss_components